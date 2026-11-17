#include <cmath>
#include "rasterization_api.h"
#include "forward.h"
#include "backward.h"
#include "inference.h"
#include "pruning_scores.h"
#include "torch_utils.h"
#include "rasterization_config.h"
#include "helper_math.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>
#include <functional>
#include <tuple>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
faster_gs::rasterization::forward_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    const at::cuda::CUDAGuard device_guard(means.device());

    // all optimizable tensors must be passed as contiguous CUDA float tensors
    CHECK_INPUT(config::debug, means, "means");
    CHECK_INPUT(config::debug, scales, "scales");
    CHECK_INPUT(config::debug, rotations, "rotations");
    CHECK_INPUT(config::debug, opacities, "opacities");
    CHECK_INPUT(config::debug, sh_coefficients_0, "sh_coefficients_0");
    CHECK_INPUT(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = torch::empty({3, height, width}, float_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    torch::Tensor bucket_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);
    const std::function<char*(size_t)> resize_bucket_buffers = resize_function_wrapper(bucket_buffers);

    auto [n_instances, n_buckets, instance_primitive_indices_selector] = forward(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        resize_bucket_buffers,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        at::cuda::getCurrentCUDAStream()
    );

    return {
        image,
        primitive_buffers, tile_buffers, instance_buffers, bucket_buffers,
        n_instances, n_buckets, instance_primitive_indices_selector
    };
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
faster_gs::rasterization::backward_wrapper(
    torch::Tensor& densification_info,
    torch::Tensor& visibility,
    const torch::Tensor& grad_image,
    const torch::Tensor& image,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& primitive_buffers,
    const torch::Tensor& tile_buffers,
    const torch::Tensor& instance_buffers,
    const torch::Tensor& bucket_buffers,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const int n_instances,
    const int n_buckets,
    const int instance_primitive_indices_selector,
    // Faster-GS+FusedShAdam: optimizer state of the SH rest coefficients (empty exp_avg: not fused)
    const torch::Tensor& fused_exp_avg,
    const torch::Tensor& fused_exp_avg_sq,
    const torch::Tensor& fused_last_step,
    const torch::Tensor& fused_f_table,
    const torch::Tensor& fused_step_size_table,
    const torch::Tensor& fused_bias_correction2_sqrt_rcp_table,
    const int fused_step,
    const double fused_learning_rate,
    const double fused_beta1,
    const double fused_beta2,
    const double fused_epsilon)
{
    const at::cuda::CUDAGuard device_guard(means.device());

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    torch::Tensor grad_means = torch::empty({n_primitives, 3}, float_options);
    torch::Tensor grad_scales = torch::empty({n_primitives, 3}, float_options);
    torch::Tensor grad_rotations = torch::empty({n_primitives, 4}, float_options);
    torch::Tensor grad_opacities = torch::zeros({n_primitives, 1}, float_options);
    torch::Tensor grad_sh_coefficients_0 = torch::zeros({n_primitives, 1, 3}, float_options);
    // Faster-GS+FusedShAdam: the SH rest coefficients are updated in place inside the backward pass
    const bool fuse_sh_adam = fused_exp_avg.numel() > 0;
    faster_gs::adam::FusedShAdamArgs fused_sh_adam;
    if (fuse_sh_adam) {
        TORCH_CHECK(fused_exp_avg.numel() == sh_coefficients_rest.numel() && fused_exp_avg_sq.numel() == sh_coefficients_rest.numel(),
                    "fused SH Adam: optimizer state does not match the SH rest coefficients");
        TORCH_CHECK(fused_last_step.size(0) == n_primitives, "fused SH Adam: last_step has the wrong size");
        // exactly the per-step constants and stochastic-rounding seed of lazy_adam_step_wrapper
        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(fused_beta1, fused_step));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(fused_beta2, fused_step));
        const double step_size = fused_learning_rate * bias_correction1_rcp;
        fused_sh_adam.param = sh_coefficients_rest.data_ptr<float>();
        fused_sh_adam.exp_avg = fused_exp_avg.data_ptr();
        fused_sh_adam.exp_avg_sq = fused_exp_avg_sq.data_ptr();
        fused_sh_adam.m_half = fused_exp_avg.scalar_type() == torch::kBFloat16;
        fused_sh_adam.v_half = fused_exp_avg_sq.scalar_type() == torch::kBFloat16;
        fused_sh_adam.last_step = fused_last_step.data_ptr<int>();
        fused_sh_adam.f_table = fused_f_table.data_ptr<float>();
        fused_sh_adam.step_size_table = fused_step_size_table.data_ptr<float>();
        fused_sh_adam.bias_correction2_sqrt_rcp_table = fused_bias_correction2_sqrt_rcp_table.data_ptr<float>();
        fused_sh_adam.step = fused_step;
        fused_sh_adam.step_size = static_cast<float>(step_size);
        fused_sh_adam.beta1 = static_cast<float>(fused_beta1);
        fused_sh_adam.beta2 = static_cast<float>(fused_beta2);
        fused_sh_adam.epsilon = static_cast<float>(fused_epsilon);
        fused_sh_adam.bias_correction2_sqrt_rcp = static_cast<float>(bias_correction2_sqrt_rcp);
        fused_sh_adam.log2_gamma = static_cast<float>(std::log2(fused_beta1 / std::sqrt(fused_beta2)));
        fused_sh_adam.log2_beta1 = static_cast<float>(std::log2(fused_beta1));
        fused_sh_adam.log2_beta2 = static_cast<float>(std::log2(fused_beta2));
        fused_sh_adam.seed = faster_gs::adam::sr_seed(fused_step, static_cast<int>(sh_coefficients_rest.numel()), 1);
    }
    torch::Tensor grad_sh_coefficients_rest = fuse_sh_adam ? torch::empty({0}, float_options) : torch::empty({n_primitives, total_sh_bases, 3}, float_options);
    torch::Tensor grad_mean2d_helper = torch::zeros({n_primitives, 2}, float_options);
    torch::Tensor grad_conic_helper = torch::zeros({3, n_primitives}, float_options);

    const bool update_densification_info = densification_info.size(0) > 0;
    const bool compute_abs_grad = densification_info.size(0) == 3;
    torch::Tensor grad_mean2d_abs_helper;
    if (compute_abs_grad) grad_mean2d_abs_helper = torch::zeros({n_primitives, 2}, float_options);

    backward(
        grad_image.contiguous().data_ptr<float>(),
        image.data_ptr<float>(),
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        reinterpret_cast<char*>(primitive_buffers.data_ptr()),
        reinterpret_cast<char*>(tile_buffers.data_ptr()),
        reinterpret_cast<char*>(instance_buffers.data_ptr()),
        reinterpret_cast<char*>(bucket_buffers.data_ptr()),
        reinterpret_cast<float3*>(grad_means.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_scales.data_ptr<float>()),
        reinterpret_cast<float4*>(grad_rotations.data_ptr<float>()),
        reinterpret_cast<float*>(grad_opacities.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(grad_sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float2*>(grad_mean2d_helper.data_ptr<float>()),
        grad_conic_helper.data_ptr<float>(),
        compute_abs_grad ? reinterpret_cast<float2*>(grad_mean2d_abs_helper.data_ptr<float>()) : nullptr,
        update_densification_info ? densification_info.data_ptr<float>() : nullptr,
        visibility.size(0) == n_primitives ? visibility.data_ptr<uint8_t>() : nullptr,
        fused_sh_adam,
        n_primitives,
        n_instances,
        n_buckets,
        instance_primitive_indices_selector,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        proper_antialiasing,
        at::cuda::getCurrentCUDAStream()
    );

    // an undefined tensor is returned to Python as None (no gradient for the fused SH rest coefficients)
    return {grad_means, grad_scales, grad_rotations, grad_opacities, grad_sh_coefficients_0, fuse_sh_adam ? torch::Tensor() : grad_sh_coefficients_rest};
}

torch::Tensor
faster_gs::rasterization::inference_wrapper(
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing,
    const bool to_chw,
    const bool clamp_output)
{
    const at::cuda::CUDAGuard device_guard(means.device());

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor image = to_chw ? torch::empty({3, height, width}, float_options) : torch::empty({height, width, 3}, float_options);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);

    inference(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        image.data_ptr<float>(),
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        to_chw,
        clamp_output,
        at::cuda::getCurrentCUDAStream()
    );

    return image;
}

void
faster_gs::rasterization::pruning_scores_wrapper(
    torch::Tensor& scores,
    const torch::Tensor& means,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coefficients_0,
    const torch::Tensor& sh_coefficients_rest,
    const torch::Tensor& w2c,
    const torch::Tensor& cam_position,
    const torch::Tensor& bg_color,
    const int active_sh_bases,
    const int width,
    const int height,
    const float focal_x,
    const float focal_y,
    const float center_x,
    const float center_y,
    const float near_plane,
    const float far_plane,
    const bool proper_antialiasing)
{
    const at::cuda::CUDAGuard device_guard(means.device());

    const int n_primitives = means.size(0);
    const int total_sh_bases = sh_coefficients_rest.size(1);
    const torch::TensorOptions byte_options = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
    torch::Tensor primitive_buffers = torch::empty({0}, byte_options);
    torch::Tensor tile_buffers = torch::empty({0}, byte_options);
    torch::Tensor instance_buffers = torch::empty({0}, byte_options);
    const std::function<char*(size_t)> resize_primitive_buffers = resize_function_wrapper(primitive_buffers);
    const std::function<char*(size_t)> resize_tile_buffers = resize_function_wrapper(tile_buffers);
    const std::function<char*(size_t)> resize_instance_buffers = resize_function_wrapper(instance_buffers);

    pruning_scores(
        resize_primitive_buffers,
        resize_tile_buffers,
        resize_instance_buffers,
        reinterpret_cast<float3*>(means.data_ptr<float>()),
        reinterpret_cast<float3*>(scales.data_ptr<float>()),
        reinterpret_cast<float4*>(rotations.data_ptr<float>()),
        opacities.data_ptr<float>(),
        reinterpret_cast<float3*>(sh_coefficients_0.data_ptr<float>()),
        reinterpret_cast<float3*>(sh_coefficients_rest.data_ptr<float>()),
        reinterpret_cast<float4*>(w2c.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(cam_position.contiguous().data_ptr<float>()),
        reinterpret_cast<float3*>(bg_color.contiguous().data_ptr<float>()),
        scores.data_ptr<float>(),
        n_primitives,
        active_sh_bases,
        total_sh_bases,
        width,
        height,
        focal_x,
        focal_y,
        center_x,
        center_y,
        near_plane,
        far_plane,
        proper_antialiasing,
        at::cuda::getCurrentCUDAStream()
    );
}
