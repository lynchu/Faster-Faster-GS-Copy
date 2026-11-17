#pragma once

#include <torch/extension.h>
#include <tuple>

namespace faster_gs::rasterization {

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int>
    forward_wrapper(
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
        const bool proper_antialiasing);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    backward_wrapper(
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
        const double fused_epsilon);

    torch::Tensor
    inference_wrapper(
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
        const bool clamp_output);

    void
    pruning_scores_wrapper(
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
        const bool proper_antialiasing);

}
