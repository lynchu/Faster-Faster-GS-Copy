#include "lazy_adam.h"
#include "adam_config.h"
#include "torch_utils.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "helper_math.h"

namespace faster_gs::adam {

    // zero-gradient steps beyond this window change the parameter by less than beta1^256 ~ 2e-12 of the momentum displacement
    constexpr int max_explicit_steps = 256;

    // application of k >= 1 zero-gradient Adam steps following step t0
    __device__ __forceinline__ void catch_up(
        float& param, float& moment1, float& moment2,
        const int t0, const int k,
        const float* __restrict__ f_table,
        const float* __restrict__ step_size_table,
        const float* __restrict__ bias_correction2_sqrt_rcp_table,
        const float beta1, const float beta2, const float epsilon,
        const float log2_gamma, const float log2_beta1, const float log2_beta2)
    {
        const int window = min(k, max_explicit_steps);
        // epsilon is negligible (< 1e-6 relative) if sqrt(v) stays above 1e6 * epsilon over the significant window
        const float sqrt_v_window_end = sqrtf(moment2 * exp2f(static_cast<float>(window) * log2_beta2));
        if (sqrt_v_window_end >= 1e6f * epsilon) {
            // closed form: theta -= m / sqrt(v) * sum_j f(t0 + j) gamma^j (float32 suffices: the relative error of the
            // displacement is ~1e-6, i.e., orders of magnitude below the float32 spacing of the parameter; FP64 would be
            // 64x slower on this GPU class)
            const float displacement_scale = f_table[t0] - exp2f(static_cast<float>(k) * log2_gamma) * f_table[t0 + k];
            param -= moment1 * rsqrtf(moment2) * displacement_scale;
            moment1 *= exp2f(static_cast<float>(k) * log2_beta1);
            moment2 *= exp2f(static_cast<float>(k) * log2_beta2);
        } else {
            // tiny second moment: replay the zero-gradient steps with exactly the arithmetic of adam_step_cu
            for (int j = 1; j <= window; ++j) {
                const int s = t0 + j;
                moment1 = fmaf(beta1, moment1, 0.0f);
                moment2 = fmaf(beta2, moment2, 0.0f);
                const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp_table[s] + epsilon;
                param -= step_size_table[s] * moment1 / denom;
            }
            if (k > window) {
                moment1 *= exp2f(static_cast<float>(k - window) * log2_beta1);
                moment2 *= exp2f(static_cast<float>(k - window) * log2_beta2);
            }
        }
    }

    __global__ void lazy_adam_step_cu(
        const float* __restrict__ param_grad,
        float* __restrict__ param,
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const uint8_t* __restrict__ visibility,
        const int* __restrict__ last_step,
        const float* __restrict__ f_table,
        const float* __restrict__ step_size_table,
        const float* __restrict__ bias_correction2_sqrt_rcp_table,
        const int n_elements,
        const int n_elements_per_gaussian,
        const int step,
        const float step_size,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float bias_correction2_sqrt_rcp,
        const float log2_gamma,
        const float log2_beta1,
        const float log2_beta2)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements) return;
        const int gaussian_idx = idx / n_elements_per_gaussian;
        if (visibility[gaussian_idx] == 0) return;
        float p = param[idx];
        float m1 = exp_avg[idx];
        float m2 = exp_avg_sq[idx];
        const int t0 = last_step[gaussian_idx];
        const int k = step - 1 - t0;
        if (k > 0) catch_up(p, m1, m2, t0, k, f_table, step_size_table, bias_correction2_sqrt_rcp_table, beta1, beta2, epsilon, log2_gamma, log2_beta1, log2_beta2);
        // regular step, identical to adam_step_cu
        const float grad = param_grad[idx];
        const float grad_sq = grad * grad;
        const float moment1 = fmaf(beta1, m1 - grad, grad);
        const float moment2 = fmaf(beta2, m2 - grad_sq, grad_sq);
        const float denom = sqrtf(moment2) * bias_correction2_sqrt_rcp + epsilon;
        param[idx] = p - step_size * moment1 / denom;
        exp_avg[idx] = moment1;
        exp_avg_sq[idx] = moment2;
    }

    __global__ void lazy_adam_flush_cu(
        float* __restrict__ param,
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const int* __restrict__ last_step,
        const float* __restrict__ f_table,
        const float* __restrict__ step_size_table,
        const float* __restrict__ bias_correction2_sqrt_rcp_table,
        const int n_elements,
        const int n_elements_per_gaussian,
        const int target_step,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float log2_gamma,
        const float log2_beta1,
        const float log2_beta2)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements) return;
        const int gaussian_idx = idx / n_elements_per_gaussian;
        const int t0 = last_step[gaussian_idx];
        const int k = target_step - t0;
        if (k <= 0) return;
        float p = param[idx];
        float m1 = exp_avg[idx];
        float m2 = exp_avg_sq[idx];
        catch_up(p, m1, m2, t0, k, f_table, step_size_table, bias_correction2_sqrt_rcp_table, beta1, beta2, epsilon, log2_gamma, log2_beta1, log2_beta2);
        param[idx] = p;
        exp_avg[idx] = m1;
        exp_avg_sq[idx] = m2;
    }

    __global__ void lazy_adam_mark_cu(
        const uint8_t* __restrict__ visibility,
        int* __restrict__ last_step,
        const int n_gaussians,
        const int step)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_gaussians) return;
        if (visibility[idx] != 0) last_step[idx] = step;
    }

    // ----- Faster-GS+LazyAdam-Exact -----

    __device__ __forceinline__ void quaternion_to_rotation(const float4 q, float* R) {
        const float r = q.x, x = q.y, y = q.z, z = q.w;
        const float norm_sq = r * r + x * x + y * y + z * z;
        const float s = norm_sq > 0.0f ? 1.0f / norm_sq : 0.0f;
        R[0] = 1.0f - 2.0f * (y * y + z * z) * s; R[1] = 2.0f * (x * y - r * z) * s; R[2] = 2.0f * (x * z + r * y) * s;
        R[3] = 2.0f * (x * y + r * z) * s; R[4] = 1.0f - 2.0f * (x * x + z * z) * s; R[5] = 2.0f * (y * z - r * x) * s;
        R[6] = 2.0f * (x * z - r * y) * s; R[7] = 2.0f * (y * z + r * x) * s; R[8] = 1.0f - 2.0f * (x * x + y * y) * s;
    }

    __global__ void lazy_candidates_cu(
        const float3* __restrict__ means,
        const float3* __restrict__ scales,
        const float4* __restrict__ rotations,
        const float4* __restrict__ w2c,
        uint8_t* __restrict__ candidates,
        const int n,
        const float width, const float height,
        const float focal_x, const float focal_y,
        const float center_x, const float center_y,
        const float near_plane, const float far_plane,
        const float extent_scale, const float extent_pad)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        const float3 m = means[idx];
        const float4 r1 = w2c[0], r2 = w2c[1], r3 = w2c[2];
        const float depth = r3.x * m.x + r3.y * m.y + r3.z * m.z + r3.w;
        // widened depth range (the real test is near_plane <= depth <= far_plane)
        if (depth < 0.5f * near_plane || depth > 2.0f * far_plane) { candidates[idx] = 0; return; }
        const float x = (r1.x * m.x + r1.y * m.y + r1.z * m.z + r1.w) / depth;
        const float y = (r2.x * m.x + r2.y * m.y + r2.z * m.z + r2.w) / depth;
        // 3d covariance
        const float3 s = scales[idx];
        const float vx = expf(2.0f * s.x), vy = expf(2.0f * s.y), vz = expf(2.0f * s.z);
        float R[9];
        quaternion_to_rotation(rotations[idx], R);
        const float c11 = R[0] * R[0] * vx + R[1] * R[1] * vy + R[2] * R[2] * vz;
        const float c12 = R[0] * R[3] * vx + R[1] * R[4] * vy + R[2] * R[5] * vz;
        const float c13 = R[0] * R[6] * vx + R[1] * R[7] * vy + R[2] * R[8] * vz;
        const float c22 = R[3] * R[3] * vx + R[4] * R[4] * vy + R[5] * R[5] * vz;
        const float c23 = R[3] * R[6] * vx + R[4] * R[7] * vy + R[5] * R[8] * vz;
        const float c33 = R[6] * R[6] * vx + R[7] * R[7] * vy + R[8] * R[8] * vz;
        // projection as in preprocess (clipped jacobian)
        const float xc = fminf(fmaxf(x, (-0.15f * width - center_x) / focal_x), (1.15f * width - center_x) / focal_x);
        const float yc = fminf(fmaxf(y, (-0.15f * height - center_y) / focal_y), (1.15f * height - center_y) / focal_y);
        const float j11 = focal_x / depth, j13 = -j11 * xc, j22 = focal_y / depth, j23 = -j22 * yc;
        const float a1 = j11 * r1.x + j13 * r3.x, a2 = j11 * r1.y + j13 * r3.y, a3 = j11 * r1.z + j13 * r3.z;
        const float b1 = j22 * r2.x + j23 * r3.x, b2 = j22 * r2.y + j23 * r3.y, b3 = j22 * r2.z + j23 * r3.z;
        const float cov_xx = a1 * (a1 * c11 + a2 * c12 + a3 * c13) + a2 * (a1 * c12 + a2 * c22 + a3 * c23) + a3 * (a1 * c13 + a2 * c23 + a3 * c33) + 0.3f;
        const float cov_yy = b1 * (b1 * c11 + b2 * c12 + b3 * c13) + b2 * (b1 * c12 + b2 * c22 + b3 * c23) + b3 * (b1 * c13 + b2 * c23 + b3 * c33) + 0.3f;
        // extent for opacity 1 (largest possible) at the 1/255 cutoff, inflated by extent_scale plus extent_pad pixels
        constexpr float cutoff = 2.0f * 5.54126354516f;
        const float ex = extent_scale * sqrtf(fmaxf(cov_xx, 0.0f) * cutoff) + extent_pad;
        const float ey = extent_scale * sqrtf(fmaxf(cov_yy, 0.0f) * cutoff) + extent_pad;
        const float mx = x * focal_x + center_x, my = y * focal_y + center_y;
        const bool overlaps = (mx + ex >= 0.0f) && (mx - ex <= width) && (my + ey >= 0.0f) && (my - ey <= height);
        candidates[idx] = (overlaps || !isfinite(ex) || !isfinite(ey)) ? 1 : 0;
    }

    __global__ void lazy_adam_flush_masked_cu(
        float* __restrict__ param,
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const uint8_t* __restrict__ mask,
        const int* __restrict__ last_step,
        const float* __restrict__ f_table,
        const float* __restrict__ step_size_table,
        const float* __restrict__ bias_correction2_sqrt_rcp_table,
        const int n_elements,
        const int n_elements_per_gaussian,
        const int target_step,
        const float beta1, const float beta2, const float epsilon,
        const float log2_gamma, const float log2_beta1, const float log2_beta2)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n_elements) return;
        const int gaussian_idx = idx / n_elements_per_gaussian;
        if (mask[gaussian_idx] == 0) return;
        const int t0 = last_step[gaussian_idx];
        const int k = target_step - t0;
        if (k <= 0) return;
        float p = param[idx], m1 = exp_avg[idx], m2 = exp_avg_sq[idx];
        catch_up(p, m1, m2, t0, k, f_table, step_size_table, bias_correction2_sqrt_rcp_table, beta1, beta2, epsilon, log2_gamma, log2_beta1, log2_beta2);
        param[idx] = p;
        exp_avg[idx] = m1;
        exp_avg_sq[idx] = m2;
    }

    __global__ void lazy_mark_masked_cu(const uint8_t* __restrict__ mask, int* __restrict__ last_step, const int n, const int target_step) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        if (mask[idx] != 0 && last_step[idx] < target_step) last_step[idx] = target_step;
    }

    __global__ void lazy_count_stale_visible_cu(const uint8_t* __restrict__ visibility, const int* __restrict__ last_step, const int n,
                                                const int step_before_forward, unsigned long long* __restrict__ count) {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n) return;
        if (visibility[idx] != 0 && last_step[idx] < step_before_forward) atomicAdd(count, 1ull);
    }

    void lazy_candidates_wrapper(
        const torch::Tensor& means, const torch::Tensor& scales, const torch::Tensor& rotations, const torch::Tensor& w2c,
        torch::Tensor& candidates, const int width, const int height, const float focal_x, const float focal_y,
        const float center_x, const float center_y, const float near_plane, const float far_plane,
        const float extent_scale, const float extent_pad)
    {
        const at::cuda::CUDAGuard device_guard(means.device());
        const int n = means.size(0);
        if (n == 0) return;
        const auto w2c_c = w2c.contiguous();
        lazy_candidates_cu<<<div_round_up(n, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const float3*>(means.data_ptr<float>()), reinterpret_cast<const float3*>(scales.data_ptr<float>()),
            reinterpret_cast<const float4*>(rotations.data_ptr<float>()), reinterpret_cast<const float4*>(w2c_c.data_ptr<float>()),
            candidates.data_ptr<uint8_t>(), n, static_cast<float>(width), static_cast<float>(height),
            focal_x, focal_y, center_x, center_y, near_plane, far_plane, extent_scale, extent_pad);
        CHECK_CUDA(config::debug, "lazy_candidates")
    }

    void lazy_adam_flush_masked_wrapper(
        torch::Tensor& param, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq, const torch::Tensor& mask, const torch::Tensor& last_step,
        const torch::Tensor& f_table, const torch::Tensor& step_size_table, const torch::Tensor& bias_correction2_sqrt_rcp_table,
        const int target_step, const double beta1, const double beta2, const double epsilon)
    {
        const at::cuda::CUDAGuard device_guard(param.device());
        const int n_elements = param.numel();
        const int n_gaussians = last_step.size(0);
        if (n_elements == 0) return;
        TORCH_CHECK(n_elements % n_gaussians == 0, "lazy_adam_flush_masked: parameter size is not a multiple of the number of Gaussians");
        lazy_adam_flush_masked_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<float>(), exp_avg.data_ptr<float>(), exp_avg_sq.data_ptr<float>(), mask.data_ptr<uint8_t>(), last_step.data_ptr<int>(),
            f_table.data_ptr<float>(), step_size_table.data_ptr<float>(), bias_correction2_sqrt_rcp_table.data_ptr<float>(),
            n_elements, n_elements / n_gaussians, target_step,
            static_cast<float>(beta1), static_cast<float>(beta2), static_cast<float>(epsilon),
            static_cast<float>(std::log2(beta1 / std::sqrt(beta2))), static_cast<float>(std::log2(beta1)), static_cast<float>(std::log2(beta2)));
        CHECK_CUDA(config::debug, "lazy_adam_flush_masked")
    }

    void lazy_mark_masked_wrapper(const torch::Tensor& mask, torch::Tensor& last_step, const int target_step) {
        const at::cuda::CUDAGuard device_guard(last_step.device());
        const int n = last_step.size(0);
        if (n == 0) return;
        lazy_mark_masked_cu<<<div_round_up(n, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            mask.data_ptr<uint8_t>(), last_step.data_ptr<int>(), n, target_step);
    }

    int64_t lazy_count_stale_visible_wrapper(const torch::Tensor& visibility, const torch::Tensor& last_step, const int step_before_forward) {
        const at::cuda::CUDAGuard device_guard(last_step.device());
        const int n = last_step.size(0);
        auto count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt64).device(last_step.device()));
        if (n == 0) return 0;
        lazy_count_stale_visible_cu<<<div_round_up(n, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            visibility.data_ptr<uint8_t>(), last_step.data_ptr<int>(), n, step_before_forward,
            reinterpret_cast<unsigned long long*>(count.data_ptr<int64_t>()));
        return count.item<int64_t>();
    }

    void lazy_adam_step_wrapper(
        const torch::Tensor& param_grad,
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const torch::Tensor& visibility,
        const torch::Tensor& last_step,
        const torch::Tensor& f_table,
        const torch::Tensor& step_size_table,
        const torch::Tensor& bias_correction2_sqrt_rcp_table,
        const int step_count,
        const double learning_rate,
        const double beta1,
        const double beta2,
        const double epsilon)
    {
        const at::cuda::CUDAGuard device_guard(param.device());
        CHECK_INPUT(config::debug, param, "param");
        CHECK_INPUT(config::debug, exp_avg, "exp_avg");
        CHECK_INPUT(config::debug, exp_avg_sq, "exp_avg_sq");
        CHECK_INPUT(config::debug, param_grad, "param_grad");
        const int n_elements = param.numel();
        const int n_gaussians = visibility.size(0);
        if (n_elements == 0) return;
        TORCH_CHECK(n_elements % n_gaussians == 0, "lazy_adam_step: parameter size is not a multiple of the number of Gaussians");
        TORCH_CHECK(last_step.size(0) == n_gaussians, "lazy_adam_step: last_step has the wrong size");
        const double bias_correction1_rcp = 1.0 / (1.0 - std::pow(beta1, step_count));
        const double bias_correction2_sqrt_rcp = 1.0 / std::sqrt(1.0 - std::pow(beta2, step_count));
        const double step_size = learning_rate * bias_correction1_rcp;
        lazy_adam_step_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            param_grad.data_ptr<float>(),
            param.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            visibility.data_ptr<uint8_t>(),
            last_step.data_ptr<int>(),
            f_table.data_ptr<float>(),
            step_size_table.data_ptr<float>(),
            bias_correction2_sqrt_rcp_table.data_ptr<float>(),
            n_elements,
            n_elements / n_gaussians,
            step_count,
            static_cast<float>(step_size),
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(epsilon),
            static_cast<float>(bias_correction2_sqrt_rcp),
            static_cast<float>(std::log2(beta1 / std::sqrt(beta2))),
            static_cast<float>(std::log2(beta1)),
            static_cast<float>(std::log2(beta2))
        );
        CHECK_CUDA(config::debug, "lazy_adam_step")
    }

    void lazy_adam_flush_wrapper(
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const torch::Tensor& last_step,
        const torch::Tensor& f_table,
        const torch::Tensor& step_size_table,
        const torch::Tensor& bias_correction2_sqrt_rcp_table,
        const int target_step,
        const double beta1,
        const double beta2,
        const double epsilon)
    {
        const at::cuda::CUDAGuard device_guard(param.device());
        const int n_elements = param.numel();
        const int n_gaussians = last_step.size(0);
        if (n_elements == 0) return;
        TORCH_CHECK(n_elements % n_gaussians == 0, "lazy_adam_flush: parameter size is not a multiple of the number of Gaussians");
        lazy_adam_flush_cu<<<div_round_up(n_elements, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            param.data_ptr<float>(),
            exp_avg.data_ptr<float>(),
            exp_avg_sq.data_ptr<float>(),
            last_step.data_ptr<int>(),
            f_table.data_ptr<float>(),
            step_size_table.data_ptr<float>(),
            bias_correction2_sqrt_rcp_table.data_ptr<float>(),
            n_elements,
            n_elements / n_gaussians,
            target_step,
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(epsilon),
            static_cast<float>(std::log2(beta1 / std::sqrt(beta2))),
            static_cast<float>(std::log2(beta1)),
            static_cast<float>(std::log2(beta2))
        );
        CHECK_CUDA(config::debug, "lazy_adam_flush")
    }

    void lazy_adam_mark_wrapper(
        const torch::Tensor& visibility,
        torch::Tensor& last_step,
        const int step_count)
    {
        const at::cuda::CUDAGuard device_guard(last_step.device());
        const int n_gaussians = last_step.size(0);
        if (n_gaussians == 0) return;
        lazy_adam_mark_cu<<<div_round_up(n_gaussians, config::block_size_adam_step), config::block_size_adam_step, 0, at::cuda::getCurrentCUDAStream()>>>(
            visibility.data_ptr<uint8_t>(),
            last_step.data_ptr<int>(),
            n_gaussians,
            step_count
        );
        CHECK_CUDA(config::debug, "lazy_adam_mark")
    }

}
