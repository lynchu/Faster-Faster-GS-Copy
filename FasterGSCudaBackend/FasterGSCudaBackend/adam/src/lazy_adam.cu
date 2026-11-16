#include "lazy_adam.h"
#include "adam_config.h"
#include "torch_utils.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>

namespace faster_gs::adam {

    // closed-form application of k >= 1 zero-gradient Adam steps following step t0
    __device__ __forceinline__ void catch_up(
        float& param, float& moment1, float& moment2,
        const int t0, const int k,
        const double* __restrict__ f_table,
        const double log_gamma, const float log2_beta1, const float log2_beta2)
    {
        if (moment2 > 0.0f) {
            const double displacement_scale = f_table[t0] - exp(static_cast<double>(k) * log_gamma) * f_table[t0 + k];
            param -= static_cast<float>(static_cast<double>(moment1) * rsqrt(static_cast<double>(moment2)) * displacement_scale);
        }
        moment1 *= exp2f(static_cast<float>(k) * log2_beta1);
        moment2 *= exp2f(static_cast<float>(k) * log2_beta2);
    }

    __global__ void lazy_adam_step_cu(
        const float* __restrict__ param_grad,
        float* __restrict__ param,
        float* __restrict__ exp_avg,
        float* __restrict__ exp_avg_sq,
        const uint8_t* __restrict__ visibility,
        const int* __restrict__ last_step,
        const double* __restrict__ f_table,
        const int n_elements,
        const int n_elements_per_gaussian,
        const int step,
        const float step_size,
        const float beta1,
        const float beta2,
        const float epsilon,
        const float bias_correction2_sqrt_rcp,
        const double log_gamma,
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
        if (k > 0) catch_up(p, m1, m2, t0, k, f_table, log_gamma, log2_beta1, log2_beta2);
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
        const double* __restrict__ f_table,
        const int n_elements,
        const int n_elements_per_gaussian,
        const int target_step,
        const double log_gamma,
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
        catch_up(p, m1, m2, t0, k, f_table, log_gamma, log2_beta1, log2_beta2);
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

    void lazy_adam_step_wrapper(
        const torch::Tensor& param_grad,
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const torch::Tensor& visibility,
        const torch::Tensor& last_step,
        const torch::Tensor& f_table,
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
            f_table.data_ptr<double>(),
            n_elements,
            n_elements / n_gaussians,
            step_count,
            static_cast<float>(step_size),
            static_cast<float>(beta1),
            static_cast<float>(beta2),
            static_cast<float>(epsilon),
            static_cast<float>(bias_correction2_sqrt_rcp),
            std::log(beta1 / std::sqrt(beta2)),
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
        const int target_step,
        const double beta1,
        const double beta2)
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
            f_table.data_ptr<double>(),
            n_elements,
            n_elements / n_gaussians,
            target_step,
            std::log(beta1 / std::sqrt(beta2)),
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
