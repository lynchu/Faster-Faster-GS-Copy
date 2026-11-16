#pragma once

#include <torch/extension.h>

// Faster-GS+LazyAdam: exact lazy Adam for Gaussians that are invisible in the current view.
// Zero-gradient Adam steps are applied in closed form when a Gaussian becomes visible again (or at a flush):
//   m <- beta1^k m,  v <- beta2^k v,  theta <- theta - (m / sqrt(v)) * (F[t0] - gamma^k F[t0 + k]),  gamma = beta1 / sqrt(beta2)
// where F[t] = gamma * (f[t + 1] + F[t + 1]) and f[s] = lr_s * sqrt(1 - beta2^s) / (1 - beta1^s) (precomputed in float64).
// If epsilon is not negligible against sqrt(v) (tiny gradients), the skipped steps are replayed with the exact dense arithmetic.
namespace faster_gs::adam {

    // regular Adam step t for visible Gaussians, preceded by the closed-form catch-up of steps last_step + 1 .. t - 1
    void lazy_adam_step_wrapper(
        const torch::Tensor& param_grad,
        torch::Tensor& param,
        torch::Tensor& exp_avg,
        torch::Tensor& exp_avg_sq,
        const torch::Tensor& visibility,  // uint8 [n_gaussians]
        const torch::Tensor& last_step,   // int32 [n_gaussians]
        const torch::Tensor& f_table,     // float32 [T + 2], the F[t] sums of this parameter group (computed in float64)
        const torch::Tensor& step_size_table,                  // float32 [T + 2], lr_s / (1 - beta1^s) as in adam_step
        const torch::Tensor& bias_correction2_sqrt_rcp_table,  // float32 [T + 2], 1 / sqrt(1 - beta2^s) as in adam_step
        const int step_count,
        const double learning_rate,
        const double beta1,
        const double beta2,
        const double epsilon);

    // catch-up of steps last_step + 1 .. target_step for every Gaussian (no regular step)
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
        const double epsilon);

    // last_step[i] = step for every visible Gaussian (called once per iteration after all parameter groups)
    void lazy_adam_mark_wrapper(
        const torch::Tensor& visibility,
        torch::Tensor& last_step,
        const int step_count);

}
