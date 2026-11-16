#pragma once

#include <torch/extension.h>

// Faster-GS+FusedLoss: L = w_l1 * mean|x - y| + w_dssim * (1 - mean(ssim_map)) as one autograd node
namespace faster_gs::loss {

    // returns the scalar loss (1-element tensor); sums are accumulated with one reduction kernel
    torch::Tensor l1_dssim_forward_wrapper(
        const torch::Tensor& image,
        const torch::Tensor& target,
        const torch::Tensor& ssim_map,
        const double weight_l1,
        const double weight_dssim);

    // grad_ssim (from fusedssim_backward) += grad_out * weight_l1 * sign(image - target) / n, in place
    void l1_grad_combine_wrapper(
        torch::Tensor& grad_ssim,
        const torch::Tensor& image,
        const torch::Tensor& target,
        const torch::Tensor& grad_out,
        const double weight_l1);

}
