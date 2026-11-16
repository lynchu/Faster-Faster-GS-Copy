#include "fused_loss.h"
#include "utils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace faster_gs::loss {

    constexpr int block_size = 256;

    // sums |x - y| and the SSIM map in one pass; the last block turns the sums into the loss value
    __global__ void l1_dssim_forward_cu(
        const float* __restrict__ image,
        const float* __restrict__ target,
        const float* __restrict__ ssim_map,
        double* __restrict__ sums,           // [2], zero-initialized
        unsigned int* __restrict__ counter,  // [1], zero-initialized
        float* __restrict__ loss,
        const int n,
        const float weight_l1,
        const float weight_dssim)
    {
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<32>(block);
        float l1 = 0.0f, ssim = 0.0f;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            l1 += fabsf(image[i] - target[i]);
            ssim += ssim_map[i];
        }
        l1 = cg::reduce(warp, l1, cg::plus<float>());
        ssim = cg::reduce(warp, ssim, cg::plus<float>());
        __shared__ float partial[2][block_size / 32];
        if (warp.thread_rank() == 0) {
            partial[0][warp.meta_group_rank()] = l1;
            partial[1][warp.meta_group_rank()] = ssim;
        }
        block.sync();
        if (threadIdx.x == 0) {
            double l1_block = 0.0, ssim_block = 0.0;
            for (int w = 0; w < block_size / 32; ++w) {
                l1_block += partial[0][w];
                ssim_block += partial[1][w];
            }
            atomicAdd(&sums[0], l1_block);
            atomicAdd(&sums[1], ssim_block);
            __threadfence();
            if (atomicAdd(counter, 1u) == gridDim.x - 1) {
                const double l1_sum = atomicAdd(&sums[0], 0.0), ssim_sum = atomicAdd(&sums[1], 0.0);
                loss[0] = static_cast<float>(weight_l1 * (l1_sum / n) + weight_dssim * (1.0 - ssim_sum / n));
            }
        }
    }

    __global__ void l1_grad_combine_cu(
        float* __restrict__ grad,
        const float* __restrict__ image,
        const float* __restrict__ target,
        const float* __restrict__ grad_out,
        const int n,
        const float weight_l1)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n) return;
        const float d = image[i] - target[i];
        const float sign = static_cast<float>((d > 0.0f) - (d < 0.0f));
        // same factor order as autograd: (grad_out * w_l1) / n * sign
        grad[i] += sign * ((grad_out[0] * weight_l1) / static_cast<float>(n));
    }

    torch::Tensor l1_dssim_forward_wrapper(
        const torch::Tensor& image,
        const torch::Tensor& target,
        const torch::Tensor& ssim_map,
        const double weight_l1,
        const double weight_dssim)
    {
        const at::cuda::CUDAGuard device_guard(image.device());
        const int n = image.numel();
        TORCH_CHECK(target.numel() == n && ssim_map.numel() == n, "l1_dssim_forward: size mismatch");
        auto workspace = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat64).device(image.device()));
        auto loss = torch::empty({}, torch::TensorOptions().dtype(torch::kFloat32).device(image.device()));
        const int n_blocks = std::min(div_round_up(n, block_size), 1024);
        l1_dssim_forward_cu<<<n_blocks, block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            image.data_ptr<float>(), target.data_ptr<float>(), ssim_map.data_ptr<float>(),
            workspace.data_ptr<double>(), reinterpret_cast<unsigned int*>(workspace.data_ptr<double>() + 2),
            loss.data_ptr<float>(), n, static_cast<float>(weight_l1), static_cast<float>(weight_dssim));
        return loss;
    }

    void l1_grad_combine_wrapper(
        torch::Tensor& grad_ssim,
        const torch::Tensor& image,
        const torch::Tensor& target,
        const torch::Tensor& grad_out,
        const double weight_l1)
    {
        const at::cuda::CUDAGuard device_guard(image.device());
        const int n = image.numel();
        l1_grad_combine_cu<<<div_round_up(n, block_size), block_size, 0, at::cuda::getCurrentCUDAStream()>>>(
            grad_ssim.data_ptr<float>(), image.data_ptr<float>(), target.data_ptr<float>(),
            grad_out.data_ptr<float>(), n, static_cast<float>(weight_l1));
    }

}
