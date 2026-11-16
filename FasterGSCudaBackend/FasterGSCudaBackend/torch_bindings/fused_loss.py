import torch

from FasterGSCudaBackend import _C
from fused_ssim_cuda import fusedssim, fusedssim_backward

C1 = 0.01 ** 2
C2 = 0.03 ** 2


class _FusedL1DSSIM(torch.autograd.Function):
    """Faster-GS+FusedLoss: w_l1 * L1 + w_dssim * (1 - SSIM) as a single autograd node.

    Forward: fused SSIM map + one reduction kernel that also produces the loss value.
    Backward: fused SSIM backward + one kernel adding the L1 gradient in place.
    This replaces about 24 small kernels (and their autograd nodes) of the generic loss path with 6.
    """

    @staticmethod
    def forward(ctx, image: torch.Tensor, target: torch.Tensor, weight_l1: float, weight_dssim: float) -> torch.Tensor:
        img = (image[None] if image.dim() == 3 else image).contiguous()
        tgt = (target[None] if target.dim() == 3 else target).contiguous()
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img, tgt, True)
        loss = _C.l1_dssim_forward(img, tgt, ssim_map, weight_l1, weight_dssim)
        ctx.save_for_backward(img.detach(), tgt, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.weights = (weight_l1, weight_dssim)
        ctx.input_shape = image.shape
        return loss

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        img, tgt, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        weight_l1, weight_dssim = ctx.weights
        n = img.numel()
        # d/d ssim_map of w_dssim * (1 - mean(ssim_map))
        dL_dmap = (grad_out * (-weight_dssim / n)).expand(img.shape).contiguous()
        grad = fusedssim_backward(C1, C2, img, tgt, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        _C.l1_grad_combine(grad, img, tgt, grad_out.contiguous(), weight_l1)
        return grad.view(ctx.input_shape), None, None, None


def fused_l1_dssim(image: torch.Tensor, target: torch.Tensor, weight_l1: float, weight_dssim: float) -> torch.Tensor:
    return _FusedL1DSSIM.apply(image, target, weight_l1, weight_dssim)
