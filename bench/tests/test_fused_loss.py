"""Equivalence of Faster-GS+FusedLoss with the generic loss path (0.8 * L1 + 0.2 * (1 - fused SSIM))."""
import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
sys.path.insert(0, os.environ.get('FFGS_BACKEND', os.path.join(os.path.dirname(__file__), '..', '..', 'FasterGSCudaBackend')))
import torch
import torch.nn.functional as F
from fused_ssim import fused_ssim
from FasterGSCudaBackend.torch_bindings.fused_loss import fused_l1_dssim

torch.manual_seed(0)
for H, W in [(840, 1297), (1039, 1559)]:
    gt = torch.rand(3, H, W, device='cuda')
    img0 = (gt + 0.1 * torch.randn_like(gt)).clamp(0, 1.2)
    x_ref = img0.clone().requires_grad_(True)
    loss_ref = F.l1_loss(x_ref, gt) * 0.8 + (1.0 - fused_ssim(x_ref[None], gt[None])) * 0.2
    loss_ref.backward()
    x = img0.clone().requires_grad_(True)
    loss = fused_l1_dssim(x, gt, 0.8, 0.2)
    loss.backward()
    d = (x.grad - x_ref.grad).abs()
    print(f'{H}x{W}: loss ref {loss_ref.item():.8f} fused {loss.item():.8f} (|diff| {abs(loss_ref.item() - loss.item()):.2e}); '
          f'grad max |diff| {d.max().item():.3e} (grad scale {x_ref.grad.abs().mean().item():.3e}), bitwise equal fraction {(d == 0).float().mean().item():.4f}')
