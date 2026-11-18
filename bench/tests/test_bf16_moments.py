"""Checks that bfloat16 Adam moments follow the float32 EMA without systematic bias.

1. decay: v = 1, zero gradients, 1000 steps -> v should be 0.999^1000 = 0.368 (round-to-nearest gets stuck at 1.0)
2. drift: random gradients with a slowly varying scale; relative bias and spread of m, v and the parameter
   trajectory against a float32 reference run through the same kernel

usage: python test_bf16_moments.py <variant> [--gpu 4]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--same-dtype-only', action='store_true', help='kernels that require m and v of the same dtype')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import backend_dir
    sys.path.insert(0, str(backend_dir(args.variant)))
    import torch
    from FasterGSCudaBackend import _C

    n = 1 << 18
    lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-15

    def run(m_dtype, v_dtype, grads, v0=None):
        p = torch.zeros(n, device='cuda')
        m = torch.zeros(n, device='cuda', dtype=m_dtype)
        v = (torch.zeros(n, device='cuda') if v0 is None else v0.clone()).to(v_dtype)
        for step, g in enumerate(grads, start=1):
            _C.adam_step(g, p, m, v, step, lr, b1, b2, eps)
        return p, m.float(), v.float()

    cases = {'bf16 m, bf16 v': (torch.bfloat16, torch.bfloat16)}
    if not args.same_dtype_only:
        cases['bf16 m, fp32 v'] = (torch.bfloat16, torch.float32)

    # 1. pure decay (one step with a gradient to initialize, then zero gradients)
    g0 = torch.ones(n, device='cuda')
    zeros = torch.zeros(n, device='cuda')
    steps = [g0] + [zeros] * 1000
    _, m32, v32 = run(torch.float32, torch.float32, steps)
    for name, (md, vd) in cases.items():
        _, mh, vh = run(md, vd, steps)
        print(f'decay  {name}: v mean {vh.mean().item():.5f} (fp32 {v32.mean().item():.5f}), relative std {(vh / v32 - 1).std().item():.4f}')

    # 2. random gradients, log-scale random walk of the gradient magnitude, 3000 steps
    torch.manual_seed(0)
    scale = torch.exp(torch.randn(n, device='cuda') * 2.0)
    grads = []
    for t in range(3000):
        if t % 50 == 0:
            scale = scale * torch.exp(torch.randn(n, device='cuda') * 0.2)
        g = torch.randn(n, device='cuda') * scale
        g[torch.rand(n, device='cuda') < 0.7] = 0.0  # ~30% of the Gaussians are visible per step
        grads.append(g)
    p32, m32, v32 = run(torch.float32, torch.float32, grads)
    for name, (md, vd) in cases.items():
        ph, mh, vh = run(md, vd, grads)
        ok = v32 > 0
        rv = vh[ok] / v32[ok] - 1
        dp = (ph - p32).abs()
        print(f'random {name}: v bias {rv.mean().item():+.4f} spread {rv.std().item():.4f} | '
              f'param |diff| mean {dp.mean().item():.3e} vs |param| mean {p32.abs().mean().item():.3e}')


if __name__ == '__main__':
    main()
