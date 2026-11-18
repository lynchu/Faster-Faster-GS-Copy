"""Faster-GS+FusedShAdam against the unfused lazy step, through the real rasterizer.

A synthetic scene is trained for a few steps with LazyAdam (camera moves every step, so the visible set changes and
the in-backward update also exercises the catch-up of Gaussians that were invisible in the previous step). Runs:
fused, unfused, unfused again (the noise floor of the non-deterministic atomics in blend_backward). Reports the
largest relative difference of every parameter and of the SH-rest moments.

usage: python test_fused_sh.py <variant> [--gpu 4] [--steps 30] [--fp32]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--fp32', action='store_true', help='float32 moments (bitwise comparison of the optimizer math)')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import backend_dir
    sys.path.insert(0, str(backend_dir(args.variant)))
    import torch
    from FasterGSCudaBackend.torch_bindings.rasterization import diff_rasterize, RasterizerSettings
    from FasterGSCudaBackend.torch_bindings.adam import LazyAdam
    dev = 'cuda'
    N, W, H = 200_000, 800, 520
    names = ['means', 'scales', 'rotations', 'opacities', 'sh_coefficients_0', 'sh_coefficients_rest']

    def init():
        g = torch.Generator().manual_seed(0)
        t = {
            'means': torch.stack([torch.empty(N).uniform_(-4, 4, generator=g), torch.empty(N).uniform_(-3, 3, generator=g), torch.empty(N).uniform_(2, 10, generator=g)], 1),
            'scales': torch.empty(N, 3).uniform_(-5.8, -3.5, generator=g),
            'rotations': torch.randn(N, 4, generator=g),
            'opacities': torch.randn(N, 1, generator=g),
            'sh_coefficients_0': torch.randn(N, 1, 3, generator=g) * 0.5,
            'sh_coefficients_rest': torch.randn(N, 15, 3, generator=g) * 0.1,
        }
        return {k: torch.nn.Parameter(v.to(dev)) for k, v in t.items()}

    target = torch.rand(3, H, W, generator=torch.Generator().manual_seed(1)).to(dev)

    def run(fused: bool):
        params = init()
        opt = LazyAdam([{'params': [params[n]], 'lr': 1e-3 if n != 'means' else 1e-4, 'name': n} for n in names], lr=0.0, eps=1e-15, num_steps=args.steps)
        if args.fp32:
            opt.moment_dtypes = (torch.float32, torch.float32)
        opt.fuse_sh_rest = fused
        n_fused = 0
        for t in range(args.steps):
            cam_x = 1.5 * ((t * 7) % 5 - 2)  # moving camera: changes the visible set every step
            w2c = torch.eye(4, device=dev)
            w2c[0, 3] = -cam_x
            settings = RasterizerSettings(
                w2c=w2c, cam_position=torch.tensor([cam_x, 0.0, 0.0], device=dev), bg_color=torch.zeros(3, device=dev),
                active_sh_bases=16, width=W, height=H, focal_x=700.0, focal_y=700.0, center_x=W / 2, center_y=H / 2,
                near_plane=0.2, far_plane=100.0, proper_antialiasing=False)
            fused_args = opt.fused_sh_rest(params['sh_coefficients_rest'])
            n_fused += fused_args is not None
            image = diff_rasterize(*(params[n] for n in names), torch.empty(0, device=dev), settings,
                                   visibility=opt.visibility_buffer(), fused_sh_adam=fused_args)
            ((image - target) ** 2).mean().backward()
            opt.step()
            for p in params.values():
                p.grad = None
        opt.flush()
        st = opt.state[params['sh_coefficients_rest']]
        out = {n: params[n].detach().clone() for n in names}
        out['sh_rest_m'] = st['exp_avg'].float().clone()
        out['sh_rest_v'] = st['exp_avg_sq'].float().clone()
        return out, n_fused

    a, nf = run(True)
    b, _ = run(False)
    c, _ = run(False)
    print(f'{args.steps} steps, {nf} with the fused SH-rest update; moments {"float32" if args.fp32 else "bfloat16 (SR)"}')
    for k in a:
        scale = b[k].abs().max().item()
        print(f'  {k:22s} fused vs unfused {(a[k] - b[k]).abs().max().item() / scale:.3e}   unfused vs unfused (noise floor) {(c[k] - b[k]).abs().max().item() / scale:.3e}')


if __name__ == '__main__':
    main()
