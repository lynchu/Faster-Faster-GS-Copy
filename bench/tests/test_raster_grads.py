"""Gradient equivalence of rasterizer variants on a synthetic scene.

`run <variant> <out.pt>` renders a fixed random scene (seeded) with diff_rasterize, backpropagates a fixed random
image-space gradient, and saves the image, all parameter gradients and the densification info.
`compare <a.pt> <b.pt> [<c.pt>]` prints the relative differences (c: second run of a, the atomics noise floor).

usage: python test_raster_grads.py run baseline /tmp/a.pt --gpu 4
       python test_raster_grads.py compare /tmp/a.pt /tmp/b.pt /tmp/a2.pt
"""
import argparse
import os
import sys
from pathlib import Path


def run(args):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import backend_dir
    sys.path.insert(0, str(backend_dir(args.variant)))
    import torch
    from FasterGSCudaBackend.torch_bindings.rasterization import diff_rasterize, RasterizerSettings
    torch.manual_seed(0)
    dev = 'cuda'
    N, W, H = 300_000, 1000, 650
    means = torch.stack([torch.empty(N).uniform_(-3, 3), torch.empty(N).uniform_(-2, 2), torch.empty(N).uniform_(2, 10)], 1).to(dev)
    scales = torch.empty(N, 3).uniform_(-5.8, -3.5).to(dev)
    rotations = torch.randn(N, 4).to(dev)
    opacities = torch.randn(N, 1).to(dev)
    sh0 = (torch.randn(N, 1, 3) * 0.5).to(dev)
    sh_rest = (torch.randn(N, 15, 3) * 0.1).to(dev)
    params = [p.requires_grad_() for p in (means, scales, rotations, opacities, sh0, sh_rest)]
    grad_image = torch.randn(3, H, W, generator=torch.Generator().manual_seed(1)).to(dev)
    settings = RasterizerSettings(
        w2c=torch.eye(4, device=dev), cam_position=torch.zeros(3, device=dev), bg_color=torch.zeros(3, device=dev),
        active_sh_bases=16, width=W, height=H, focal_x=900.0, focal_y=900.0, center_x=W / 2, center_y=H / 2,
        near_plane=0.2, far_plane=100.0, proper_antialiasing=False)
    out = {}
    for rows in (2, 3):
        dens = torch.zeros(rows, N, device=dev)
        for p in params:
            p.grad = None
        image = diff_rasterize(*params, dens, settings)
        (image * grad_image).sum().backward()
        out[f'image_{rows}'] = image.detach().cpu()
        out[f'dens_{rows}'] = dens.cpu()
        for name, p in zip(('means', 'scales', 'rotations', 'opacities', 'sh0', 'sh_rest'), params):
            out[f'grad_{name}_{rows}'] = p.grad.detach().cpu()
    torch.save(out, args.out)
    print(f'saved {args.out}: image mean {out["image_2"].mean().item():.4f}')


def compare(args):
    import torch
    a, b = torch.load(args.a), torch.load(args.b)
    c = torch.load(args.c) if args.c else None
    for k in a:
        scale = a[k].abs().max().item()
        d_ab = (a[k] - b[k]).abs().max().item() / max(scale, 1e-30)
        line = f'{k:18s} max |a-b| / max|a| = {d_ab:.3e}'
        if c is not None:
            d_ac = (a[k] - c[k]).abs().max().item() / max(scale, 1e-30)
            line += f'   noise floor (a vs a rerun) = {d_ac:.3e}'
        print(line)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('variant')
    r.add_argument('out')
    r.add_argument('--gpu', type=int, default=0)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('c', nargs='?')
    args = parser.parse_args()
    run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    main()
