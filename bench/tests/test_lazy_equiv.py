"""Bitwise equivalence of two builds of the lazy Adam kernels (e.g. Faster-GS+WarpLazy against Combo-SR).

`run <variant> <out.pt>` drives LazyAdam through a fixed synthetic sequence that exercises every lazy kernel: per step a
random candidate mask is caught up before the "forward pass" (lazy_adam_flush_masked + lazy_mark_masked, as in
LazyAdam.prepare_forward), then a step with random visibility (a subset of the candidates) and gradients, and a full
flush every 100 steps. Groups with 1, 3, 4, 45 and 7 values per Gaussian, an element count that is not a multiple of
32 Gaussians, gradient scales 1e-18..1e-2 (exercises the epsilon-aware replay) and both moment storages (float32,
bfloat16 with stochastic rounding). Saves parameters and optimizer state.
`compare <a.pt> <b.pt>` reports the number of differing values (bitwise).
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
    torch.set_default_device('cuda')
    from FasterGSCudaBackend import _C
    from FasterGSCudaBackend.torch_bindings.adam import LazyAdam

    N, T = 100_003, args.steps
    dims = {'opacities': 1, 'means': 3, 'rotations': 4, 'sh_coefficients_rest': 45, 'other': 7}
    out = {}
    for dtypes_name, dtypes in [('fp32', (torch.float32, torch.float32)), ('bf16', (torch.bfloat16, torch.bfloat16))]:
        gen = torch.Generator(device='cuda').manual_seed(0)
        params = {n: torch.nn.Parameter(torch.randn(N, d, generator=gen)) for n, d in dims.items()}
        scale = {n: 10.0 ** (-2.0 - 16.0 * torch.rand(N, 1, generator=gen)) for n in dims}
        opt = LazyAdam([{'params': [p], 'lr': 1e-3, 'name': n} for n, p in params.items()], lr=0.0, eps=1e-15, num_steps=T)
        opt.moment_dtypes = dtypes
        p_cand = torch.rand(N, generator=gen) * 0.6 + 0.05
        for t in range(1, T + 1):
            if opt.step_count > 0:
                opt._ensure_last_step()
                candidates = (torch.rand(N, generator=gen) < p_cand).to(torch.uint8)
                for group in opt.param_groups:
                    param = group['params'][0]
                    state = opt.state[param]
                    _C.lazy_adam_flush_masked(param, state['exp_avg'], state['exp_avg_sq'], candidates, opt.last_step,
                                              *opt._f_table(group), opt.step_count, *group['betas'], group['eps'])
                _C.lazy_mark_masked(candidates, opt.last_step, opt.step_count)
            else:
                candidates = torch.ones(N, dtype=torch.uint8)
            visible = candidates.bool() & (torch.rand(N, generator=gen) < 0.75)
            for n, p in params.items():
                p.grad = torch.randn(N, dims[n], generator=gen) * scale[n] * visible[:, None]
            opt.visibility_buffer().copy_(visible.to(torch.uint8))
            opt.step()
            if t % 100 == 0:
                opt.flush()
        opt.flush()
        for n, p in params.items():
            st = opt.state[p]
            out[f'{dtypes_name}/{n}/param'] = p.detach().cpu()
            out[f'{dtypes_name}/{n}/m'] = st['exp_avg'].float().cpu()
            out[f'{dtypes_name}/{n}/v'] = st['exp_avg_sq'].float().cpu()
    torch.save(out, args.out)
    print(f'saved {args.out}')


def compare(args):
    import torch
    a, b = torch.load(args.a), torch.load(args.b)
    total = 0
    for k in a:
        diff = (a[k].view(torch.int32) != b[k].view(torch.int32)).sum().item()
        total += diff
        print(f'{k:36s} differing values: {diff} / {a[k].numel()}')
    print('BITWISE IDENTICAL' if total == 0 else f'{total} values differ')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('variant')
    r.add_argument('out')
    r.add_argument('--gpu', type=int, default=0)
    r.add_argument('--steps', type=int, default=400)
    c = sub.add_parser('compare')
    c.add_argument('a')
    c.add_argument('b')
    args = parser.parse_args()
    run(args) if args.cmd == 'run' else compare(args)


if __name__ == '__main__':
    main()
