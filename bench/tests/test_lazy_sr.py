"""Bias check of the bfloat16 moments in the lazy Adam kernels (Faster-GS+Combo-SR) against dense float32 Adam.

Synthetic Gaussians with random visibility (5%..65% per step) and log-uniform gradient scales; flush every 100 steps
(like densification). Compared after a final flush: relative bias / spread of v and the parameter error relative to
the parameter motion, for LazyAdam with (fp32 m, fp32 v), (bf16 m, fp32 v) and (bf16 m, bf16 v with stochastic rounding).

usage: python test_lazy_sr.py <variant> [--gpu 4] [--steps 3000]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3000)
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import backend_dir
    sys.path.insert(0, str(backend_dir(args.variant)))
    import torch
    torch.set_default_device('cuda')
    from FasterGSCudaBackend.torch_bindings.adam import FusedAdam, LazyAdam

    N, D, T, lr = 1 << 16, 16, args.steps, 1e-3
    torch.manual_seed(0)
    p_vis = torch.rand(N) * 0.6 + 0.05
    scale = 10.0 ** (-8.0 + 6.0 * torch.rand(N, 1))
    theta0 = torch.randn(N, D)

    def make(cls, dtypes=None):
        p = torch.nn.Parameter(theta0.clone())
        kw = dict(num_steps=T) if cls is LazyAdam else {}
        opt = cls([{'params': [p], 'lr': lr, 'name': 'x'}], lr=0.0, eps=1e-15, **kw)
        if dtypes is not None:
            opt.moment_dtypes = dtypes
        return p, opt

    configs = {'dense fp32': make(FusedAdam),
               'lazy fp32 m, fp32 v': make(LazyAdam, (torch.float32, torch.float32)),
               'lazy bf16 m, fp32 v': make(LazyAdam, (torch.bfloat16, torch.float32)),
               'lazy bf16 m, bf16 v (SR)': make(LazyAdam, (torch.bfloat16, torch.bfloat16))}
    for t in range(1, T + 1):
        visible = torch.rand(N) < p_vis
        g = torch.randn(N, D) * scale * visible[:, None]
        for name, (p, opt) in configs.items():
            p.grad = g.clone()
            if isinstance(opt, LazyAdam):
                opt.visibility_buffer().copy_(visible.to(torch.uint8))
            opt.step()
            if isinstance(opt, LazyAdam) and t % 100 == 0:
                opt.flush()
    for p, opt in configs.values():
        if isinstance(opt, LazyAdam):
            opt.flush()
    p_ref, o_ref = configs['dense fp32']
    v_ref = o_ref.state[p_ref]['exp_avg_sq'].float()
    motion = (p_ref.detach() - theta0).abs().mean().item()
    print(f'N={N}x{D}, T={T}: mean parameter motion {motion:.3e}')
    for name, (p, opt) in configs.items():
        if p is p_ref:
            continue
        v = opt.state[p]['exp_avg_sq'].float()
        ok = v_ref > 0
        rv = v[ok] / v_ref[ok] - 1
        err = (p.detach() - p_ref.detach()).abs().mean().item()
        print(f'  {name:26s}: v bias {rv.mean().item():+.4f} spread {rv.std().item():.4f} | param error {err / motion:.4f} x motion')


if __name__ == '__main__':
    main()
