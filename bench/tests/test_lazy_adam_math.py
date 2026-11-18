"""Numerical check of the exact lazy Adam catch-up (reference implementation in PyTorch, float32 state like the CUDA kernel).

Dense Adam (Faster-GS adam_step_cu semantics) is compared with a lazy variant that only updates Gaussians whose gradient
is non-zero and applies the skipped zero-gradient steps when the Gaussian becomes visible again (or at a flush):
  - closed form if epsilon is negligible against sqrt(v):
        m <- beta1^k m,  v <- beta2^k v,
        theta <- theta - (m / sqrt(v)) * (F(t0) - gamma^k F(t0 + k)),  gamma = beta1 / sqrt(beta2),
        F(t) = gamma * (f(t + 1) + F(t + 1)),  f(s) = lr(s) * sqrt(1 - beta2^s) / (1 - beta1^s)
  - otherwise an exact replay of up to 256 zero-gradient steps with the dense arithmetic.
Both are compared with a float64 dense Adam to separate the method error from float32 rounding.

usage: python test_lazy_adam_math.py [T]
"""
import math
import sys

import torch

torch.manual_seed(0)
dev = 'cuda'
T = int(sys.argv[1]) if len(sys.argv) > 1 else 30000
N, D = 20000, 4
beta1, beta2, eps = 0.9, 0.999, 1e-15
lr0, lr1 = 1.6e-4, 1.6e-6
J = 256


def lr(s):  # exponential decay like LRDecayPolicy for the means
    return math.exp(math.log(lr0) * (1 - s / T) + math.log(lr1) * (s / T))


gamma = beta1 / math.sqrt(beta2)
f_list = [0.0] * (T + 2)
for step in range(1, T + 1):
    f_list[step] = lr(step) * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
F_list = [0.0] * (T + 2)
for t in range(T - 1, -1, -1):
    F_list[t] = gamma * (f_list[t + 1] + F_list[t + 1])
F = torch.tensor(F_list, dtype=torch.float64, device=dev)
step_size_t = torch.tensor([0.0] + [lr(s) * (1.0 / (1.0 - math.pow(beta1, s))) for s in range(1, T + 2)], dtype=torch.float32, device=dev)
bc2_t = torch.tensor([0.0] + [1.0 / math.sqrt(1.0 - math.pow(beta2, s)) for s in range(1, T + 2)], dtype=torch.float32, device=dev)

p_vis = torch.rand(N, device=dev) * 0.6 + 0.05  # per-Gaussian visibility probability (5% .. 65%)
grad_scale = 10.0 ** (-2.0 - 16.0 * torch.rand(N, 1, device=dev))  # per-Gaussian gradient scale 1e-18 .. 1e-2
theta_d = torch.randn(N, D, device=dev)
theta_0 = theta_d.clone()
theta_l = theta_d.clone()
m_d = torch.zeros(N, D, device=dev)
v_d = torch.zeros(N, D, device=dev)
m_l = torch.zeros(N, D, device=dev)
v_l = torch.zeros(N, D, device=dev)
theta_64 = theta_d.double()
m_64 = torch.zeros_like(theta_64)
v_64 = torch.zeros_like(theta_64)
last = torch.zeros(N, dtype=torch.int64, device=dev)  # last step at which this Gaussian was brought up to date
flush_steps = set(range(600, 15000, 100)) | {T}


def adam_dense_step(theta, m, v, g, s):
    m.mul_(beta1).add_(g, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
    step_size = lr(s) / (1 - beta1 ** s)
    denom = v.sqrt() / math.sqrt(1 - beta2 ** s) + eps
    theta.sub_(step_size * m / denom)


def catch_up(idx, upto):
    """Apply the skipped zero-gradient steps last+1 .. upto for the given Gaussians."""
    k = upto - last[idx]
    sel = k > 0
    idx, k = idx[sel], k[sel]
    if idx.numel() == 0:
        return
    m, v, th = m_l[idx], v_l[idx], theta_l[idx]
    window = k.clamp_max(J)
    sv_end = (v * (beta2 ** window.to(torch.float32))[:, None]).sqrt()
    closed = sv_end >= 1e6 * eps
    # closed form
    kf = k.to(torch.float64)[:, None]
    S = (F[last[idx]] - torch.exp(kf[:, 0] * math.log(gamma)) * F[last[idx] + k])[:, None]
    ratio = torch.where(v > 0, m.double() / v.double().sqrt(), torch.zeros_like(m, dtype=torch.float64))
    th_closed = (th.double() - ratio * S).float()
    m_closed = (m.double() * beta1 ** kf).float()
    v_closed = (v.double() * beta2 ** kf).float()
    # exact replay
    th_r, m_r, v_r = th.clone(), m.clone(), v.clone()
    t0 = last[idx]
    for j in range(1, J + 1):
        act = (j <= k)[:, None] & ~closed
        if not act.any():
            break
        s_idx = (t0 + j).clamp_max(T + 1)
        m_new = m_r * beta1
        v_new = v_r * beta2
        denom = v_new.sqrt() * bc2_t[s_idx][:, None] + eps
        th_new = th_r - step_size_t[s_idx][:, None] * m_new / denom
        th_r = torch.where(act, th_new, th_r)
        m_r = torch.where(act, m_new, m_r)
        v_r = torch.where(act, v_new, v_r)
    rest = (k - window).to(torch.float32)[:, None]
    m_r = m_r * beta1 ** rest
    v_r = v_r * beta2 ** rest
    theta_l[idx] = torch.where(closed, th_closed, th_r)
    m_l[idx] = torch.where(closed, m_closed, m_r)
    v_l[idx] = torch.where(closed, v_closed, v_r)
    last[idx] = upto


for s in range(1, T + 1):
    visible = torch.rand(N, device=dev) < p_vis
    g = torch.randn(N, D, device=dev) * grad_scale * visible[:, None]
    adam_dense_step(theta_d, m_d, v_d, g, s)
    adam_dense_step(theta_64, m_64, v_64, g.double(), s)
    vis_idx = visible.nonzero()[:, 0]
    catch_up(vis_idx, s - 1)
    tl, ml, vl = theta_l[vis_idx], m_l[vis_idx], v_l[vis_idx]
    adam_dense_step(tl, ml, vl, g[vis_idx], s)
    theta_l[vis_idx], m_l[vis_idx], v_l[vis_idx] = tl, ml, vl
    last[vis_idx] = s
    if s in flush_steps:
        catch_up(torch.arange(N, device=dev), s)

motion = (theta_d - theta_0).abs()
diff = (theta_d - theta_l).abs()
e32 = (theta_d.double() - theta_64).abs()
el = (theta_l.double() - theta_64).abs()
print(f'T={T} N={N}x{D}: parameter motion mean {motion.mean().item():.3e}; |dense-lazy| max {diff.max().item():.3e}, mean {diff.mean().item():.3e}')
print(f'vs float64 dense: dense-fp32 error mean {e32.mean().item():.3e} max {e32.max().item():.3e} | '
      f'lazy-fp32 error mean {el.mean().item():.3e} max {el.max().item():.3e}')
for lo, hi in [(-18, -14), (-14, -10), (-10, -6), (-6, -2)]:
    sel = (grad_scale[:, 0] >= 10.0 ** lo) & (grad_scale[:, 0] < 10.0 ** hi)
    print(f'  grad scale 1e{lo}..1e{hi}: lazy max err {el[sel].max().item():.3e} | dense-fp32 max err {e32[sel].max().item():.3e} | '
          f'motion mean {motion[sel].mean().item():.3e}')
print(f'max |m diff| {(m_d - m_l).abs().max().item():.3e}, max |v diff| rel {((v_d - v_l).abs() / v_d.clamp_min(1e-30)).max().item():.3e}')
