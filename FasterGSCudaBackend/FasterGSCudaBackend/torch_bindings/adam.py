import math
import torch

from FasterGSCudaBackend import _C


class FusedAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps) -> None:
        super().__init__(params=params, lr=lr, eps=eps)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            assert len(group['params']) == 1, 'more than one tensor in group'
            param = group['params'][0]

            if param.grad is None or param.numel() == 0:
                continue

            state = self.state[param]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param)
                state['exp_avg_sq'] = torch.zeros_like(param)

            state['step'] += 1

            _C.adam_step(
                param.grad,
                param,
                state['exp_avg'],
                state['exp_avg_sq'],
                state['step'],
                group['lr'],
                *group['betas'],
                group['eps'],
            )


class LazyAdam(FusedAdam):
    """Faster-GS+LazyAdam: Adam that only touches Gaussians visible in the current view, but stays equivalent to dense Adam.

    A Gaussian that is invisible has exactly zero gradients, so a dense Adam step only decays its moments and moves it along
    the remaining momentum. These steps are skipped and later applied in closed form (see lazy_adam.h), either when the
    Gaussian becomes visible again or in flush(), which must run before anything reads or rewrites parameters or optimizer
    state of all Gaussians (densification, opacity reset, sorting, end of training).
    """

    def __init__(self, params, lr, eps, num_steps: int, lr_schedules: dict | None = None, exact: bool = False) -> None:
        super().__init__(params=params, lr=lr, eps=eps)
        # Faster-GS+LazyAdam-Exact: bring every Gaussian that may be visible up to date before the forward pass
        self.exact = exact
        self.moment_dtypes = (torch.bfloat16, torch.bfloat16)
        # Faster-GS+FusedShAdam: the SH rest coefficients are updated inside the rasterizer backward pass
        self.fuse_sh_rest = True
        self.fused_done = set()  # names of the groups already updated in the backward pass of the current step
        self.candidates = None
        self.check_staleness = False
        self.stale_visible_counts = []
        self.candidate_fractions = []
        # inflation of the candidate test. The pad must cover the tile-grid padding of preprocess (bounds are clamped to
        # the grid padded to multiples of 16 x 12 px, so up to 15 px beyond the image still count as visible):
        # 1.1x + 4 px missed ~1-2.5k visible Gaussians per step, 1.5x + 16 px missed none (garden, iter 2601 / 15100)
        self.extent_scale = 1.5
        self.extent_pad = 16.0
        self.num_steps = num_steps
        self.lr_schedules = lr_schedules or {}  # group name -> callable(step) -> lr (defaults to the constant group lr)
        self.step_count = 0
        self.flushed_step = 0
        self.visibility = None
        self.last_step = None
        self.f_tables = {}

    def _f_table(self, group) -> torch.Tensor:
        name = group['name']
        if name not in self.f_tables:
            beta1, beta2 = group['betas']
            gamma = beta1 / math.sqrt(beta2)
            schedule = self.lr_schedules.get(name)
            T = self.num_steps + 1
            f = [0.0] * (T + 2)
            for s in range(1, T + 1):
                lr = schedule(s) if schedule is not None else group['lr']
                f[s] = lr * math.sqrt(1.0 - beta2 ** s) / (1.0 - beta1 ** s)
            F = [0.0] * (T + 2)
            for t in range(T - 1, -1, -1):
                F[t] = gamma * (f[t + 1] + F[t + 1])
            # per-step factors with exactly the double-precision formulas of adam_step_wrapper, then rounded to float32
            step_size = [0.0] * (T + 2)
            bc2 = [0.0] * (T + 2)
            for s in range(1, T + 1):
                lr = schedule(s) if schedule is not None else group['lr']
                step_size[s] = lr * (1.0 / (1.0 - math.pow(beta1, s)))
                bc2[s] = 1.0 / math.sqrt(1.0 - math.pow(beta2, s))
            self.f_tables[name] = (torch.tensor(F, dtype=torch.float64, device='cuda').to(torch.float32),
                                   torch.tensor(step_size, dtype=torch.float32, device='cuda'),
                                   torch.tensor(bc2, dtype=torch.float32, device='cuda'))
        return self.f_tables[name]

    def n_gaussians(self) -> int:
        return self.param_groups[0]['params'][0].shape[0]

    def visibility_buffer(self) -> torch.Tensor:
        """Buffer the rasterizer backward pass fills with the visibility of every Gaussian."""
        n = self.n_gaussians()
        if self.visibility is None or self.visibility.shape[0] != n:
            self.visibility = torch.zeros(n, dtype=torch.uint8, device='cuda')
        return self.visibility

    def _ensure_last_step(self) -> None:
        n = self.n_gaussians()
        if self.last_step is None or self.last_step.shape[0] != n:
            if self.last_step is not None and self.flushed_step != self.step_count:
                raise RuntimeError('LazyAdam: the number of Gaussians changed without a preceding flush()')
            self.last_step = torch.full((n,), self.step_count, dtype=torch.int32, device='cuda')

    @torch.no_grad()
    def prepare_forward(self, means: torch.Tensor, scales: torch.Tensor, rotations: torch.Tensor, settings) -> None:
        """Catches up all candidate-visible Gaussians to the last completed step, so that the forward pass of the next
        step sees exactly the parameters dense Adam would have (Faster-GS+LazyAdam-Exact)."""
        if not self.exact or self.step_count == 0:
            return
        self._ensure_last_step()
        n = self.n_gaussians()
        if self.candidates is None or self.candidates.shape[0] != n:
            self.candidates = torch.zeros(n, dtype=torch.uint8, device='cuda')
        _C.lazy_candidates(means, scales, rotations, settings.w2c, self.candidates, settings.width, settings.height,
                           settings.focal_x, settings.focal_y, settings.center_x, settings.center_y,
                           settings.near_plane, settings.far_plane, self.extent_scale, self.extent_pad)
        if self.check_staleness:
            self.candidate_fractions.append(self.candidates.float().mean().item())
        for group in self.param_groups:
            param = group['params'][0]
            state = self.state.get(param)
            if not state or param.numel() == 0:
                continue
            _C.lazy_adam_flush_masked(param, state['exp_avg'], state['exp_avg_sq'], self.candidates, self.last_step,
                                      *self._f_table(group), self.step_count, *group['betas'], group['eps'])
        _C.lazy_mark_masked(self.candidates, self.last_step, self.step_count)

    def _prepare_state(self, param):
        state = self.state[param]
        # Faster-GS+Combo-SR: bfloat16 m (round-to-nearest) and v (stochastic rounding in the kernels)
        m_dtype, v_dtype = self.moment_dtypes
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(param, dtype=m_dtype)
            state['exp_avg_sq'] = torch.zeros_like(param, dtype=v_dtype)
        else:
            # NeRFICG's extend_param_groups concatenates float32 zeros, which promotes the state to float32
            if state['exp_avg'].dtype != m_dtype:
                state['exp_avg'] = state['exp_avg'].to(m_dtype)
            if state['exp_avg_sq'].dtype != v_dtype:
                state['exp_avg_sq'] = state['exp_avg_sq'].to(v_dtype)
        return state

    @torch.no_grad()
    def fused_sh_rest(self, param: torch.Tensor):
        """Faster-GS+FusedShAdam: arguments for the lazy Adam step of the SH rest coefficients inside the rasterizer
        backward pass of the coming step, or None (fusion disabled, first step, or `param` is not the optimized tensor)."""
        if not self.fuse_sh_rest or self.step_count == 0:
            return None
        group = next((g for g in self.param_groups if g['name'] == 'sh_coefficients_rest'), None)
        if group is None or group['params'][0] is not param or param.numel() == 0:
            return None
        if len(self.state[param]) == 0:
            return None
        state = self._prepare_state(param)
        if state['step'] != self.step_count:
            return None
        self._ensure_last_step()
        return _FusedShAdam(self, group, state)

    @torch.no_grad()
    def step(self) -> None:
        self._ensure_last_step()
        if self.exact and self.check_staleness and self.step_count > 0:
            # visible Gaussians that the candidate test missed (would have been rendered with lagging parameters)
            self.stale_visible_counts.append(_C.lazy_count_stale_visible(self.visibility_buffer(), self.last_step, self.step_count))
        self.step_count += 1
        visibility = self.visibility_buffer()
        for group in self.param_groups:
            assert len(group['params']) == 1, 'more than one tensor in group'
            param = group['params'][0]
            if group['name'] in self.fused_done:
                # updated in place by the rasterizer backward pass of this step (Faster-GS+FusedShAdam)
                state = self.state[param]
                state['step'] += 1
                if state['step'] != self.step_count:
                    raise RuntimeError('LazyAdam: parameter groups are out of step')
                continue
            if param.grad is None:
                raise RuntimeError(f'LazyAdam: missing gradient for group {group["name"]}')
            if param.numel() == 0:
                continue
            state = self._prepare_state(param)
            state['step'] += 1
            if state['step'] != self.step_count:
                raise RuntimeError('LazyAdam: parameter groups are out of step')
            _C.lazy_adam_step(
                param.grad, param, state['exp_avg'], state['exp_avg_sq'],
                visibility, self.last_step, *self._f_table(group),
                state['step'], group['lr'], *group['betas'], group['eps'],
            )
        _C.lazy_adam_mark(visibility, self.last_step, self.step_count)
        self.fused_done.clear()

    @torch.no_grad()
    def flush(self) -> None:
        """Applies all skipped zero-gradient steps so that every Gaussian is up to date with the current step."""
        if self.last_step is None or self.flushed_step == self.step_count:
            self.flushed_step = self.step_count
            return
        if self.last_step.shape[0] != self.n_gaussians():
            raise RuntimeError('LazyAdam: flush() called after the number of Gaussians changed')
        for group in self.param_groups:
            param = group['params'][0]
            state = self.state.get(param)
            if not state or param.numel() == 0:
                continue
            _C.lazy_adam_flush(param, state['exp_avg'], state['exp_avg_sq'], self.last_step, *self._f_table(group),
                               self.step_count, *group['betas'], group['eps'])
        self.last_step.fill_(self.step_count)
        self.flushed_step = self.step_count


class _FusedShAdam:
    """Faster-GS+FusedShAdam: optimizer state handed to the rasterizer backward pass (step index of the coming step)."""

    def __init__(self, optimizer: 'LazyAdam', group: dict, state: dict) -> None:
        self.optimizer = optimizer
        self.group = group
        self.state = state

    def backward_args(self) -> tuple:
        group, state = self.group, self.state
        f_table, step_size_table, bc2_table = self.optimizer._f_table(group)
        beta1, beta2 = group['betas']
        return (state['exp_avg'], state['exp_avg_sq'], self.optimizer.last_step, f_table, step_size_table, bc2_table,
                self.optimizer.step_count + 1, group['lr'], beta1, beta2, group['eps'])

    def done(self) -> None:
        self.optimizer.fused_done.add(self.group['name'])
