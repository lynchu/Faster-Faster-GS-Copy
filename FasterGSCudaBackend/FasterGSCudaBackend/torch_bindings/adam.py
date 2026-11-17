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
            if param.grad is None:
                raise RuntimeError(f'LazyAdam: missing gradient for group {group["name"]}')
            if param.numel() == 0:
                continue
            state = self.state[param]
            # Faster-GS+HalfMoments-SH: bfloat16 moments for the SH rest coefficients only
            half = group.get('name') == 'sh_coefficients_rest'
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(param, dtype=torch.bfloat16 if half else torch.float32)
                state['exp_avg_sq'] = torch.zeros_like(param, dtype=torch.bfloat16 if half else torch.float32)
            elif half and state['exp_avg'].dtype != torch.bfloat16:
                # NeRFICG's extend_param_groups concatenates float32 zeros, which promotes the state to float32
                state['exp_avg'] = state['exp_avg'].to(torch.bfloat16)
                state['exp_avg_sq'] = state['exp_avg_sq'].to(torch.bfloat16)
            state['step'] += 1
            if state['step'] != self.step_count:
                raise RuntimeError('LazyAdam: parameter groups are out of step')
            _C.lazy_adam_step(
                param.grad, param, state['exp_avg'], state['exp_avg_sq'],
                visibility, self.last_step, *self._f_table(group),
                state['step'], group['lr'], *group['betas'], group['eps'],
            )
        _C.lazy_adam_mark(visibility, self.last_step, self.step_count)

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
