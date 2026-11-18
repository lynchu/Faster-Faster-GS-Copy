"""Profiling harness for Faster-GS training inside NeRFICG.

Runs a normal training (same config as run_bench.py, same per-callback timer synchronization) and adds:
  - NVTX ranges: iter / render_fwd / loss_fwd / backward / raster_bwd / adam
  - cudaProfilerStart/Stop windows at chosen iterations (for nsys --capture-range=cudaProfilerApi or ncu --profile-from-start off)
  - per-iteration statistics in a stats window right after each capture window (N, visible count, instances, buckets)
  - optional per-iteration wall time + N for the whole run ("timeline")

Designed to also run as root under `sudo ncu`: environment is configured from CLI args (sudo drops env vars),
bytecode writing must be disabled with `python -B`, and NeRFICG outputs go to --out-root.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

def main():
    global args, captures, state, nvtx, R
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    parser.add_argument('--scene', default='garden')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--capture', default='500,5000,15000,25000', help='iterations at which a profiler window starts')
    parser.add_argument('--capture-len', type=int, default=50)
    parser.add_argument('--stats-len', type=int, default=20)
    parser.add_argument('--max-iter', type=int, default=None, help='stop training after this many iterations (default: last window end)')
    parser.add_argument('--timeline', action='store_true', help='record per-iteration wall time for the whole run')
    parser.add_argument('--no-profiler-api', action='store_true', help='do not call cudaProfilerStart/Stop')
    parser.add_argument('--out', required=True, help='json file for statistics')
    parser.add_argument('--out-root', default=None, help='override NeRFICG output root')
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--probe', action='store_true', help='read probe counters in stats windows (probe variant only)')
    args = parser.parse_args()
    args.out = str(Path(args.out).resolve())
    if args.out_root:
        args.out_root = str(Path(args.out_root).resolve())

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # match nvidia-smi numbering
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    sys.dont_write_bytecode = True
    BENCH = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BENCH))
    from common import backend_dir, variant_dir  # noqa: E402

    sys.path.insert(0, str(backend_dir(args.variant)))
    NERFICG = variant_dir(args.variant) / 'nerficg'
    os.chdir(NERFICG)
    sys.path.insert(0, str(NERFICG / 'scripts'))
    sys.path.insert(0, str(NERFICG / 'src'))

    import torch  # noqa: E402
    import yaml  # noqa: E402
    import run_bench  # noqa: E402
    import Framework  # noqa: E402

    captures = sorted(int(c) for c in args.capture.split(',') if c)
    cfg = run_bench.scene_config(args.variant, args.scene, args.seed, json.loads(args.overrides))
    cfg['TRAINING']['MODEL_NAME'] = f'prof_{args.variant}_{args.scene}_s{args.seed}'
    cfg['TRAINING']['BACKUP']['FINAL_CHECKPOINT'] = False
    cfg['TRAINING']['BACKUP']['RENDER_TESTSET'] = False
    last = max(captures) + args.capture_len + args.stats_len if captures else cfg['TRAINING']['NUM_ITERATIONS']
    cfg['TRAINING']['NUM_ITERATIONS'] = min(cfg['TRAINING']['NUM_ITERATIONS'], args.max_iter or last)
    cfg_dir = Path(args.out_root) if args.out_root else NERFICG / 'configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f'{cfg["TRAINING"]["MODEL_NAME"]}.yaml'
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    if args.out_root:
        Framework.Directories.OUTPUT_DIR = Path(args.out_root)
    Framework.setup(config_path=str(cfg_path), require_custom_config=True)
    from Implementations import Methods as MI  # noqa: E402
    from Implementations import Datasets as DI  # noqa: E402

    trainer = MI.get_training_instance(method='FasterGS', checkpoint=None)
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)

    import Methods.FasterGS.Trainer as T  # noqa: E402
    import Methods.FasterGS.Loss as L  # noqa: E402
    # NeRFICG imports the bindings through a relative package path, so patch every loaded copy of the modules
    raster_modules = [m for n, m in list(sys.modules.items()) if n.endswith('torch_bindings.rasterization')]
    adam_modules = [m for n, m in list(sys.modules.items()) if n.endswith('torch_bindings.adam')]
    assert raster_modules and adam_modules, 'Faster-GS bindings not loaded'
    R = raster_modules[0]

    nvtx = torch.cuda.nvtx
    state = {'iteration': -1, 'stats': [], 'timeline': []}


    def in_stats_window(it: int) -> bool:
        return any(c + args.capture_len <= it < c + args.capture_len + args.stats_len for c in captures)


    class _CProxy:
        def __init__(self, c):
            self._c = c

        def __getattr__(self, name):
            return getattr(self._c, name)

        def forward(self, *a, **k):
            nvtx.range_push('raster_fwd')
            out = self._c.forward(*a, **k)
            nvtx.range_pop()
            state['last_fwd'] = {'n_instances': int(out[5]), 'n_buckets': int(out[6])}
            return out

        def backward(self, *a, **k):
            nvtx.range_push('raster_bwd')
            out = self._c.backward(*a, **k)
            nvtx.range_pop()
            if state.get('in_stats'):
                # invisible Gaussians (n_touched_tiles == 0) get exactly zero mean/scale/rotation gradients
                grad_means, grad_scales = out[0], out[1]
                state['last_bwd'] = {'n_visible': int(((grad_means != 0).any(1) | (grad_scales != 0).any(1)).sum().item())}
            return out


    proxy = _CProxy(raster_modules[0]._C)
    for m in raster_modules:
        m._C = proxy
    R = raster_modules[0]

    _loss_forward = L.FasterGSLoss.forward


    def loss_forward(self, *a, **k):
        nvtx.range_push('loss_fwd')
        out = _loss_forward(self, *a, **k)
        nvtx.range_pop()
        return out


    L.FasterGSLoss.forward = loss_forward

    def make_adam_step(orig):
        def adam_step(self, *a, **k):
            nvtx.range_push('adam')
            out = orig(self, *a, **k)
            nvtx.range_pop()
            return out
        return adam_step


    for m in adam_modules:
        m.FusedAdam.step = make_adam_step(m.FusedAdam.step)

    _tensor_backward = torch.Tensor.backward


    def tensor_backward(self, *a, **k):
        nvtx.range_push('backward')
        out = _tensor_backward(self, *a, **k)
        nvtx.range_pop()
        return out


    torch.Tensor.backward = tensor_backward

    _orig_iter = T.FasterGSTrainer.training_iteration


    def training_iteration(self, iteration, ds):
        if not args.no_profiler_api and iteration in captures:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
        if not args.no_profiler_api and any(iteration == c + args.capture_len for c in captures):
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        state['in_stats'] = in_stats_window(iteration)
        if args.probe and state['in_stats']:
            R._C._c.probe_read_reset()
        t0 = time.perf_counter()
        nvtx.range_push(f'iter {iteration}')
        _orig_iter(self, iteration, ds)
        nvtx.range_pop()
        if args.timeline or in_stats_window(iteration):
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            n = self.model.gaussians.means.shape[0]
            if args.timeline:
                state['timeline'].append((iteration, dt, n))
            if in_stats_window(iteration):
                g = self.model.gaussians
                entry = {'iteration': iteration, 'n_gaussians': n, 'iter_s': dt, **state.get('last_fwd', {}),
                         **state.get('last_bwd', {}), 'active_sh_bases': g.active_sh_bases}
                if args.probe:
                    names = ['fwd_pairs_evaluated', 'fwd_pairs_contributing', 'fwd_checkpoint_writes', 'fwd_pixel_threads',
                             'bwd_slots', 'bwd_pairs_evaluated', 'bwd_pairs_contributing', 'bwd_buckets_launched', 'bwd_buckets_active',
                         'bwd_union_pixels', 'bwd_union_rows', 'bwd_union_subtiles']
                    vals = R._C._c.probe_read_reset().tolist()
                    entry.update({k: v for k, v in zip(names, vals)})
                state['stats'].append(entry)


    for attr in ['callback_type', 'active', 'priority', 'start_iteration', 'end_iteration', 'iteration_stride']:
        setattr(training_iteration, attr, getattr(_orig_iter, attr))
    training_iteration.__name__ = 'training_iteration'
    T.FasterGSTrainer.training_iteration = training_iteration

    trainer.run(dataset)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'args': vars(args), 'stats': state['stats'], 'timeline': state['timeline']}, f)
    print(f'wrote {args.out}')
    Framework.teardown()


if __name__ == '__main__':
    main()
