"""In-process exactness test of Faster-GS+LazyAdam against the baseline dense FusedAdam with the same real gradients.

Real training (lazyadam variant) runs to iteration START. There the lazy optimizer is flushed and its parameters and
state are cloned into a dense FusedAdam. For the next STEPS iterations, the gradients produced by real training are fed
to both optimizers (lazy: visible Gaussians only + closed-form catch-up; dense: every Gaussian every step). One extra
flush happens halfway. At the end both are compared group by group.

usage: python test_lazy_adam_integrated.py [--start 2500] [--steps 300] [--gpu 4]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=2500)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scene', default='garden')
    parser.add_argument('--variant', default='lazyadam')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    bench = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(bench))
    from common import backend_dir, variant_dir
    sys.path.insert(0, str(backend_dir(args.variant)))
    nerficg = variant_dir(args.variant) / 'nerficg'
    os.chdir(nerficg)
    sys.path.insert(0, str(nerficg / 'scripts'))
    sys.path.insert(0, str(nerficg / 'src'))
    import torch
    import yaml
    import run_bench
    import Framework

    cfg = run_bench.scene_config(args.variant, args.scene, 0, {})
    cfg['TRAINING']['MODEL_NAME'] = f'test_lazyadam_{args.scene}'
    cfg['TRAINING']['BACKUP']['FINAL_CHECKPOINT'] = False
    cfg['TRAINING']['BACKUP']['RENDER_TESTSET'] = False
    cfg['TRAINING']['NUM_ITERATIONS'] = 30000  # keep the real LR schedule / F tables; the run stops after the test window
    cfg_path = nerficg / 'configs' / 'test_lazyadam.yaml'
    cfg_path.parent.mkdir(exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
    Framework.setup(config_path=str(cfg_path), require_custom_config=True)
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    trainer = MI.get_training_instance(method='FasterGS', checkpoint=None)
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
    import Methods.FasterGS.Trainer as T
    adam_modules = [m for n, m in sys.modules.items() if n.endswith('torch_bindings.adam')]
    FusedAdam = adam_modules[0].FusedAdam

    test = {}

    def snapshot(opt):
        out = {}
        for g in opt.param_groups:
            p = g['params'][0]
            st = opt.state[p]
            out[g['name']] = (p.detach().clone(), st['exp_avg'].clone(), st['exp_avg_sq'].clone())
        return out

    orig_iter = T.FasterGSTrainer.training_iteration

    def training_iteration(self, iteration, ds):
        gs = self.model.gaussians
        if iteration == args.start:
            lazy = gs.optimizer
            lazy.flush()
            # dense copy with identical parameters and state
            params = []
            for g in lazy.param_groups:
                p = torch.nn.Parameter(g['params'][0].detach().clone())
                params.append({'params': [p], 'lr': g['lr'], 'name': g['name']})
            dense = FusedAdam(params, lr=0.0, eps=1e-15)
            for gl, gd in zip(lazy.param_groups, dense.param_groups):
                sl = lazy.state[gl['params'][0]]
                dense.state[gd['params'][0]] = {'step': sl['step'], 'exp_avg': sl['exp_avg'].clone(), 'exp_avg_sq': sl['exp_avg_sq'].clone()}
            test['dense'] = dense
            test['start_snapshot'] = snapshot(lazy)
            test['visible_frac'] = []
            orig_step = lazy.step

            def step_both():
                grads = {g['name']: g['params'][0].grad.detach().clone() for g in lazy.param_groups}
                # invisible Gaussians have exactly zero gradients; their slots may be unwritten when zero writes are skipped
                invisible = lazy.visibility_buffer() == 0
                for name in grads:
                    grads[name][invisible] = 0.0
                orig_step()
                test['visible_frac'].append(lazy.visibility.float().mean().item())
                for gd in dense.param_groups:
                    gd['lr'] = next(g['lr'] for g in lazy.param_groups if g['name'] == gd['name'])
                    gd['params'][0].grad = grads[gd['name']]
                dense.step()
                if lazy.step_count == args.start + args.steps // 2:
                    lazy.flush()
            lazy.step = step_both
        orig_iter(self, iteration, ds)
        if iteration == args.start + args.steps - 1:
            lazy = gs.optimizer
            lazy.flush()
            dense = test['dense']
            lazy_snap, dense_snap, start = snapshot(lazy), snapshot(dense), test['start_snapshot']
            print(f'\n== exactness after {args.steps} steps from iteration {args.start} (N = {gs.means.shape[0]:,}, mean visible fraction '
                  f'{sum(test["visible_frac"]) / len(test["visible_frac"]):.3f})')
            worst = 0.0
            for name in lazy_snap:
                (pl, ml, vl), (pd, md, vd), (p0, _, _) = lazy_snap[name], dense_snap[name], start[name]
                motion = (pd - p0).abs().mean().item()
                diff = (pl - pd).abs()
                rel = diff.mean().item() / max(motion, 1e-30)
                worst = max(worst, rel)
                print(f'  {name:22s} param motion mean {motion:.3e} | |lazy-dense| mean {diff.mean().item():.3e} max {diff.max().item():.3e} '
                      f'(mean/motion {rel:.2e}) | m max diff {(ml - md).abs().max().item():.2e} v max rel diff '
                      f'{((vl - vd).abs() / vd.abs().clamp_min(1e-30)).max().item():.2e}')
            print(f'RESULT worst mean-diff / motion = {worst:.2e}')
            sys.stdout.flush()
            os._exit(0)

    for attr in ['callback_type', 'active', 'priority', 'start_iteration', 'end_iteration', 'iteration_stride']:
        setattr(training_iteration, attr, getattr(orig_iter, attr))
    training_iteration.__name__ = 'training_iteration'
    T.FasterGSTrainer.training_iteration = training_iteration
    trainer.run(dataset)


if __name__ == '__main__':
    main()
