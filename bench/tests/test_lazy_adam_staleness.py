"""Trajectory-level comparison of Faster-GS+LazyAdam with dense Adam (each copy runs its own forward/backward).

Three parameter sets start from an identical (flushed) state at iteration START:
  L  : the live model, trained with LazyAdam (visible-only updates, deferred catch-up)
  D1 : dense copy trained with the baseline FusedAdam
  D2 : second dense copy (its distance to D1 is the noise floor from non-deterministic atomic accumulation)
Every iteration uses the same view for all three. Before each forward pass, the parameters of the Gaussians visible in
that view (per D1) are compared, in units of the mean per-step parameter motion of D1.

usage: python test_lazy_adam_staleness.py [--start 15100] [--steps 100] [--gpu 5] [--scene garden]
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=15100)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--scene', default='garden')
    parser.add_argument('--variant', default='lazyadam')
    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    bench = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(bench))
    from common import backend_dir, variant_dir
    variant = args.variant
    sys.path.insert(0, str(backend_dir(variant)))
    nerficg = variant_dir(variant) / 'nerficg'
    os.chdir(nerficg)
    sys.path.insert(0, str(nerficg / 'scripts'))
    sys.path.insert(0, str(nerficg / 'src'))
    import torch
    import yaml
    import run_bench
    import Framework

    cfg = run_bench.scene_config(variant, args.scene, 0, {})
    cfg['TRAINING']['MODEL_NAME'] = f'test_staleness_{args.scene}'
    cfg['TRAINING']['BACKUP']['FINAL_CHECKPOINT'] = False
    cfg['TRAINING']['BACKUP']['RENDER_TESTSET'] = False
    cfg_path = nerficg / 'configs' / 'test_staleness.yaml'
    yaml.safe_dump(cfg, open(cfg_path, 'w'), sort_keys=False)
    Framework.setup(config_path=str(cfg_path), require_custom_config=True)
    from Implementations import Methods as MI
    from Implementations import Datasets as DI
    trainer = MI.get_training_instance(method='FasterGS', checkpoint=None)
    dataset = DI.get_dataset(dataset_type=Framework.config.GLOBAL.DATASET_TYPE, path=Framework.config.DATASET.PATH)
    import Methods.FasterGS.Trainer as T
    from Datasets.utils import apply_background_color
    adam_modules = [m for n, m in sys.modules.items() if n.endswith('torch_bindings.adam')]
    FusedAdam = adam_modules[0].FusedAdam
    names = ['_means', '_sh_coefficients_0', '_sh_coefficients_rest', '_opacities', '_scales', '_rotations']
    group_of = {'_means': 'means', '_sh_coefficients_0': 'sh_coefficients_0', '_sh_coefficients_rest': 'sh_coefficients_rest',
                '_opacities': 'opacities', '_scales': 'scales', '_rotations': 'rotations'}
    test = {}

    def clone_dense(gs):
        lazy = gs.optimizer
        params, state = {}, {}
        for g in lazy.param_groups:
            p = g['params'][0]
            params[g['name']] = torch.nn.Parameter(p.detach().clone())
            st = lazy.state[p]
            state[g['name']] = (st['step'], st['exp_avg'].clone(), st['exp_avg_sq'].clone())
        opt = FusedAdam([{'params': [params[n]], 'lr': g['lr'], 'name': n} for n, g in zip(params, lazy.param_groups)], lr=0.0, eps=1e-15)
        for g in opt.param_groups:
            s, m, v = state[g['name']]
            opt.state[g['params'][0]] = {'step': s, 'exp_avg': m, 'exp_avg_sq': v}
        return params, opt

    def dense_iteration(self, gs, copy, view, iteration):
        """Forward/backward/step of a dense copy on the given view, with its parameters swapped into the model."""
        params, opt = copy
        live = {n: getattr(gs, n) for n in names}
        exact = getattr(gs.optimizer, 'exact', False)
        gs.optimizer.exact = False  # the lazy optimizer must not touch the dense copy's parameters
        for n in names:
            setattr(gs, n, params[group_of[n]])
        for g in opt.param_groups:
            g['lr'] = next(lg['lr'] for lg in gs.optimizer.param_groups if lg['name'] == g['name'])
        bg = view.camera.background_color
        image = self.renderer.render_image_training(view=view, update_densification_info=False, bg_color=bg)
        rgb_gt = view.rgb
        if (alpha_gt := view.alpha) is not None:
            rgb_gt = apply_background_color(rgb_gt, alpha_gt, bg)
        self.loss(image, rgb_gt).backward()
        visible = gs.optimizer.visibility_buffer().clone().bool()  # filled by this copy's backward pass
        for g in opt.param_groups:
            g['params'][0].grad[~visible] = 0.0  # invisible gradients are unwritten with LazyAdam's backward
        opt.step()
        opt.zero_grad()
        for n in names:
            setattr(gs, n, live[n])
        gs.optimizer.exact = exact
        return visible

    orig_iter = T.FasterGSTrainer.training_iteration
    orig_get = None

    def training_iteration(self, iteration, ds):
        gs = self.model.gaussians
        if iteration == args.start:
            gs.optimizer.flush()
            gs.optimizer.check_staleness = True
            test['D1'] = clone_dense(gs)
            test['D2'] = clone_dense(gs)
            test['rows'] = []
            test['prev_D1'] = {k: v.detach().clone() for k, v in test['D1'][0].items()}
        if args.start <= iteration < args.start + args.steps:
            # fix the view for this iteration and reuse it for the dense copies
            view = self.train_sampler.get(dataset=ds)['view']
            original_get = self.train_sampler.get
            self.train_sampler.get = lambda dataset: {'view': view}
            orig_iter(self, iteration, ds)  # live model: LazyAdam
            self.train_sampler.get = original_get
            vis1 = dense_iteration(self, gs, test['D1'], view, iteration)
            dense_iteration(self, gs, test['D2'], view, iteration)
            # compare at the visible Gaussians of the NEXT forward pass is not known yet; use this step's visible set
            row = {}
            P1 = test['D1'][0]
            for n in names:
                gname = group_of[n]
                live = getattr(gs, n).detach()
                motion = (P1[gname].detach() - test['prev_D1'][gname]).abs()[vis1].mean().item()
                row[gname] = (
                    (live - P1[gname].detach()).abs()[vis1].mean().item() / max(motion, 1e-30),
                    (test['D2'][0][gname].detach() - P1[gname].detach()).abs()[vis1].mean().item() / max(motion, 1e-30),
                )
            test['prev_D1'] = {k: v.detach().clone() for k, v in P1.items()}
            test['rows'].append(row)
            if iteration == args.start + args.steps - 1:
                print(f'\n== trajectory divergence over {args.steps} steps from iteration {args.start} ({args.scene}); '
                      f'units: mean per-step motion of the visible Gaussians')
                for gname in test['rows'][0]:
                    for label, window in [('first 10 steps', test['rows'][:10]), ('last 10 steps', test['rows'][-10:])]:
                        lz = sum(r[gname][0] for r in window) / len(window)
                        nf = sum(r[gname][1] for r in window) / len(window)
                        print(f'  {gname:22s} {label:15s}: |lazy - dense1| = {lz:7.3f}   |dense2 - dense1| (noise floor) = {nf:7.3f}')
                counts = getattr(gs.optimizer, 'stale_visible_counts', [])
                if counts:
                    print(f'  visible Gaussians not caught up before the forward pass: mean {sum(counts) / len(counts):.2f}, max {max(counts)} per step')
                fr = getattr(gs.optimizer, 'candidate_fractions', [])
                if fr:
                    print(f'  candidate fraction {sum(fr) / len(fr):.3f} of all Gaussians (extent scale {gs.optimizer.extent_scale}, pad {gs.optimizer.extent_pad} px)')
                sys.stdout.flush()
                os._exit(0)
            return
        orig_iter(self, iteration, ds)

    for attr in ['callback_type', 'active', 'priority', 'start_iteration', 'end_iteration', 'iteration_stride']:
        setattr(training_iteration, attr, getattr(orig_iter, attr))
    training_iteration.__name__ = 'training_iteration'
    T.FasterGSTrainer.training_iteration = training_iteration
    trainer.run(dataset)


if __name__ == '__main__':
    main()
