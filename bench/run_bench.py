"""Runs full trainings of a variant on Mip-NeRF360 scenes and records results in the registry.

A run is identified by (variant, commit, config hash, scene, seed, gpu name). Runs that already exist in the
registry are skipped, so results are never recomputed.

usage:
  python run_bench.py <variant> [--scenes garden,bonsai|all|outdoor|indoor] [--seeds 0,1,2] [--overrides '{"TRAINING.X": 1}']
"""
import argparse
import copy
import datetime
import json
import os
import re
import threading
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from common import (ALL13, BENCH, DEEPBLENDING, DEV_SCENES, GPU_INDEX, INDOOR, MIPNERF360, OUTDOOR, PYTHON, TANDT, append_registry, load_registry,
                    extension_hash, method_dir, nerficg_commit, run_env, stable_hash, variant_commit, variant_dir)


def base_config(variant: str) -> dict:
    with open(method_dir(variant) / 'fastergs_garden.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['GLOBAL']['GPU_INDICES'] = [0]
    cfg['TRAINING']['GUI']['ACTIVATE'] = False
    cfg['TRAINING']['WANDB']['ACTIVATE'] = False
    cfg['TRAINING']['TIMING']['ACTIVATE'] = True
    cfg['TRAINING']['WRITE_VRAM_STATS'] = True
    cfg['TRAINING']['RUN_VALIDATION'] = False
    cfg['TRAINING']['BACKUP']['FINAL_CHECKPOINT'] = True  # required: test-set rendering happens in the final checkpoint callback
    cfg['TRAINING']['BACKUP']['RENDER_TESTSET'] = True
    cfg['TRAINING']['BACKUP']['INTERMEDIATE_RENDERINGS'] = False
    cfg['TRAINING']['BACKUP']['INTERVAL'] = -1
    cfg['DATASET']['APPLY_PCA'] = False  # benchmark setting (see comment in fastergs_garden.yaml)
    return cfg


def set_key(cfg: dict, dotted: str, value) -> None:
    node = cfg
    keys = dotted.split('.')
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


def scene_config(variant: str, scene: str, seed: int, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_config(variant))
    cfg['GLOBAL']['RANDOM_SEED'] = seed
    if scene in TANDT or scene in DEEPBLENDING:
        # 3DGS benchmark data (tandt_db.zip), trained at the provided resolution; the T&T images are stored at half the
        # resolution of their COLMAP intrinsics, which only the dedicated loader handles. Near/far planes and all other
        # settings stay as for Mip-NeRF360 (no PCA alignment, near plane 0.2 like the 3DGS frustum culling).
        cfg['GLOBAL']['DATASET_TYPE'] = 'TanksAndTemples_3DGS' if scene in TANDT else 'Colmap'
        cfg['DATASET']['PATH'] = f'dataset/tandt_db/{"tandt" if scene in TANDT else "db"}/{scene}'
        cfg['DATASET']['IMAGE_SCALE_FACTOR'] = None
    else:
        cfg['DATASET']['PATH'] = f'dataset/mipnerf360/{scene}'
        cfg['DATASET']['IMAGE_SCALE_FACTOR'] = 0.25 if scene in OUTDOOR else 0.5
    cfg['TRAINING']['MODEL_NAME'] = f'bench_{variant}_{scene}_s{seed}'
    for k, v in overrides.items():
        set_key(cfg, k, v)
    return cfg


def gpu_name(gpu: int) -> str:
    return subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=name', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.strip()


def gpu_busy(gpu: int) -> bool:
    out = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True).stdout.strip()
    return bool(out)


def parse_outputs(out_dir: Path) -> dict:
    res = {'output_dir': str(out_dir)}
    res['train_s'] = float(re.search(r'Time:([\d.eE+-]+)', (out_dir / 'timings.txt').read_text()).group(1))
    vram = (out_dir / 'vram_stats.txt').read_text()
    res['vram_alloc_bytes'] = int(re.search(r'VRAM_allocated:(\d+)', vram).group(1))
    res['vram_reserved_bytes'] = int(re.search(r'VRAM_reserved:(\d+)', vram).group(1))
    res['n_gaussians'] = int(re.search(r'N_Gaussians:(\d+)', (out_dir / 'n_gaussians.txt').read_text()).group(1))
    if (out_dir / 'loop_wall.txt').exists():
        res['loop_wall_s'] = float(re.search(r'LoopWall:([\d.eE+-]+)', (out_dir / 'loop_wall.txt').read_text()).group(1))
    metric_files = sorted(out_dir.rglob('metrics_8bit.txt'))
    if not metric_files:
        raise RuntimeError(f'no metrics_8bit.txt found in {out_dir}')
    last = metric_files[0].read_text().strip().splitlines()[-1]
    for name, val in re.findall(r'(\w+):([\d.eE+-]+)', last):
        res[name.lower()] = float(val)
    res['timings_txt'] = (out_dir / 'timings.txt').read_text()
    return res


def telemetry_summary(path: Path) -> dict:
    clocks, temps, throttled, rows = [], [], 0, 0
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 7:
            continue
        try:
            clk, temp, util, reasons = float(parts[1]), float(parts[3]), float(parts[5]), int(parts[6], 16)
        except ValueError:
            continue
        if util < 50:  # ignore samples during data loading / test rendering
            continue
        rows += 1
        clocks.append(clk)
        temps.append(temp)
        # ignore GPU idle (0x1) and application clock setting (0x2); count power/thermal/other slowdowns
        throttled += bool(reasons & ~0x3)
    if not rows:
        return {}
    return {'sm_clock_mean': sum(clocks) / rows, 'sm_clock_min': min(clocks), 'temp_max': max(temps), 'throttled_frac': throttled / rows,
            'telemetry': str(path)}


def run_one(variant: str, scene: str, seed: int, overrides: dict, gpu: int, keep_ckpt: bool) -> dict | None:
    commit, dirty = variant_commit(variant)
    nerficg, nerficg_dirty = nerficg_commit(variant)
    if dirty or nerficg_dirty:
        sys.exit(f'variant {variant} has uncommitted changes; commit them before benchmarking')
    cfg = scene_config(variant, scene, seed, overrides)
    cfg_for_hash = copy.deepcopy(cfg)
    cfg_for_hash['TRAINING'].pop('MODEL_NAME')
    key = {'variant': variant, 'commit': commit, 'nerficg_commit': nerficg, 'config': stable_hash(cfg_for_hash), 'scene': scene, 'seed': seed,
           'gpu': gpu_name(gpu), 'ext_hash': extension_hash(variant)}
    for e in load_registry():
        if all(e.get(k) == v for k, v in key.items()):
            print(f'[skip] {variant} {scene} seed {seed}: already recorded')
            return None
    while gpu_busy(gpu):
        print(f'[wait] GPU {gpu} is in use by another process, waiting 60 s')
        time.sleep(60)
    nerficg_root = variant_dir(variant) / 'nerficg'
    cfg_path = nerficg_root / 'configs' / f'{cfg["TRAINING"]["MODEL_NAME"]}.yaml'
    cfg_path.parent.mkdir(exist_ok=True)
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    log_path = BENCH / 'logs' / f'{cfg["TRAINING"]["MODEL_NAME"]}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log'
    print(f'[run] {variant} {scene} seed {seed} -> {log_path}')
    t0 = time.time()
    telemetry_path = log_path.with_suffix('.gpu.csv')
    with open(telemetry_path, 'w') as tel:
        telemetry = subprocess.Popen(['nvidia-smi', '-i', str(gpu), '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,'
                                      'utilization.gpu,clocks_throttle_reasons.active', '--format=csv,noheader,nounits', '-lms', '2000'],
                                     stdout=tel, stderr=subprocess.DEVNULL)
    loads = []
    stop_sampling = threading.Event()

    def sample_load():
        while not stop_sampling.wait(10.0):
            loads.append(os.getloadavg()[0])

    sampler = threading.Thread(target=sample_load, daemon=True)
    sampler.start()
    with open(log_path, 'w') as log:
        proc = subprocess.run([str(PYTHON), 'scripts/train.py', '-c', str(cfg_path)], cwd=nerficg_root,
                              env=run_env(variant, gpu), stdout=log, stderr=subprocess.STDOUT)
    telemetry.terminate()
    stop_sampling.set()
    if proc.returncode != 0:
        print(f'[fail] {variant} {scene} seed {seed}, see {log_path}')
        return None
    outs = sorted((nerficg_root / 'output' / 'FasterGS').glob(f'{cfg["TRAINING"]["MODEL_NAME"]}_*'), key=lambda p: p.stat().st_mtime)
    res = parse_outputs(outs[-1])
    if not keep_ckpt:
        for ckpt in (outs[-1] / 'checkpoints').glob('*.pt'):
            ckpt.unlink()
    entry = {**key, **res, **telemetry_summary(telemetry_path), 'load_mean': sum(loads) / max(len(loads), 1), 'load_max': max(loads, default=0.0),
             'overrides': overrides, 'wall_s': time.time() - t0, 'log': str(log_path),
             'finished': f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}'}
    append_registry(entry)
    print(f'[done] {variant} {scene} s{seed}: PSNR {res.get("psnr"):.2f} SSIM {res.get("ssim"):.3f} LPIPS {res.get("lpips"):.3f} '
          f'train {res["train_s"]:.1f}s VRAM {res["vram_alloc_bytes"] / 2**30:.2f}GiB #Gs {res["n_gaussians"]:,}')
    return entry


def parse_scenes(s: str) -> list[str]:
    return {'all': MIPNERF360, 'mipnerf360': MIPNERF360, 'outdoor': OUTDOOR, 'indoor': INDOOR, 'dev': DEV_SCENES, 'tandt': TANDT,
            'db': DEEPBLENDING, 'tandt_db': TANDT + DEEPBLENDING, 'all13': ALL13}.get(s, None) or s.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant')
    parser.add_argument('--scenes', default='dev')
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--overrides', default='{}')
    parser.add_argument('--gpu', type=int, default=GPU_INDEX)
    parser.add_argument('--keep-ckpt', action='store_true')
    args = parser.parse_args()
    overrides = json.loads(args.overrides)
    for seed in [int(s) for s in args.seeds.split(',')]:
        for scene in parse_scenes(args.scenes):
            run_one(args.variant, scene, seed, overrides, args.gpu, args.keep_ckpt)
            sys.stdout.flush()
