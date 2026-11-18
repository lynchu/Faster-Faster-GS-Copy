"""Interleaved benchmark: for every (seed, scene) all variants run back to back, in rotating order, so that every
method sees the same host conditions. Already recorded runs are skipped (see run_bench.run_one).

usage: python run_suite.py baseline,syncfree,lazyadam [--scenes all] [--seeds 0,1,2]
"""
import argparse
import json
import sys

import run_bench

parser = argparse.ArgumentParser()
parser.add_argument('variants')
parser.add_argument('--scenes', default='all')
parser.add_argument('--seeds', default='0,1,2')
parser.add_argument('--overrides', default='{}')
parser.add_argument('--gpu', type=int, default=run_bench.GPU_INDEX)
args = parser.parse_args()
variants = args.variants.split(',')
overrides = json.loads(args.overrides)
k = 0
for seed in [int(s) for s in args.seeds.split(',')]:
    for scene in run_bench.parse_scenes(args.scenes):
        order = variants[k % len(variants):] + variants[:k % len(variants)]
        k += 1
        for v in order:
            run_bench.run_one(v, scene, seed, overrides, args.gpu, keep_ckpt=False)
            sys.stdout.flush()
