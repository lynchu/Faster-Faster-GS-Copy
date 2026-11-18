"""Summarizes nsys captures of prof_train.py into per-stage GPU time per iteration.

usage: python analyze_nsys.py <out.json> <capture1.sqlite> [<capture2.sqlite> ...]

Stages (match the dashboard pipeline blocks):
  pre    preprocess_cu (b2)
  sort   depth/tile radix sorts, scans, apply_depth_ordering, create_instances, range/bucket extraction (b3-b5)
  fwd    blend_cu (b6)
  loss   kernels launched in NVTX 'loss_fwd' or in 'backward' outside 'raster_bwd' (b7)
  bwd    blend_backward_cu (b8)
  prebwd preprocess_backward_cu (b9)
  adam   adam_step_cu (b10)
  other  every remaining kernel/memset/memcpy plus GPU idle time inside the iteration
"""
import json
import re
import sqlite3
import sys
from collections import defaultdict

SORT_PAT = re.compile(r'DeviceRadixSort|DeviceScan|Onesweep|apply_depth_ordering|create_instances|extract_instance_ranges|extract_bucket_counts')


def load(db_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    strings = dict(cur.execute('SELECT id, value FROM StringIds'))
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    # CPU launch time for each correlation id
    launch = {}
    for start, corr in cur.execute('SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        launch[corr] = start
    gpu = []  # (start, end, name, launch_time)
    for start, end, name_id, corr in cur.execute('SELECT start, end, demangledName, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL'):
        gpu.append((start, end, strings.get(name_id, '?'), launch.get(corr, start)))
    if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables:
        for start, end, corr in cur.execute('SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_MEMSET'):
            gpu.append((start, end, '[memset]', launch.get(corr, start)))
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for start, end, kind, corr in cur.execute('SELECT start, end, copyKind, correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY'):
            gpu.append((start, end, f'[memcpy kind {kind}]', launch.get(corr, start)))
    ranges = defaultdict(list)
    for start, end, text, text_id in cur.execute('SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        label = text if text is not None else strings.get(text_id, '')
        ranges[label.split(' ')[0]].append((start, end, label))
    return gpu, ranges


def inside(t, rs):
    return any(s <= t <= e for s, e, _ in rs)


def classify(name, t, ranges):
    if 'preprocess_backward' in name:
        return 'prebwd'
    if 'blend_backward' in name:
        return 'bwd'
    if 'adam_step' in name:
        return 'adam'
    if 'lazy_adam_warp' in name:
        # Faster-GS+WarpLazy runs the optimizer step and the pre-forward catch-up with the same kernel: launches before
        # the iteration's rasterizer forward are the catch-up, launches after it are the step
        it = next(((s, e) for s, e, _ in ranges['iter'] if s <= t <= e), None)
        if it is not None:
            fwd_starts = [s for s, _, _ in ranges['raster_fwd'] if it[0] <= s <= it[1]]
            if fwd_starts and t < min(fwd_starts):
                return 'other'
        return 'adam'
    if 'preprocess_cu' in name:
        return 'pre'
    if 'blend_cu' in name:
        return 'fwd'
    if inside(t, ranges['raster_fwd']) and SORT_PAT.search(name):
        return 'sort'
    if inside(t, ranges['loss_fwd']) or (inside(t, ranges['backward']) and not inside(t, ranges['raster_bwd'])):
        return 'loss'
    return 'other'


def busy_union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def summarize(db_path):
    gpu, ranges = load(db_path)
    iters = sorted(ranges['iter'])
    per_iter = []
    # an iteration's period runs from the start of its 'iter' range to the start of the next one, so it includes the
    # per-callback timer syncs, the other callbacks and the Python loop overhead; the last iteration has no successor
    for (s, _, label), (s_next, _, _) in zip(iters[:-1], iters[1:]):
        items = [g for g in gpu if s <= g[3] < s_next]
        stage = defaultdict(float)
        names = []
        for gs, ge, name, t in items:
            st = classify(name, t, ranges)
            stage[st] += (ge - gs) / 1e6
            names.append((re.sub(r'\(.*', '', name)[:90], (ge - gs) / 1e6))
        wall = (s_next - s) / 1e6
        busy = busy_union([(max(g[0], s), min(g[1], s_next)) for g in items if g[1] > s]) / 1e6
        stage['idle'] = max(0.0, wall - busy)
        per_iter.append({'iteration': int(label.split(' ')[1]), 'wall_ms': wall, 'busy_ms': busy, 'names': names, **stage})
    # drop outliers caused by periodic callbacks (densification, Morton ordering) that share the iteration's period
    walls = sorted(p['wall_ms'] for p in per_iter)
    med = walls[len(walls) // 2] if walls else 0.0
    kept = [p for p in per_iter if p['wall_ms'] <= 2.0 * med]
    dropped = [p['iteration'] for p in per_iter if p['wall_ms'] > 2.0 * med]
    kernel_tot = defaultdict(lambda: [0.0, 0])
    for p in kept:
        for short, ms in p.pop('names'):
            kernel_tot[short][0] += ms
            kernel_tot[short][1] += 1
    per_iter = kept
    n = len(per_iter)
    keys = ['pre', 'sort', 'fwd', 'loss', 'bwd', 'prebwd', 'adam', 'other', 'idle']
    mean = {k: sum(p.get(k, 0.0) for p in per_iter) / n for k in keys + ['wall_ms', 'busy_ms']}
    kernels = sorted(({'name': k, 'total_ms_per_iter': v[0] / n, 'launches_per_iter': v[1] / n} for k, v in kernel_tot.items()),
                     key=lambda x: -x['total_ms_per_iter'])
    return {'file': db_path, 'n_iters': n, 'dropped_iters': dropped, 'first_iter': per_iter[0]['iteration'] if per_iter else None, 'mean_ms': mean,
            'kernels': kernels[:40], 'launches_per_iter': sum(v[1] for v in kernel_tot.values()) / max(n, 1)}


if __name__ == '__main__':
    out = [summarize(p) for p in sys.argv[2:]]
    with open(sys.argv[1], 'w') as f:
        json.dump(out, f, indent=1)
    for o in out:
        m = o['mean_ms']
        print(f"iter {o['first_iter']}: wall {m['wall_ms']:.2f} ms busy {m['busy_ms']:.2f} | " +
              ' '.join(f"{k} {m[k]:.2f}" for k in ['pre', 'sort', 'fwd', 'loss', 'bwd', 'prebwd', 'adam', 'other', 'idle']) +
              f" | launches/iter {o['launches_per_iter']:.0f}")
