"""Aggregates registry results into the dashboard table (dashboard/data.json) and prints a text table.

Per scene: quality metrics are averaged over seeds, train time and VRAM use the median over seeds.
Dataset values are the mean over scenes (only shown as final when every scene has all seeds).

usage: python summarize.py [--seeds 3] [--no-dashboard]
"""
import argparse
import json
import os
from collections import defaultdict
from statistics import mean, median

from common import ALL13, DEEPBLENDING, INDOOR, MIPNERF360, OUTDOOR, ROOT, TANDT, load_registry

DASH = ROOT / 'dashboard' / 'data.json'
DISPLAY = {'baseline': 'Faster-GS', 'syncfree': 'Faster-GS+SyncFree', 'warpsync': 'Faster-GS+WarpSync', 'fusedloss': 'Faster-GS+FusedLoss',
           'lazyadam': 'Faster-GS+LazyAdam', 'lazyexact': 'Faster-GS+LazyAdam-Exact', 'combo': 'Faster-GS+Combo',
           'comboexact': 'Faster-GS+Combo-Exact', 'leanloop': 'Faster-GS+LeanLoop', 'halfmom': 'Faster-GS+HalfMoments', 'halfmomsh': 'Faster-GS+HalfMoments-SH', 'combofast': 'Faster-GS+Combo-Fast',
           'halfmomsr': 'Faster-GS+HalfMoments-SR', 'halfm1': 'Faster-GS+HalfFirstMoment',
           'combosr': 'Faster-GS+Combo-SR', 'leanshfl3': 'Faster-GS+LeanShuffle', 'lazyexactw': 'Faster-GS+WarpLazy',
           'bwdhoist2': 'Faster-GS+BwdHoist', 'fusedshi': 'Faster-GS+FusedShAdam',
           'combofull3': 'Faster-GS+Combo-Full',
           'cfnobh': 'Combo-Full − BwdHoist', 'cfnols': 'Combo-Full − LeanShuffle − BwdHoist'}
NOTES = {'baseline': 'baseline · 本機量測', 'comboexact': 'SyncFree + FusedLoss + LeanLoop + WarpSync + LazyAdam-Exact',
         'combosr': 'Combo-Exact + HalfMoments-SR', 'lazyexactw': '含前提 LazyAdam-Exact', 'bwdhoist2': '含前提 LeanShuffle',
         'fusedshi': '含前提 LazyAdam-Exact + WarpLazy', 'combofull3': 'Combo-SR + LeanShuffle + BwdHoist + WarpLazy + FusedShAdam',
         'cfnobh': '消融：Combo-Full 拿掉 BwdHoist', 'cfnols': '消融：Combo-Full 拿掉 LeanShuffle 與 BwdHoist'}
ORDER = ['baseline', 'syncfree', 'fusedloss', 'leanloop', 'warpsync', 'lazyadam', 'lazyexact', 'lazyexactw', 'fusedshi', 'halfmomsr', 'leanshfl3', 'bwdhoist2', 'comboexact', 'combosr', 'combofull3', 'cfnols', 'cfnobh']


TABLE_GPU = os.environ.get('FFGS_TABLE_GPU', 'NVIDIA RTX PRO 6000 Blackwell Server Edition')  # GPU whose runs form the table


# seeds per method: 3 for the baseline and the combinations (2026-11-17) and, since 2026-11-19, also for the 10 individual
# methods inside Combo-Full; the failed or superseded methods stay at 1 seed (seed 0)
SEEDS_FULL = {'baseline', 'comboexact', 'combosr', 'combofull3', 'syncfree', 'fusedloss', 'leanloop', 'warpsync', 'lazyexact',
              'halfmomsr', 'leanshfl3', 'bwdhoist2', 'lazyexactw', 'fusedshi'}


def target_seeds(variant: str, n_seeds: int) -> int:
    if variant in ABLATION:
        return len(ABLATION_SEEDS)
    return n_seeds if variant in SEEDS_FULL else 1


# ablations of Combo-Full: measured interleaved with Combo-Full itself on seeds 3 and 4 (the baseline has no runs with
# these seeds); their speedup is Combo-Full's 3-seed speedup times the paired Combo-Full / ablation ratio on seeds 3, 4
ABLATION = {'cfnols', 'cfnobh'}
ABLATION_SEEDS = (3, 4)
ABLATION_REF = 'combofull3'


def aggregate(entries, n_seeds_full):
    by_variant = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e.get('gpu') != TABLE_GPU:
            continue
        by_variant[e['variant']][e['scene']].append(e)
    rows = {}
    for variant, scenes in by_variant.items():
        n_seeds = target_seeds(variant, n_seeds_full)
        per_scene = {}
        for scene, runs in scenes.items():
            # keep only the latest commit's runs of this variant
            latest = max(runs, key=lambda r: r['finished'])['commit']
            runs = sorted((r for r in runs if r['commit'] == latest), key=lambda r: r['seed'])
            runs = [r for r in runs if r['seed'] in ABLATION_SEEDS] if variant in ABLATION else runs[:n_seeds]
            per_scene[scene] = {
                'runs': len(runs),
                'psnr': mean(r['psnr'] for r in runs), 'ssim': mean(r['ssim'] for r in runs), 'lpips': mean(r['lpips'] for r in runs),
                'train_s': median(r['train_s'] for r in runs),
                'vram_gib': median(r['vram_alloc_bytes'] for r in runs) / 2**30,
                'n_gs_m': mean(r['n_gaussians'] for r in runs) / 1e6,
                'psnr_runs': [round(r['psnr'], 3) for r in runs], 'train_runs': [round(r['train_s'], 1) for r in runs],
                'seed_train': {r['seed']: r['train_s'] for r in runs},
            }
        done = [s for s in MIPNERF360 if s in per_scene and per_scene[s]['runs'] >= n_seeds]
        complete = len(done) == len(MIPNERF360)
        use = [s for s in MIPNERF360 if s in per_scene]
        row = {'variant': variant, 'scenes': per_scene, 'complete': complete, 'seeds': n_seeds,
               'runs_done': sum(min(per_scene[s]['runs'], n_seeds) for s in use), 'runs_total': n_seeds * len(MIPNERF360)}
        if complete:
            for k in ['psnr', 'ssim', 'lpips', 'train_s', 'vram_gib', 'n_gs_m']:
                row[k] = mean(per_scene[s][k] for s in MIPNERF360)
        elif all(s in per_scene for s in MIPNERF360):
            # provisional dataset average over the seeds completed for every scene (shown separately on the dashboard)
            k_seeds = min(per_scene[s]['runs'] for s in MIPNERF360)
            row['provisional'] = {k: mean(per_scene[s][k] for s in MIPNERF360) for k in ['psnr', 'ssim', 'lpips', 'train_s', 'vram_gib', 'n_gs_m']}
            row['provisional']['seeds'] = k_seeds
        rows[variant] = row
    # speedup against the baseline runs with the same seeds (paired: a 1-seed method is compared with baseline seed 0)
    base = rows.get('baseline')
    for variant, row in rows.items():
        if base is None or variant == 'baseline':
            continue
        if row['complete']:
            k = row['seeds']
        elif 'provisional' in row:
            k = row['provisional']['seeds']
        else:
            continue
        seeds = set(range(k))
        try:
            t_base = mean(median(base['scenes'][s]['seed_train'][sd] for sd in seeds) for s in MIPNERF360)
            t_var = mean(median(row['scenes'][s]['seed_train'][sd] for sd in seeds) for s in MIPNERF360)
        except KeyError:
            continue
        row['speedup'] = t_base / t_var
        row['speedup_seeds'] = k
    ref = rows.get(ABLATION_REF)
    ref_runs = defaultdict(dict)
    for e in entries:
        if e.get('gpu') == TABLE_GPU and e['variant'] == ABLATION_REF and e['seed'] in ABLATION_SEEDS:
            ref_runs[e['scene']][e['seed']] = e['train_s']
    for variant in ABLATION & set(rows):
        row = rows[variant]
        if not (row['complete'] and ref and 'speedup' in ref and all(len(ref_runs[s]) == len(ABLATION_SEEDS) for s in MIPNERF360)):
            continue
        t_ref = mean(median(ref_runs[s].values()) for s in MIPNERF360)
        t_var = mean(median(row['scenes'][s]['seed_train'].values()) for s in MIPNERF360)
        row['speedup'] = ref['speedup'] * t_ref / t_var
        row['speedup_seeds'] = len(ABLATION_SEEDS)
        # shown on Combo-Full's basis so the column compares like with like; the raw seed 3/4 times go into the note
        row['measured_train_s'], row['ref_train_s'] = t_var, t_ref
        row['train_s'] = ref['train_s'] * t_var / t_ref
    for row in rows.values():
        for sc in row['scenes'].values():
            sc.pop('seed_train', None)
    return rows


# ---------------------------------------------------------------- paper-style result tables (dashboard Result tab)
GPU_KEYS = {'NVIDIA RTX PRO 6000 Blackwell Server Edition': 'blackwell', 'NVIDIA RTX A6000': 'a6000', 'NVIDIA A100 80GB PCIe': 'a100'}
GROUPS = {'outdoor': OUTDOOR, 'indoor': INDOOR, 'mipnerf360': MIPNERF360, 'tandt': TANDT, 'db': DEEPBLENDING, 'all13': ALL13}
# runs on the A6000 and A100 before this date were development runs, not the GPU comparison (Table 4)
T4_SINCE = '2026-11-19'
# reference of each relative value: Faster-GS for Tables 1 and 2, Combo-Full for the Full - X ablation, 3DGS for Table 4
REFS = ('baseline', 'combofull3', '3dgs')
METRICS = ['psnr', 'ssim', 'lpips', 'train_s', 'vram_gib', 'n_gs_m']


def seed_target(variant: str) -> tuple:
    if variant in ABLATION or variant.startswith('fullx_'):
        return ABLATION_SEEDS
    return (0, 1, 2) if variant in SEEDS_FULL or variant in ('3dgs', 'taming') else (0,)


def result_tables(entries):
    """{gpu: {variant: {group: values}}}: quality is averaged over seeds, train time and VRAM use the per-scene median
    over seeds, then everything is averaged over the group's scenes. Only seeds that finished every scene of the group
    count. 'rel' holds paired ratios (reference / variant, on the seeds both finished) against each reference in REFS."""
    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for e in entries:
        g = GPU_KEYS.get(e.get('gpu'))
        if g is None or (g != 'blackwell' and e.get('finished', '') < T4_SINCE):
            continue
        runs[g][e['variant']][e['scene']].append(e)
    out = {}
    for g, variants in runs.items():
        by_seed = {}  # variant -> scene -> seed -> run (latest commit of the variant on that scene)
        for v, scenes in variants.items():
            by_seed[v] = {}
            for sc, rs in scenes.items():
                latest = max(rs, key=lambda r: r['finished'])['commit']
                by_seed[v][sc] = {r['seed']: r for r in rs if r['commit'] == latest}

        def agg(v, scenes, seeds):
            per = [[by_seed[v][sc][sd] for sd in seeds] for sc in scenes]
            res = {k: mean(mean(r[k] for r in rs) for rs in per) for k in ('psnr', 'ssim', 'lpips')}
            res['train_s'] = mean(median(r['train_s'] for r in rs) for rs in per)
            res['vram_gib'] = mean(median(r['vram_alloc_bytes'] for r in rs) for rs in per) / 2**30
            res['n_gs_m'] = mean(mean(r['n_gaussians'] for r in rs) for rs in per) / 1e6
            return res

        def has(v, sd, scenes):
            return all(sd in by_seed[v].get(sc, {}) for sc in scenes)

        def complete_seeds(v, scenes):
            return [sd for sd in seed_target(v) if has(v, sd, scenes)]

        out[g] = {}
        for v in by_seed:
            vres = {}
            for gname, scenes in GROUPS.items():
                seeds = complete_seeds(v, scenes)
                cell = {'seeds': len(seeds), 'target': len(seed_target(v)),
                        'scenes_done': sum(1 for sc in scenes if any(sd in by_seed[v].get(sc, {}) for sd in seed_target(v))), 'n_scenes': len(scenes)}
                if seeds:
                    cell.update(agg(v, scenes, seeds))
                    rel = {}
                    for ref in REFS:
                        if ref == v or ref not in by_seed:
                            continue
                        common = [sd for sd in seeds if has(ref, sd, scenes)]
                        if common:
                            a, b = agg(ref, scenes, common), agg(v, scenes, common)
                            rel[ref] = {'train': a['train_s'] / b['train_s'], 'vram': b['vram_gib'] / a['vram_gib'], 'seeds': len(common)}
                    cell['rel'] = rel
                vres[gname] = cell
            out[g][v] = vres
    return out


def update_dashboard(rows, entries=()):
    d = json.loads(DASH.read_text(encoding='utf-8'))
    table = [r for r in d['table']['rows'] if r.get('role') == 'reference' or r.get('example')]
    order = [v for v in ORDER if v in rows] + sorted(v for v in rows if v not in ORDER and v not in ('probe', 'baseline_dev'))
    for v in order:
        if v not in rows:
            continue
        r = rows[v]
        entry = {'method': DISPLAY.get(v, f'Faster-GS+{v}'), 'variant': v,
                 'note': NOTES.get(v, ''),
                 'status': 'done' if r['complete'] else ('running' if r['runs_done'] else 'pending'),
                 'runs_done': r['runs_done'], 'runs_total': r['runs_total'], 'seeds': r['seeds']}
        if v == 'baseline':
            entry['role'] = 'baseline'
        for k in ['psnr', 'ssim', 'lpips', 'train_s', 'vram_gib', 'n_gs_m']:
            entry[k] = r.get(k)
        if 'provisional' in r:
            entry['provisional'] = r['provisional']
        if 'speedup' in r:
            entry['speedup'] = r['speedup']
            entry['speedup_seeds'] = r['speedup_seeds']
        table.append(entry)
    for v in ORDER:
        if v not in rows and v != 'baseline':
            table.append({'method': DISPLAY[v], 'variant': v, 'note': NOTES.get(v, ''), 'status': 'pending', 'runs_done': 0,
                          'runs_total': 9 * target_seeds(v, 3), 'seeds': target_seeds(v, 3),
                          'psnr': None, 'ssim': None, 'lpips': None, 'train_s': None, 'vram_gib': None, 'n_gs_m': None})
    if 'baseline' not in rows:
        table.append({'method': 'Faster-GS', 'note': 'baseline · 本機量測', 'role': 'baseline', 'status': 'pending', 'runs_done': 0,
                      'runs_total': 27, 'psnr': None, 'ssim': None, 'lpips': None, 'train_s': None, 'vram_gib': None, 'n_gs_m': None})
    # reference first, then baseline, then others, then examples
    rank = {v: i for i, v in enumerate(ORDER)}
    table.sort(key=lambda r: (0 if r.get('role') == 'reference' else 1 if r.get('role') == 'baseline' else 3 if r.get('example') else 2,
                              rank.get(r.get('variant'), 99)))
    d['table']['rows'] = table
    d['scene_details'] = {v: rows[v]['scenes'] for v in order if v in rows}
    d['results_data'] = result_tables(entries)
    DASH.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--no-dashboard', action='store_true')
    args = parser.parse_args()
    rows = aggregate(load_registry(), args.seeds)
    for v, r in rows.items():
        print(f'== {v}: {r["runs_done"]}/{r["runs_total"]} runs' + (' (complete)' if r['complete'] else ''))
        for s, x in r['scenes'].items():
            print(f'   {s:9s} n={x["runs"]} PSNR {x["psnr"]:.2f} SSIM {x["ssim"]:.3f} LPIPS {x["lpips"]:.3f} '
                  f'train {x["train_s"]:.1f}s VRAM {x["vram_gib"]:.2f}GiB #Gs {x["n_gs_m"]:.2f}M  runs {x["psnr_runs"]} {x["train_runs"]}')
    if not args.no_dashboard:
        update_dashboard(rows, load_registry())
