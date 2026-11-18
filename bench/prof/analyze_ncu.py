"""Extracts the metrics used for bottleneck classification from an ncu report.

usage: python analyze_ncu.py <report.ncu-rep> <out.json>
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

NCU = os.environ.get('NCU', 'ncu')
METRICS = {
    'duration_us': 'gpu__time_duration.sum',
    'dram_pct': 'gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed',
    'sm_pct': 'sm__throughput.avg.pct_of_peak_sustained_elapsed',
    'mem_pct': 'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed',
    'l1_pct': 'l1tex__throughput.avg.pct_of_peak_sustained_active',
    'l2_pct': 'lts__throughput.avg.pct_of_peak_sustained_elapsed',
    'achieved_occ': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'theoretical_occ': 'sm__maximum_warps_per_active_cycle_pct',
    'limit_blocks': 'launch__occupancy_limit_blocks',
    'limit_regs': 'launch__occupancy_limit_registers',
    'limit_smem': 'launch__occupancy_limit_shared_mem',
    'limit_warps': 'launch__occupancy_limit_warps',
    'regs': 'launch__registers_per_thread',
    'block': 'launch__block_size',
    'grid': 'launch__grid_size',
    'l1_hit': 'l1tex__t_sector_hit_rate.pct',
    'l2_hit': 'lts__t_sector_hit_rate.pct',
    'ld_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'ld_requests': 'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'st_sectors': 'l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum',
    'st_requests': 'l1tex__t_requests_pipe_lsu_mem_global_op_st.sum',
    'dram_read_bytes': 'dram__bytes_read.sum',
    'dram_write_bytes': 'dram__bytes_write.sum',
    'warp_eff': 'smsp__thread_inst_executed_per_inst_executed.ratio',
    'ipc': 'sm__inst_executed.avg.per_cycle_active',
    'fma_pct': 'sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active',
    'alu_pct': 'sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active',
    'xu_pct': 'sm__pipe_xu_cycles_active.avg.pct_of_peak_sustained_active',
    'atom_red_requests': 'l1tex__t_requests_pipe_lsu_mem_global_op_red.sum',
}
STALL_PREFIX = 'smsp__average_warps_issue_stalled_'
STALL_SUFFIX = '_per_issue_active.ratio'


def to_float(v):
    try:
        return float(v.replace(',', ''))
    except (ValueError, AttributeError):
        return None


def main(rep, out):
    raw = subprocess.run([NCU, '--import', rep, '--csv', '--page', 'raw'], capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(raw)))
    header, units, data = rows[0], rows[1], rows[2:]
    col = {name: i for i, name in enumerate(header)}
    kernels = defaultdict(list)
    for r in data:
        name = re.sub(r'\(.*', '', r[col['Kernel Name']])
        name = name.replace('void ', '').split('::')[-1].split('<')[0]
        k = {}
        for key, metric in METRICS.items():
            if metric in col:
                v = to_float(r[col[metric]])
                unit = units[col[metric]]
                if v is not None:
                    v *= {'Gbyte': 2**30, 'Mbyte': 2**20, 'Kbyte': 2**10, 'Tbyte': 2**40}.get(unit, 1.0)
                    v *= {'ms': 1000.0, 'msecond': 1000.0, 'ns': 1e-3, 'nsecond': 1e-3, 's': 1e6, 'second': 1e6}.get(unit, 1.0) if key == 'duration_us' else 1.0
                k[key] = v
        stalls = {}
        for metric, i in col.items():
            if metric.startswith(STALL_PREFIX) and metric.endswith(STALL_SUFFIX):
                v = to_float(r[i])
                if v:
                    stalls[metric[len(STALL_PREFIX):-len(STALL_SUFFIX)]] = v
        total = sum(stalls.values()) or 1.0
        k['stalls_top'] = sorted(((s, round(v / total * 100, 1)) for s, v in stalls.items()), key=lambda x: -x[1])[:5]
        if k.get('ld_requests'):
            k['ld_sectors_per_req'] = k['ld_sectors'] / k['ld_requests']
        if k.get('st_requests'):
            k['st_sectors_per_req'] = k['st_sectors'] / k['st_requests']
        kernels[name].append(k)
    with open(out, 'w') as f:
        json.dump(kernels, f, indent=1)
    for name, ks in kernels.items():
        for k in ks:
            g = lambda key, fmt='{:.1f}': fmt.format(k[key]) if k.get(key) is not None else '-'
            print(f"{name:28s} {g('duration_us')}us DRAM {g('dram_pct')}% SM {g('sm_pct')}% L1 {g('l1_pct')}% L2 {g('l2_pct')}% "
                  f"occ {g('achieved_occ')}/{g('theoretical_occ')}% lim(b/r/s/w) {g('limit_blocks','{:.0f}')}/{g('limit_regs','{:.0f}')}/{g('limit_smem','{:.0f}')}/{g('limit_warps','{:.0f}')} "
                  f"regs {g('regs','{:.0f}')} ld s/r {g('ld_sectors_per_req')} L1hit {g('l1_hit')} L2hit {g('l2_hit')} warp_eff {g('warp_eff')} "
                  f"R {k.get('dram_read_bytes', 0) / 2**20:.0f}MiB W {k.get('dram_write_bytes', 0) / 2**20:.0f}MiB fma {g('fma_pct')} xu {g('xu_pct')}")
            print(f"{'':28s} stalls: {k['stalls_top']}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
