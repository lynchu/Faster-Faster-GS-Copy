#!/bin/bash
# Nsight Compute profile of the main Faster-GS kernels for one training iteration.
# usage: [NCU_GPU=3] run_ncu.sh <variant> <scene> <iteration> <tag> [kernel regex] [launch count]
# Runs as root through the passwordless `sudo ncu` rule; the harness configures its own environment
# (sudo drops env vars), runs with python -B (no root-owned .pyc files) and writes NeRFICG outputs to /tmp.
set -euo pipefail
VARIANT=$1; SCENE=$2; ITER=$3; TAG=$4
KREGEX=${5:-'regex:preprocess_cu|blend_cu|blend_backward|preprocess_backward|adam_step|create_instances'}
BENCH=$(cd "$(dirname "$0")/.." && pwd)
NCU=${NCU:-ncu}; PY=${PYTHON:-python}
OUT=$BENCH/prof/ncu
mkdir -p "$OUT"
sudo -n "$NCU" \
  --profile-from-start off --target-processes all \
  --set full --kernel-name "$KREGEX" --launch-count ${6:-11} \
  --export "$OUT/${TAG}" --force-overwrite \
  "$PY" -B "$BENCH/prof/prof_train.py" \
    --variant "$VARIANT" --scene "$SCENE" --gpu ${NCU_GPU:-0} --capture "$ITER" --capture-len 1 --stats-len 0 \
    --out "$OUT/${TAG}_stats.json" --out-root /tmp/fgs_ncu_out
echo "report: $OUT/${TAG}.ncu-rep"
