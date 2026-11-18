#!/bin/bash
# A/B nsys comparison of variants on one GPU. usage: [TAG=ab] ab_nsys.sh <gpu> <scene> <captures> <variant>...
# TAG prefixes the output names (e.g. TAG=bw for the Blackwell captures, so the A6000 ones are not overwritten).
GPU=$1; SCENE=$2; CAPS=$3; shift 3
cd "$(dirname "$0")"
NSYS=${NSYS:-nsys}; PY=${PYTHON:-python}
N=$(echo $CAPS | tr ',' '\n' | wc -l)
for v in "$@"; do
  tag=${TAG:-ab}_${v}_${SCENE}
  mkdir -p nsys out ../logs
  rm -f nsys/$tag.*.nsys-rep nsys/$tag.*.sqlite
  $NSYS profile -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=repeat:$N -o nsys/$tag -f true \
    $PY prof_train.py --variant $v --gpu $GPU --scene $SCENE --capture $CAPS \
    --capture-len 50 --stats-len 0 --out out/$tag.json > ../logs/$tag.log 2>&1
  files=""
  for i in $(seq 1 $N); do $NSYS export --type sqlite -o nsys/$tag.$i.sqlite -f true nsys/$tag.$i.nsys-rep > /dev/null 2>&1; files="$files nsys/$tag.$i.sqlite"; done
  echo "== $v $SCENE"; python3 analyze_nsys.py out/nsys_$tag.json $files
done
