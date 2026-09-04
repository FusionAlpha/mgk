#!/bin/bash
set -eo pipefail

export GACODE_ROOT=/mnt/d/Codes/gacode-master
source "$GACODE_ROOT/shared/bin/gacode_setup" >/dev/null 2>&1
case_dir=/mnt/c/Users/THINK/Desktop/handmake/analysis/cgyro_miller_rewoldt2007_tem_nradial9
cd "$case_dir"
python3 "$GACODE_ROOT/cgyro/bin/cgyro_parse.py"
start=$SECONDS
OMP_NUM_THREADS=1 mpirun -np 8 "$GACODE_ROOT/cgyro/src/cgyro" 0 > run.log 2>&1
printf '%d\n' "$((SECONDS-start))" > wall_seconds.txt
tail -n 1 out.cgyro.freq
grep -m1 'Linear converged' out.cgyro.info
