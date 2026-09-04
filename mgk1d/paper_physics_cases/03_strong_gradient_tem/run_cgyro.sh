#!/bin/bash
set -eo pipefail

export GACODE_ROOT=/mnt/d/Codes/gacode-master
case_dir=/mnt/c/Users/THINK/Desktop/handmake/analysis/cgyro_tem_eps0p018_etai0_etae3p13_ky07_nradial17_scaled_restart
source "$GACODE_ROOT/shared/bin/gacode_setup"
set -u
cd "$case_dir"

python3 "$GACODE_ROOT/cgyro/bin/cgyro_parse.py"
export OMP_NUM_THREADS=1
/usr/bin/time -f '%e' -o wall_seconds.txt \
    mpirun -np 4 "$GACODE_ROOT/cgyro/src/cgyro" 0 > run.log 2>&1

tail -n 5 out.cgyro.freq
grep -E 'Linear converged|EXIT' out.cgyro.info | tail -n 3
printf 'wall_seconds='
cat wall_seconds.txt
