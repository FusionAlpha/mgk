#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
template="$root/input_template.cgyro"
GACODE_ROOT=${GACODE_ROOT:-/mnt/d/Codes/gacode-master}
CGYRO_EXE=${CGYRO_EXE:-$GACODE_ROOT/cgyro/src/cgyro}
PYTHON=${PYTHON:-python3}
NPROC=${NPROC:-4}
OMP_THREADS=${OMP_THREADS:-1}
scan_mode=theta
action=${1:-run}
if [[ "$action" == radial ]]; then
    scan_mode=radial
    action=${2:-run}
elif [[ "$action" == performance ]]; then
    scan_mode=performance
    action=${2:-run}
fi

if [[ "$scan_mode" == radial ]]; then
    runs_root="$root/radial_runs"
    ntheta_values=(20)
    read -r -a nradial_values <<< "${NRADIAL_VALUES:-5 7 9}"
elif [[ "$scan_mode" == performance ]]; then
    runs_root="$root/performance_runs"
    read -r -a ntheta_values <<< "${NTHETA_VALUES:-8 12 16 20 24 32}"
    nradial_values=(4)
else
    runs_root="$root/runs"
    read -r -a ntheta_values <<< "${NTHETA_VALUES:-8 12 16 20 24 32}"
    nradial_values=(5)
fi

case_dir() {
    local ntheta=$1 nradial=$2 eta=$3
    if [[ "$scan_mode" == radial ]]; then
        printf '%s/nradial_%03d/eta_%s' "$runs_root" "$nradial" \
            "${eta/./p}"
    else
        printf '%s/ntheta_%03d/eta_%s' "$runs_root" "$ntheta" \
            "${eta/./p}"
    fi
}

prepare_case() {
    local ntheta=$1 nradial=$2 eta=$3 directory candidate
    directory=$(case_dir "$ntheta" "$nradial" "$eta")
    mkdir -p "$directory"
    candidate="$directory/input.cgyro.new"
    sed -e "s/@NTHETA@/$ntheta/g" -e "s/@NRADIAL@/$nradial/g" \
        -e "s/@ETAI@/$eta/g" \
        "$template" > "$candidate"
    if [[ -f "$directory/.complete" ]] && \
            ! cmp -s "$candidate" "$directory/input.cgyro"; then
        rm -f "$candidate"
        printf 'Completed case has different input: %s\n' "$directory" >&2
        return 1
    fi
    mv "$candidate" "$directory/input.cgyro"
    printf 'ntheta\t%s\nnradial\t%s\neta\t%s\nemax\t8.0\n' \
        "$ntheta" "$nradial" "$eta" \
        > "$directory/case_meta.tsv"
}

prepare_all() {
    local ntheta nradial eta
    for ntheta in "${ntheta_values[@]}"; do
        for nradial in "${nradial_values[@]}"; do
            for eta in 2.3 2.4 2.5 2.6 2.7; do
                prepare_case "$ntheta" "$nradial" "$eta"
            done
        done
    done
}

clear_partial_outputs() {
    local directory=$1
    rm -f "$directory"/bin.cgyro.* "$directory"/out.cgyro.* \
        "$directory"/input.cgyro.gen "$directory"/run.log \
        "$directory"/wall_seconds.txt "$directory"/exit_status.txt
}

run_case() {
    local ntheta=$1 nradial=$2 eta=$3 seed=${4:-} directory status
    directory=$(case_dir "$ntheta" "$nradial" "$eta")
    if [[ -f "$directory/.complete" ]]; then
        printf 'SKIP ntheta=%s nradial=%s eta=%s\n' \
            "$ntheta" "$nradial" "$eta"
        return
    fi
    clear_partial_outputs "$directory"
    if [[ -n "$seed" ]]; then
        cp "$(case_dir "$ntheta" "$nradial" "$seed")/bin.cgyro.restart" \
            "$directory/bin.cgyro.restart"
    fi
    (
        cd "$directory"
        "$PYTHON" "$GACODE_ROOT/cgyro/bin/cgyro_parse.py"
        set +e
        OMP_NUM_THREADS="$OMP_THREADS" /usr/bin/time -f '%e' \
            -o wall_seconds.txt mpirun -np "$NPROC" "$CGYRO_EXE" 0 \
            > run.log 2>&1
        status=$?
        set -e
        printf '%d\n' "$status" > exit_status.txt
        if (( status == 0 )); then touch .complete; fi
        exit "$status"
    )
}

run_sequence() {
    local ntheta=$1 nradial=$2
    run_case "$ntheta" "$nradial" 2.5
    run_case "$ntheta" "$nradial" 2.4 2.5
    run_case "$ntheta" "$nradial" 2.3 2.4
    run_case "$ntheta" "$nradial" 2.6 2.5
    run_case "$ntheta" "$nradial" 2.7 2.6
}

prepare_all
if [[ "$action" == prepare ]]; then
    exit 0
fi
set +u
source "$GACODE_ROOT/shared/bin/gacode_setup" >/dev/null 2>&1
set -u
for ntheta in "${ntheta_values[@]}"; do
    for nradial in "${nradial_values[@]}"; do
        run_sequence "$ntheta" "$nradial"
    done
done
