#!/usr/bin/env bash
set -eo pipefail

export GACODE_ROOT=${GACODE_ROOT:-/mnt/d/Codes/gacode-master}
source "$GACODE_ROOT/shared/bin/gacode_setup" >/dev/null 2>&1
set -u

root=/mnt/c/Users/THINK/Desktop/handmake/analysis/cgyro_standard_miller_triangularity
source_case=/mnt/c/Users/THINK/Desktop/handmake/analysis/cgyro_standard_miller_ae/ky030
template="$source_case/input.cgyro"
cgyro_exe=${CGYRO_EXE:-$GACODE_ROOT/cgyro/src/cgyro}
parser=${CGYRO_PARSER:-$GACODE_ROOT/cgyro/bin/cgyro_parse.py}
mpi_ranks=${MPI_RANKS:-1}

names=(delta_m0p4 delta_m0p2 delta_0p0 delta_p0p2 delta_p0p4)
deltas=(-0.4 -0.2 0.0 0.2 0.4)

mkdir -p "$root"
if [[ ! -f "$template" ]]; then
    echo "Missing sole input template: $template" >&2
    exit 1
fi
if [[ ! -x "$cgyro_exe" ]]; then
    echo "Missing CGYRO executable: $cgyro_exe" >&2
    exit 1
fi

# The circular case is copied verbatim from the converged standard-Miller run.
mkdir -p "$root/delta_0p0"
cp -a "$source_case"/. "$root/delta_0p0"/

replace_delta() {
    local destination=$1
    local delta=$2
    python3 - "$template" "$destination" "$delta" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_bytes()
destination = pathlib.Path(sys.argv[2])
delta = sys.argv[3].encode("ascii")
updated, count = re.subn(
    rb"(?m)^DELTA=[^\r\n]*",
    b"DELTA=" + delta,
    source,
)
if count != 1:
    raise SystemExit(f"Expected one DELTA line in {sys.argv[1]}, found {count}")
destination.write_bytes(updated)
PY
}

clear_outputs() {
    local directory=$1
    rm -f "$directory"/bin.cgyro.* "$directory"/out.cgyro.* \
        "$directory"/input.cgyro.gen "$directory"/run.log
}

for index in "${!names[@]}"; do
    name=${names[$index]}
    delta=${deltas[$index]}
    directory="$root/$name"
    mkdir -p "$directory"
    candidate="$directory/input.cgyro.new"
    replace_delta "$candidate" "$delta"
    if [[ -f "$directory/input.cgyro" ]] && ! cmp -s "$candidate" "$directory/input.cgyro"; then
        if [[ "$name" != delta_0p0 ]]; then
            clear_outputs "$directory"
        fi
    fi
    mv "$candidate" "$directory/input.cgyro"
done

# Normalize only the DELTA value, then require exact byte identity with the template.
python3 - "$template" "$root" "${names[@]}" <<'PY'
import pathlib
import re
import sys

template = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
names = sys.argv[3:]

def normalized(path):
    data = path.read_bytes()
    data, count = re.subn(
        rb"(?m)^DELTA=[^\r\n]*",
        b"DELTA=@DELTA@",
        data,
    )
    if count != 1:
        raise SystemExit(f"Expected one DELTA line in {path}, found {count}")
    return data

reference = normalized(template)
for name in names:
    path = root / name / "input.cgyro"
    if normalized(path) != reference:
        raise SystemExit(f"Input differs from the template outside DELTA: {path}")
print("Validated: all case inputs differ from the template only in DELTA.")
PY

run_case() {
    local name=$1
    local directory="$root/$name"
    if [[ -f "$directory/out.cgyro.info" ]] && \
            grep -q 'Linear converged' "$directory/out.cgyro.info"; then
        printf 'Reusing converged case: %s\n' "$name"
        return
    fi

    clear_outputs "$directory"
    (
        cd "$directory"
        python3 "$parser"
        OMP_NUM_THREADS=1 mpirun -np "$mpi_ranks" "$cgyro_exe" 0 \
            > run.log 2>&1
    )
}

pids=()
run_names=()
for name in "${names[@]}"; do
    if [[ "$name" == delta_0p0 ]]; then
        continue
    fi
    run_case "$name" &
    pids+=("$!")
    run_names+=("$name")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        printf 'CGYRO failed: %s\n' "${run_names[$index]}" >&2
        failed=1
    fi
done
if (( failed )); then
    exit 1
fi

summary="$root/cgyro_triangularity_summary.csv"
printf 'case,delta,omega_r,gamma,status\n' > "$summary"
for index in "${!names[@]}"; do
    name=${names[$index]}
    delta=${deltas[$index]}
    directory="$root/$name"
    if [[ ! -f "$directory/out.cgyro.info" ]] || \
            ! grep -q 'Linear converged' "$directory/out.cgyro.info"; then
        echo "Case did not report Linear converged: $name" >&2
        exit 1
    fi
    if [[ ! -s "$directory/out.cgyro.freq" ]]; then
        echo "Missing frequency history: $name" >&2
        exit 1
    fi
    read -r omega_r gamma < <(awk 'NF >= 2 {r=$1; g=$2} END {print r, g}' \
        "$directory/out.cgyro.freq")
    printf '%s,%s,%s,%s,Linear converged\n' \
        "$name" "$delta" "$omega_r" "$gamma" >> "$summary"
    printf '%-12s DELTA=%5s  omega=% .6e %+.6ei  Linear converged\n' \
        "$name" "$delta" "$omega_r" "$gamma"
done
