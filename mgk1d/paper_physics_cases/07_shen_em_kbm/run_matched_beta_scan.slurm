#!/usr/bin/env bash
# Matched Shen beta scan: 18 beta points times two electromagnetic closures.
# Each array task runs one CGYRO case with eight OpenMP cores.
#SBATCH --job-name=shen25_matched
#SBATCH --partition=cpu
#SBATCH --array=0-35%24
#SBATCH --cpus-per-task=8
#SBATCH --mem=14G
#SBATCH --time=04:00:00
#SBATCH --output=/home/velo/codex_runs/cgyro_shen2025_matched_beta_scan_20260820/slurm_%A_%a.out
#SBATCH --error=/home/velo/codex_runs/cgyro_shen2025_matched_beta_scan_20260820/slurm_%A_%a.err

set -u

task_root=/home/velo/codex_runs/cgyro_shen2025_matched_beta_scan_20260820
gacode_root=/home/velo/codex_runs/cgyro_cbc_nr15_20260811/gacode
export GACODE_ROOT="$gacode_root"
export GACODE_PLATFORM=NODE04
export NODE04_DEPS=/home/velo/codex_runs/cgyro_cbc_nr15_20260811/deps/root
export PATH="$GACODE_ROOT/shared/bin:$GACODE_ROOT/cgyro/bin:$PATH"
export PYTHONPATH="$GACODE_ROOT/f2py/pygacode${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1

betas=(0.0080 0.0090 0.0100 0.0110 0.0120 0.0130 0.0140 0.0150 0.0160 0.0170 0.0180 0.0190 0.0200 0.0210 0.0220 0.0230 0.0240 0.0250)
fields=(2 3)
index="${SLURM_ARRAY_TASK_ID:-0}"
beta_index=$((index / 2))
field_index=$((index % 2))
beta="${betas[$beta_index]}"
n_field="${fields[$field_index]}"
beta_name=${beta/./p}
case_dir="$task_root/beta_${beta_name}_f${n_field}"

mkdir -p "$case_dir"
sed -e "s/@BETA@/$beta/" -e "s/@N_FIELD@/$n_field/" \
    "$task_root/input_matched_template.cgyro" > "$case_dir/input.cgyro"

cd "$case_dir" || exit 2
python3 "$GACODE_ROOT/cgyro/bin/cgyro_parse.py" > parse.log 2>&1 || {
    printf '1\n' > exit_code
    exit 2
}

start_seconds=$(date +%s)
if "$GACODE_ROOT/cgyro/src/cgyro" 0 > run.log 2>&1; then
    exit_code=0
else
    exit_code=$?
fi
end_seconds=$(date +%s)
printf '%s\n' "$exit_code" > exit_code
printf '%s\n' "$((end_seconds - start_seconds))" > wall_seconds.txt
exit "$exit_code"
