#!/usr/bin/env bash
set -euo pipefail

task_root=${1:-/home/velo/codex_runs/cgyro_shen2025_matched_beta_scan_20260820}
output=${2:-"$task_root/cgyro_matched_beta_scan.csv"}
betas=(0.0080 0.0090 0.0100 0.0110 0.0120 0.0130 0.0140 0.0150 0.0160 0.0170 0.0180 0.0190 0.0200 0.0210 0.0220 0.0230 0.0240 0.0250)

printf 'betaElectron,betaPercent,nField,omegaNative,gammaNative,omegaRoverVti,gammaRoverVti,finalTime,wallSeconds,exitCode,linearConverged\n' > "$output"
for beta in "${betas[@]}"; do
    beta_name=${beta/./p}
    for n_field in 2 3; do
        case_dir="$task_root/beta_${beta_name}_f${n_field}"
        [[ -f "$case_dir/out.cgyro.freq" ]] || continue
        freq=$(awk 'NF >= 2 {r=$1; i=$2} END {if (r != "") printf "%.12g %.12g", r, i}' "$case_dir/out.cgyro.freq")
        [[ -n "$freq" ]] || continue
        read -r omega_native gamma_native <<< "$freq"
        final_time=nan
        if [[ -f "$case_dir/out.cgyro.time" ]]; then
            final_time=$(awk 'NF >= 1 {t=$1} END {printf "%.12g", t+0}' "$case_dir/out.cgyro.time")
        fi
        wall_seconds=nan
        if [[ -f "$case_dir/wall_seconds.txt" ]]; then
            wall_seconds=$(awk 'NF >= 1 {t=$1} END {printf "%.12g", t+0}' "$case_dir/wall_seconds.txt")
        fi
        exit_code=missing
        if [[ -f "$case_dir/exit_code" ]]; then
            exit_code=$(awk 'NF >= 1 {t=$1} END {printf "%s", t}' "$case_dir/exit_code")
        fi
        linear_converged=0
        if [[ -f "$case_dir/out.cgyro.info" ]] && grep -q 'EXIT: (CGYRO) Linear converged' "$case_dir/out.cgyro.info"; then
            linear_converged=1
        fi
        # CGYRO reports omega in v_ti/a; RMAJ=R0/a=5.555... for this benchmark.
        omega_rover_vti=$(awk -v x="$omega_native" 'BEGIN {printf "%.12g", 5.555555555555556*x}')
        gamma_rover_vti=$(awk -v x="$gamma_native" 'BEGIN {printf "%.12g", 5.555555555555556*x}')
        printf '%s,%.1f,%d,%s,%s,%s,%s,%s,%s,%s,%s\n' \
            "$beta" "$(awk -v x="$beta" 'BEGIN {printf "%.1f", 100*x}')" \
            "$n_field" "$omega_native" "$gamma_native" "$omega_rover_vti" \
            "$gamma_rover_vti" "$final_time" "$wall_seconds" "$exit_code" \
            "$linear_converged" >> "$output"
    done
done
printf 'wrote %s\n' "$output"
