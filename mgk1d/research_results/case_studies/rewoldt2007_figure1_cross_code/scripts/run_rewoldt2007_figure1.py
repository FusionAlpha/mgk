#!/usr/bin/env python3
"""Run the gk-eig reproduction of Rewoldt et al. (2007), Figure 1.

The benchmark is collisionless and electrostatic, with one kinetic ion and
adiabatic electrons. Frequencies returned by gk-eig in c_s/R are converted to
the paper unit c_s/L_n by division by R/L_n=2.22.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parents[1]
PROJECT_ROOT = HERE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
R_OVER_LN = 2.22
R_OVER_LTI = 6.92
KY_POINTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)


def physical_scales() -> dict[str, float]:
    elementary_charge = 1.602176634e-19
    atomic_mass_unit = 1.66053906660e-27
    magnetic_field = 2.0
    major_radius = 1.0
    ion_mass = 2.0 * atomic_mass_unit
    temperature = 1000.0
    sound_speed = math.sqrt(temperature * elementary_charge / ion_mass)
    cyclotron_frequency = elementary_charge * magnetic_field / ion_mass
    return {
        "elementary_charge": elementary_charge,
        "magnetic_field": magnetic_field,
        "major_radius": major_radius,
        "ion_mass": ion_mass,
        "temperature": temperature,
        "sound_speed": sound_speed,
        "rho_i": sound_speed / cyclotron_frequency,
    }


def base_config(quick: bool) -> tuple[dict, dict[str, float]]:
    scales = physical_scales()
    radius = scales["major_radius"]
    if quick:
        resolution = {
            "numTheta": 65,
            "numEnergy": 12,
            "numPitch": 24,
            "numBouncePoints": 24,
        }
        tolerance = 1.0e-8
    else:
        resolution = {
            "numTheta": 129,
            "numEnergy": 24,
            "numPitch": 48,
            "numBouncePoints": 48,
        }
        tolerance = 1.0e-10
    config = {
        "physical": {
            "magneticField": scales["magnetic_field"],
            "majorRadius": radius,
            # In the local model this is the reference-surface r/R=0.18.
            "minorRadius": 0.18 * radius,
            "ionMass": scales["ion_mass"],
            "ionChargeNumber": 1,
            "ionTemperature": scales["temperature"],
            "electronTemperature": scales["temperature"],
            "electronBeta": 0.0,
            "densityGradientLength": radius / R_OVER_LN,
            "ionTemperatureGradientLength": radius / R_OVER_LTI,
            "binormalWavenumber": 0.3 / scales["rho_i"],
        },
        "geometry": {
            "q": 1.4,
            "magneticShear": 0.776,
            "alpha": 0.0,
            "ballooningAngle": 0.0,
        },
        "model": {
            "magneticMirror": True,
            "mirrorConvention": "cgyro_s_alpha",
            "aparallel": False,
            "parallelBoundary": "open",
        },
        "grid": {
            "thetaMin": -4.0 * math.pi,
            "thetaMax": 4.0 * math.pi,
            "energyMax": 12.5,
            **resolution,
        },
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "eigenBackend": "eigs",
            "modeSelection": "nearest",
            "singleShiftTimeLimit": math.inf,
            "eigenTolerance": tolerance,
            "eigenSubspaceDimension": 8,
            "eigenHotSubspaceDimension": 5,
            "enableWarmRitz": False,
            "cpuFactorizationWorkers": min(8, os.cpu_count() or 1),
            "cpuOperatorWorkers": min(8, os.cpu_count() or 1),
            "compactResult": True,
            "returnMatrices": False,
        },
    }
    return config, scales


def solve_order() -> tuple[float, ...]:
    """Start at the peak, then continue down and up the same ITG branch."""
    return (0.3, 0.2, 0.1, 0.4, 0.5, 0.6)


def run(quick: bool) -> list[dict[str, float | int | bool]]:
    import mgk

    config, scales = base_config(quick)
    frequency_reference = scales["sound_speed"] / scales["major_radius"]
    results: dict[float, dict[str, float | int | bool]] = {}
    branch_state: dict[str, tuple[object, complex, float, complex | None, float | None]] = {}

    for ky in solve_order():
        direction = "down" if ky < 0.3 else "up"
        if ky == 0.3:
            previous_result = None
            previous_omega = None
            earlier_omega = None
            previous_ky = None
            earlier_ky = None
        else:
            (
                previous_result,
                previous_omega,
                previous_ky,
                earlier_omega,
                earlier_ky,
            ) = branch_state[direction]
        config["physical"]["binormalWavenumber"] = ky / scales["rho_i"]
        if previous_omega is None:
            guess = -0.78 + 0.265j if ky == 0.3 else -2.6 * ky + 0.8j * ky
        elif earlier_omega is None:
            guess = previous_omega * (ky / previous_ky)
        else:
            step_ratio = (ky - previous_ky) / (previous_ky - earlier_ky)
            guess = previous_omega + step_ratio * (previous_omega - earlier_omega)
        config["solver"]["frequencyGuess"] = guess * frequency_reference
        config["solver"]["eigenInitialVector"] = (
            None if previous_result is None else previous_result.reducedMode
        )

        started = time.perf_counter()
        result = mgk.solve(config)
        wall_seconds = time.perf_counter() - started
        if result.eigenResidual > 1.0e-7 or result.fieldConstraintResidual > 1.0e-9:
            raise RuntimeError(
                f"unconverged gk-eig point ky={ky}: eigen={result.eigenResidual:.3e}, "
                f"field={result.fieldConstraintResidual:.3e}"
            )
        eigs_info = result.get("eigsInfo", {})
        row: dict[str, float | int | bool] = {
            "ky_rhoi": ky,
            "omega_R_over_cs": float(result.omega.real),
            "gamma_R_over_cs": float(result.omega.imag),
            "omega_cs_over_Ln": float(result.omega.real / R_OVER_LN),
            "gamma_cs_over_Ln": float(result.omega.imag / R_OVER_LN),
            "eigen_residual": float(result.eigenResidual),
            "field_residual": float(result.fieldConstraintResidual),
            "wall_seconds": wall_seconds,
            "assembly_seconds": float(result.timing.assembly),
            "factorization_seconds": float(result.timing.factorization),
            "eigensolve_seconds": float(result.timing.eigensolve),
            "operator_calls": int(eigs_info.get("operatorCalls", -1)),
            "quick": quick,
        }
        results[ky] = row
        print(
            f"gk-eig ky={ky:.1f}: omega Ln/cs={row['omega_cs_over_Ln']:+.6f}, "
            f"gamma Ln/cs={row['gamma_cs_over_Ln']:.6f}, "
            f"residual={row['eigen_residual']:.2e}, wall={wall_seconds:.2f}s",
            flush=True,
        )
        if ky == 0.3:
            anchor = (result, complex(result.omega), ky, None, None)
            branch_state["down"] = anchor
            branch_state["up"] = anchor
        else:
            branch_state[direction] = (
                result,
                complex(result.omega),
                ky,
                previous_omega,
                previous_ky,
            )
    return [results[ky] for ky in KY_POINTS]


def write_results(rows: list[dict[str, float | int | bool]], quick: bool) -> None:
    suffix = "_quick" if quick else ""
    output_dir = CASE_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"mgk_results{suffix}.csv"
    json_path = output_dir / f"mgk_results{suffix}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "paper": "Rewoldt et al., Computer Physics Communications 177 (2007) 775-780",
        "target": "Figure 1, electrostatic ITG with adiabatic electrons",
        "normalization": "gk-eig c_s/R divided by R/L_n=2.22",
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="run a reduced-grid smoke scan")
    args = parser.parse_args()
    rows = run(args.quick)
    write_results(rows, args.quick)


if __name__ == "__main__":
    main()
