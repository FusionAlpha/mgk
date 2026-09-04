"""Compute the electrostatic Cyclone Base Case ITG dispersion curve.

Run the CBC adiabatic-electron ITG scan with mode tracking. The strategy is to
discover the fastest-growing mode at anchor points with
max_growth_scan, then track neighbouring ky points with secant-predicted
frequency shifts and the previous eigenvector as the Arnoldi initial vector.

CBC parameters follow the usual local definition at r/a = 0.5:
q=1.4, s_hat=0.8, r/R=0.18, R/Ln=2.2, R/LTi=6.9, and Ti/Te=1.
Kinetic ions, adiabatic electrons, open ballooning boundaries.
With Ti=Te, the code's rho_i is equal to rho_s.
"""

from __future__ import annotations

import copy
import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT))

import mgk

ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_UNIT = 1.66053906660e-27

RESIDUAL_LIMIT = 1e-6
ANCHOR_SHIFT_TIME_LIMIT = 20.0


def build_cbc_config():
    magnetic_field = 2.0
    major_radius = 1.0
    ion_mass = 2 * ATOMIC_MASS_UNIT
    temperature = 1000.0
    sound_speed = math.sqrt(temperature * ELEMENTARY_CHARGE / ion_mass)
    cyclotron_frequency = ELEMENTARY_CHARGE * magnetic_field / ion_mass
    rho_s = sound_speed / cyclotron_frequency
    frequency_reference = sound_speed / major_radius

    config = {
        "physical": {
            "magneticField": magnetic_field,
            "majorRadius": major_radius,
            "minorRadius": 0.18 * major_radius,
            "ionMass": ion_mass,
            "ionChargeNumber": 1,
            "ionTemperature": temperature,
            "electronTemperature": temperature,
            "electronBeta": 0.0,
            "densityGradientLength": major_radius / 2.2,
            "ionTemperatureGradientLength": major_radius / 6.9,
            "binormalWavenumber": 0.30 / rho_s,
        },
        "geometry": {
            "q": 1.4,
            "magneticShear": 0.8,
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
            "thetaMin": -4 * math.pi,
            "thetaMax": 4 * math.pi,
            "numTheta": 129,
            "energyMax": 12.5,
            "numEnergy": 28,
            "numPitch": 32,
            "numBouncePoints": 48,
        },
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "eigenBackend": "eigs",
            "modeSelection": "nearest",
            "singleShiftTimeLimit": math.inf,
            "eigenTolerance": 1e-8,
            "eigenSubspaceDimension": 24,
            "eigenMaxIterations": 1000,
            "gpuArnoldiMaxRestarts": 15,
            "enableGpuResultCache": False,
            "compactResult": True,
        },
    }
    return config, rho_s, frequency_reference


def is_converged(result) -> bool:
    return (np.isfinite(result.omega)
            and result.eigenResidual <= RESIDUAL_LIMIT
            and result.fieldConstraintResidual <= RESIDUAL_LIMIT)


def require_converged(result, ky_rho_s: float) -> None:
    if not is_converged(result):
        raise RuntimeError(
            f"CBC solve failed at ky*rho_s={ky_rho_s:.3f} "
            f"(eigen {result.eigenResidual:.3e}, "
            f"QN {result.fieldConstraintResidual:.3e}).")


def main():
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    config, rho_s, frequency_reference = build_cbc_config()

    ky_rho_s = np.arange(0.05, 0.65 + 1e-12, 0.05)
    num_points = ky_rho_s.size
    anchor_index = int(np.flatnonzero(np.isclose(ky_rho_s, 0.30))[0])

    omega = np.full(num_points, np.nan, dtype=complex)
    eigen_residual = np.full(num_points, np.nan)
    field_residual = np.full(num_points, np.nan)
    runtime_seconds = np.full(num_points, np.nan)

    def record(index, result):
        require_converged(result, ky_rho_s[index])
        omega[index] = result.omega
        eigen_residual[index] = result.eigenResidual
        field_residual[index] = result.fieldConstraintResidual
        runtime_seconds[index] = result.runtime

    def discover(index):
        cfg = copy.deepcopy(config)
        cfg["physical"]["binormalWavenumber"] = ky_rho_s[index] / rho_s
        cfg["solver"]["modeSelection"] = "max_growth_scan"
        cfg["solver"]["frequencyGuess"] = None
        cfg["solver"]["eigenInitialVector"] = None
        # A remote shift can otherwise consume all 1000 ARPACK iterations.
        # The scan already treats a timed-out shift as invalid and selects
        # the fastest-growing mode from the remaining converged candidates.
        cfg["solver"]["singleShiftTimeLimit"] = ANCHOR_SHIFT_TIME_LIMIT
        print(f"starting discover ky*rho_s={ky_rho_s[index]:.2f} "
              f"({ANCHOR_SHIFT_TIME_LIMIT:.0f}s/shift limit)", flush=True)
        result = mgk.solve(cfg)
        record(index, result)
        num_valid = int(np.count_nonzero(result.modeScan.valid))
        num_timed_out = int(np.count_nonzero(result.modeScan.timedOut))
        print(f"discover ky*rho_s={ky_rho_s[index]:.2f} "
              f"omega={result.omega.real:+.9f}{result.omega.imag:+.9f}i "
              f"residual={result.eigenResidual:.2e} valid={num_valid} "
              f"timed_out={num_timed_out} "
              f"time={result.runtime:.2f}s", flush=True)
        return result

    def track(indices, previous, earlier_omega):
        previous_omega = previous.omega
        for index in indices:
            cfg = copy.deepcopy(config)
            cfg["physical"]["binormalWavenumber"] = ky_rho_s[index] / rho_s
            cfg["solver"]["modeSelection"] = "nearest"
            guess_omega = previous_omega
            if earlier_omega is not None:
                guess_omega = 2 * previous_omega - earlier_omega
            cfg["solver"]["frequencyGuess"] = guess_omega * frequency_reference
            cfg["solver"]["eigenInitialVector"] = previous.reducedMode
            current = mgk.solve(cfg)
            if not is_converged(current):
                print(f"retry ky*rho_s={ky_rho_s[index]:.2f} "
                      f"with a larger Arnoldi subspace", flush=True)
                cfg["solver"]["eigenTolerance"] = 1e-9
                cfg["solver"]["eigenSubspaceDimension"] = 48
                cfg["solver"]["gpuArnoldiMaxRestarts"] = 40
                current = mgk.solve(cfg)
            record(index, current)
            print(f"ky*rho_s={ky_rho_s[index]:.2f} "
                  f"omega={current.omega.real:+.9f}{current.omega.imag:+.9f}i "
                  f"residual={current.eigenResidual:.2e} "
                  f"time={current.runtime:.2f}s", flush=True)
            earlier_omega = previous_omega
            previous_omega = current.omega
            previous = current
        return previous

    # Use a central anchor at ky=0.30,
    # a re-discovered anchor at ky=0.60 to catch the branch switch, and
    # tracking in between.
    anchor = discover(anchor_index)
    track(range(anchor_index - 1, 0, -1), anchor, None)
    discover(0)

    upper_anchor_index = int(np.flatnonzero(np.isclose(ky_rho_s, 0.55))[0])
    upper = track(range(anchor_index + 1, upper_anchor_index + 1), anchor, None)
    second_anchor_index = int(np.flatnonzero(np.isclose(ky_rho_s, 0.60))[0])
    second_anchor = discover(second_anchor_index)
    track(range(second_anchor_index + 1, num_points), second_anchor, upper.omega)

    a_over_r = 2 * 0.18  # r/a=0.5 and r/R=0.18
    csv_file = output_dir / "cbc_dispersion_curve_python.csv"
    with csv_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kyRhoS", "omegaRoverCs", "gammaRoverCs",
                         "omegaAoverCs", "gammaAoverCs", "eigenResidual",
                         "fieldConstraintResidual", "runtimeSeconds"])
        for i in range(num_points):
            writer.writerow([
                f"{ky_rho_s[i]:.2f}",
                repr(float(omega[i].real)), repr(float(omega[i].imag)),
                repr(float(a_over_r * omega[i].real)),
                repr(float(a_over_r * omega[i].imag)),
                repr(float(eigen_residual[i])), repr(float(field_residual[i])),
                repr(float(runtime_seconds[i])),
            ])

    peak_index = int(np.nanargmax(a_over_r * omega.imag))
    print(f"CBC peak: ky*rho_s={ky_rho_s[peak_index]:.2f}, "
          f"omega*a/c_s={a_over_r * omega[peak_index].real:+.6f}, "
          f"gamma*a/c_s={a_over_r * omega[peak_index].imag:.6f}")
    print(f"wrote {csv_file}")


if __name__ == "__main__":
    main()
