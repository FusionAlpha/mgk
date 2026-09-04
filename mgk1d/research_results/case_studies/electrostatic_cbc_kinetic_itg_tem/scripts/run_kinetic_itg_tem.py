"""Compute kinetic-electron ITG and TEM branches for the Cyclone Base Case.

Run the CBC kinetic-electron ITG/TEM scan with dual-branch mode tracking:
discover both the negative-frequency (ITG)
and positive-frequency (TEM) roots at the ky=0.30 anchor on a coarse grid with
max_growth_scan, then track each branch across ky on the production grid with
secant-predicted frequency shifts and the previous eigenvector as the Arnoldi
initial vector.

Parameters: q=1.4, s_hat=0.8, r/R=0.18, R/Ln=2.2, R/LTi=R/LTe=6.9, Ti/Te=1,
beta=0, physical electron mass. Both species use mirror-orbit coordinates and
open ballooning boundaries.
"""

from __future__ import annotations

import copy
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import mgk

ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_UNIT = 1.66053906660e-27
ELECTRON_MASS = 9.1093837139e-31

RESIDUAL_LIMIT = 1e-7
DISCOVERY_SHIFT_TIME_LIMIT = 60.0


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
            "electronTemperatureGradientLength": major_radius / 6.9,
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
        "species": {
            "enabled": True,
            "items": [{
                "kind": "electron",
                "kinetic": True,
                "mass": ELECTRON_MASS,
                "temperatureGradientLength": major_radius / 6.9,
            }],
        },
        "grid": {
            "thetaMin": -4 * math.pi,
            "thetaMax": 4 * math.pi,
            "numTheta": 97,
            "energyMax": 12.5,
            "numEnergy": 16,
            "numPitch": 32,
            "numBouncePoints": 48,
        },
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "eigenBackend": "eigs",
            "modeSelection": "nearest",
            "singleShiftTimeLimit": math.inf,
            "eigenTolerance": 1e-9,
            "eigenSubspaceDimension": 80,
            "eigenMaxIterations": 1200,
            "enableGpuResultCache": False,
            "compactResult": True,
        },
    }
    return config, rho_s, frequency_reference


def require_converged(result, label, ky_rho_s):
    if (not np.isfinite(result.omega)
            or result.eigenResidual > RESIDUAL_LIMIT
            or result.fieldConstraintResidual > RESIDUAL_LIMIT):
        raise RuntimeError(
            f"CBC {label} solve failed at ky*rho_s={ky_rho_s:.3f} "
            f"(eigen {result.eigenResidual:.3e}, "
            f"QN {result.fieldConstraintResidual:.3e}).")


def deduplicate(roots):
    kept = []
    for root in roots:
        if all(abs(k - root) > 1e-3 * max(1, abs(root)) for k in kept):
            kept.append(root)
    return np.asarray(kept)


def discover_branches(config, ky_rho_s, rho_s):
    """Coarse-grid max_growth_scan at the anchor ky; split roots by sign(omega_r)."""
    cfg = copy.deepcopy(config)
    cfg["grid"].update(numTheta=65, numEnergy=12, numPitch=24,
                       numBouncePoints=32)
    cfg["physical"]["binormalWavenumber"] = ky_rho_s / rho_s
    cfg["solver"]["modeSelection"] = "max_growth_scan"
    cfg["solver"]["frequencyGuess"] = None
    cfg["solver"]["eigenInitialVector"] = None
    cfg["solver"]["singleShiftTimeLimit"] = DISCOVERY_SHIFT_TIME_LIMIT
    discovery = mgk.solve(cfg)
    scan = discovery.modeScan
    valid_roots = deduplicate(np.asarray(scan.eigenvalues)[np.asarray(scan.valid)])
    itg_roots = valid_roots[valid_roots.real < 0]
    tem_roots = valid_roots[valid_roots.real > 0]
    if itg_roots.size == 0 or tem_roots.size == 0:
        raise RuntimeError(
            "CBC discovery did not find both negative- and "
            f"positive-frequency branches (roots: {valid_roots}).")
    itg_seed = itg_roots[np.argmax(itg_roots.imag)]
    tem_seed = tem_roots[np.argmax(tem_roots.imag)]
    return itg_seed, tem_seed, discovery


def track_branch(config, ky_rho_s, anchor_index, seed, rho_s,
                 frequency_reference, label):
    num_points = ky_rho_s.size
    branch = {
        "omega": np.full(num_points, np.nan, dtype=complex),
        "eigenResidual": np.full(num_points, np.nan),
        "fieldResidual": np.full(num_points, np.nan),
        "modeOverlap": np.full(num_points, np.nan),
        "runtimeSeconds": np.full(num_points, np.nan),
    }

    def record(index, result, overlap):
        require_converged(result, label, ky_rho_s[index])
        branch["omega"][index] = result.omega
        branch["eigenResidual"][index] = result.eigenResidual
        branch["fieldResidual"][index] = result.fieldConstraintResidual
        branch["modeOverlap"][index] = overlap
        branch["runtimeSeconds"][index] = result.runtime
        print(f"{label} ky*rho_s={ky_rho_s[index]:.2f} "
              f"omega={result.omega.real:+.9f}{result.omega.imag:+.9f}i "
              f"residual={result.eigenResidual:.2e} overlap={overlap:.5f} "
              f"time={result.runtime:.2f}s", flush=True)

    def solve_at(index, guess_omega, initial_vector):
        cfg = copy.deepcopy(config)
        cfg["physical"]["binormalWavenumber"] = ky_rho_s[index] / rho_s
        cfg["solver"]["frequencyGuess"] = guess_omega * frequency_reference
        cfg["solver"]["eigenInitialVector"] = initial_vector
        return mgk.solve(cfg)

    anchor = solve_at(anchor_index, seed, None)
    record(anchor_index, anchor, np.nan)

    def track(indices, previous):
        previous_omega = previous.omega
        earlier_omega = None
        for index in indices:
            guess_omega = previous_omega
            if earlier_omega is not None:
                guess_omega = 2 * previous_omega - earlier_omega
            current = solve_at(index, guess_omega, previous.reducedMode)
            overlap = abs(np.vdot(previous.reducedMode, current.reducedMode)) / (
                np.linalg.norm(previous.reducedMode)
                * np.linalg.norm(current.reducedMode))
            record(index, current, overlap)
            earlier_omega = previous_omega
            previous_omega = current.omega
            previous = current

    track(range(anchor_index - 1, -1, -1), anchor)
    track(range(anchor_index + 1, num_points), anchor)
    return branch


def main():
    output_dir = Path(__file__).resolve().parents[1] / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    config, rho_s, frequency_reference = build_cbc_config()

    ky_rho_s = np.arange(0.05, 0.65 + 1e-12, 0.05)
    anchor_index = int(np.flatnonzero(np.isclose(ky_rho_s, 0.30))[0])

    itg_seed, tem_seed, discovery = discover_branches(
        config, ky_rho_s[anchor_index], rho_s)
    num_valid = int(np.count_nonzero(discovery.modeScan.valid))
    num_timed_out = int(np.count_nonzero(discovery.modeScan.timedOut))
    print(f"coarse discovery: ITG={itg_seed.real:+.8f}{itg_seed.imag:+.8f}i, "
          f"TEM={tem_seed.real:+.8f}{tem_seed.imag:+.8f}i, "
          f"valid={num_valid}, timed_out={num_timed_out}, "
          f"time={discovery.runtime:.2f}s", flush=True)

    itg = track_branch(config, ky_rho_s, anchor_index, itg_seed,
                       rho_s, frequency_reference, "ITG")
    tem = track_branch(config, ky_rho_s, anchor_index, tem_seed,
                       rho_s, frequency_reference, "TEM")

    a_over_r = 2 * 0.18  # r/a=0.5 and r/R=0.18
    dominant = np.where(tem["omega"].imag > itg["omega"].imag, "TEM", "ITG")
    dominant_gamma = a_over_r * np.maximum(itg["omega"].imag, tem["omega"].imag)

    csv_file = output_dir / "cbc_kinetic_electron_dispersion_curve_python.csv"
    with csv_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "kyRhoS",
            "itgOmegaRoverCs", "itgGammaRoverCs",
            "temOmegaRoverCs", "temGammaRoverCs",
            "itgOmegaAoverCs", "itgGammaAoverCs",
            "temOmegaAoverCs", "temGammaAoverCs",
            "dominantBranch", "dominantGammaAoverCs",
            "itgEigenResidual", "temEigenResidual",
            "itgFieldResidual", "temFieldResidual",
            "itgModeOverlap", "temModeOverlap",
            "itgRuntimeSeconds", "temRuntimeSeconds",
        ])
        for i in range(ky_rho_s.size):
            writer.writerow([
                f"{ky_rho_s[i]:.2f}",
                repr(float(itg["omega"][i].real)),
                repr(float(itg["omega"][i].imag)),
                repr(float(tem["omega"][i].real)),
                repr(float(tem["omega"][i].imag)),
                repr(float(a_over_r * itg["omega"][i].real)),
                repr(float(a_over_r * itg["omega"][i].imag)),
                repr(float(a_over_r * tem["omega"][i].real)),
                repr(float(a_over_r * tem["omega"][i].imag)),
                dominant[i], repr(float(dominant_gamma[i])),
                repr(float(itg["eigenResidual"][i])),
                repr(float(tem["eigenResidual"][i])),
                repr(float(itg["fieldResidual"][i])),
                repr(float(tem["fieldResidual"][i])),
                repr(float(itg["modeOverlap"][i])),
                repr(float(tem["modeOverlap"][i])),
                repr(float(itg["runtimeSeconds"][i])),
                repr(float(tem["runtimeSeconds"][i])),
            ])

    peak_itg = int(np.nanargmax(itg["omega"].imag))
    peak_tem = int(np.nanargmax(tem["omega"].imag))
    print(f"ITG peak: ky*rho_s={ky_rho_s[peak_itg]:.2f}, "
          f"gamma*a/c_s={a_over_r * itg['omega'][peak_itg].imag:.6f}, "
          f"omega*a/c_s={a_over_r * itg['omega'][peak_itg].real:+.6f}")
    print(f"TEM peak: ky*rho_s={ky_rho_s[peak_tem]:.2f}, "
          f"gamma*a/c_s={a_over_r * tem['omega'][peak_tem].imag:.6f}, "
          f"omega*a/c_s={a_over_r * tem['omega'][peak_tem].real:+.6f}")
    tem_dominant_ky = ky_rho_s[dominant == "TEM"]
    if tem_dominant_ky.size:
        print("TEM-dominant sampled ky*rho_s:"
              + "".join(f" {value:.2f}" for value in tem_dominant_ky))
    print(f"wrote {csv_file}")


if __name__ == "__main__":
    main()
