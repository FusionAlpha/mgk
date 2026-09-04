"""Reproducible MGK1D configurations for arXiv:2608.17418.

The paper reports dimensionless local inputs.  This example chooses a
convenient SI realization (R0=1 m, B0=2 T, Ti=Te=1 keV, deuterium ions); only
the dimensionless combinations affect the normalized electrostatic results.

The paper did not publish every velocity-grid order used for every physics
figure.  The grids below are therefore the documented MGK1D 0.1.2 product
reproduction grids, not an assertion about unpublished author input files.
See PUBLISHED_VALIDATION_CASES.md for the evidence and comparison rules.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import mgk


ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_UNIT = 1.66053906660e-27
ELECTRON_MASS = 9.1093837139e-31
ION_MASS = 2 * ATOMIC_MASS_UNIT
TEMPERATURE_EV = 1000.0
MAGNETIC_FIELD_T = 2.0
MAJOR_RADIUS_M = 1.0


def _scales() -> tuple[float, float]:
    vti = math.sqrt(TEMPERATURE_EV * ELEMENTARY_CHARGE / ION_MASS)
    omega_ci = ELEMENTARY_CHARGE * MAGNETIC_FIELD_T / ION_MASS
    return vti / omega_ci, vti / MAJOR_RADIUS_M


def _solver(backend: str, guess: complex) -> dict[str, Any]:
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    _, frequency_reference = _scales()
    use_gpu = backend == "gpu"
    return {
        "useGpu": use_gpu,
        "blockPrecision": "double",
        "eigenBackend": "gpu_arnoldi" if use_gpu else "eigs",
        "modeSelection": "nearest",
        "frequencyGuess": guess * frequency_reference,
        "eigenTolerance": 1e-9 if not use_gpu else 2e-6,
        "eigenSubspaceDimension": 64,
        "eigenMaxIterations": 1200,
        "gpuArnoldiMaxRestarts": 20,
        "singleShiftTimeLimit": 120.0,
        "enableFactorizationCache": False,
        "enableGpuResultCache": False,
        "compactResult": True,
    }


def _physical(*, minor_radius: float, density_gradient: float,
              ion_temperature_gradient: float,
              electron_temperature_gradient: float,
              ky_rho_i: float) -> dict[str, Any]:
    rho_i, _ = _scales()
    return {
        "magneticField": MAGNETIC_FIELD_T,
        "majorRadius": MAJOR_RADIUS_M,
        "minorRadius": minor_radius,
        "ionMass": ION_MASS,
        "ionChargeNumber": 1,
        "ionTemperature": TEMPERATURE_EV,
        "electronTemperature": TEMPERATURE_EV,
        "electronBeta": 0.0,
        "densityGradientLength": density_gradient,
        "ionTemperatureGradientLength": ion_temperature_gradient,
        "electronTemperatureGradientLength": electron_temperature_gradient,
        "binormalWavenumber": ky_rho_i / rho_i,
    }


def _kinetic_electron(density_gradient: float,
                      temperature_gradient: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "items": {
            "kind": "electron",
            "kinetic": True,
            "mass": ELECTRON_MASS,
            "temperature": TEMPERATURE_EV,
            "densityGradientLength": density_gradient,
            "temperatureGradientLength": temperature_gradient,
        },
    }


def salpha_adiabatic_itg(eta_i: float = 2.5,
                         backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 1: adiabatic-electron s-alpha ITG eta_i scan."""
    ln = 0.25 * MAJOR_RADIUS_M
    return {
        "physical": _physical(
            minor_radius=0.05 * MAJOR_RADIUS_M,
            density_gradient=ln,
            ion_temperature_gradient=ln / eta_i,
            electron_temperature_gradient=ln,
            ky_rho_i=0.45 / math.sqrt(2.0),
        ),
        "geometry": {
            "model": "s-alpha", "q": 1.0, "magneticShear": 1.0,
            "alpha": 0.0, "ballooningAngle": 0.0,
        },
        "model": {
            "electronClosure": "adiabatic", "parallelBoundary": "open",
            "mirrorConvention": "cgyro_s_alpha",
        },
        "grid": {
            "thetaMin": -5 * math.pi, "thetaMax": 5 * math.pi,
            "numTheta": 97, "energyMax": 12.5, "numEnergy": 16,
            "numPitch": 24, "numBouncePoints": 32,
        },
        "solver": _solver(backend, -0.80 + 0.35j),
    }


def cbc_kinetic_branch(ky_rho_s: float = 0.30, branch: str = "ITG",
                       backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 2(a,b): Rewoldt CBC ITG or TEM branch."""
    branch = branch.upper()
    if branch not in {"ITG", "TEM"}:
        raise ValueError("branch must be ITG or TEM")
    ln = MAJOR_RADIUS_M / 2.22
    lti = MAJOR_RADIUS_M / 6.92
    config = {
        "physical": _physical(
            minor_radius=0.18 * MAJOR_RADIUS_M,
            density_gradient=ln,
            ion_temperature_gradient=lti,
            electron_temperature_gradient=lti,
            ky_rho_i=ky_rho_s,
        ),
        "geometry": {
            "model": "s-alpha", "q": 1.4, "magneticShear": 0.776,
            "alpha": 0.0, "ballooningAngle": 0.0,
        },
        "model": {
            "electronClosure": "kinetic", "parallelBoundary": "open",
            "mirrorConvention": "cgyro_s_alpha",
        },
        "species": _kinetic_electron(ln, lti),
        "grid": {
            "thetaMin": -4 * math.pi, "thetaMax": 4 * math.pi,
            "numTheta": 97, "energyMax": 12.5, "numEnergy": 16,
            "numPitch": 32, "numBouncePoints": 48,
        },
        "solver": _solver(
            backend, (-0.89 + 0.48j) if branch == "ITG" else (0.49 + 0.19j)
        ),
    }
    return config


def cbc_companion_tem(backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 2(c): companion eta_i=1 TEM at ky*rho_i=0.335."""
    config = cbc_kinetic_branch(0.335, "TEM", backend)
    config["physical"]["ionTemperatureGradientLength"] = (
        MAJOR_RADIUS_M / 2.22
    )
    config["solver"] = _solver(backend, 0.66 + 0.44j)
    return config


def strong_gradient_tem(backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 3: strong electron-gradient kinetic mode."""
    ln = 0.018 * MAJOR_RADIUS_M
    lte = ln / 3.13
    return {
        "physical": _physical(
            minor_radius=0.18 * MAJOR_RADIUS_M,
            density_gradient=ln,
            ion_temperature_gradient=1e6 * MAJOR_RADIUS_M,
            electron_temperature_gradient=lte,
            ky_rho_i=0.7,
        ),
        "geometry": {
            "model": "s-alpha", "q": 1.4, "magneticShear": 0.776,
            "alpha": 0.0, "ballooningAngle": 0.0,
        },
        "model": {
            "electronClosure": "kinetic", "parallelBoundary": "open",
            "mirrorConvention": "cgyro_s_alpha",
        },
        "species": _kinetic_electron(ln, lte),
        "grid": {
            "thetaMin": -8 * math.pi, "thetaMax": 8 * math.pi,
            "numTheta": 193, "energyMax": 12.5, "numEnergy": 16,
            "numPitch": 32, "numBouncePoints": 48,
        },
        "solver": _solver(backend, 23.3 + 13.8j),
    }


def miller_triangularity_itg(delta: float = 0.0,
                             backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 4: adiabatic-electron Miller triangularity scan."""
    a = MAJOR_RADIUS_M / 3.0
    local_r = 0.5 * a
    return {
        "physical": _physical(
            minor_radius=local_r,
            density_gradient=a,
            ion_temperature_gradient=a / 3.0,
            electron_temperature_gradient=a / 3.0,
            ky_rho_i=0.3,
        ),
        "geometry": {
            "model": "miller", "q": 2.0, "magneticShear": 1.0,
            "alpha": 0.0, "ballooningAngle": 0.0,
            "shiftDerivative": 0.0, "elongation": 1.0,
            "elongationShear": 0.0, "triangularity": delta,
            "triangularityShear": 0.0, "betaStar": 0.0,
        },
        "model": {
            "electronClosure": "adiabatic", "parallelBoundary": "open",
            "mirrorConvention": "physical",
        },
        "grid": {
            "thetaMin": -4 * math.pi, "thetaMax": 4 * math.pi,
            "numTheta": 97, "energyMax": 12.5, "numEnergy": 16,
            "numPitch": 32, "numBouncePoints": 48,
        },
        "solver": _solver(backend, -0.67 + 0.52j),
    }


def miller_extended_tem(backend: str = "cpu") -> dict[str, Any]:
    """Paper Fig. 5: circular-Miller, extended-parallel-domain TEM."""
    config = cbc_companion_tem(backend)
    config["geometry"].update(
        model="miller", shiftDerivative=0.0, elongation=1.0,
        elongationShear=0.0, triangularity=0.0,
        triangularityShear=0.0, betaStar=0.0,
    )
    config["model"]["mirrorConvention"] = "physical"
    config["grid"].update(
        thetaMin=-5 * math.pi, thetaMax=5 * math.pi, numTheta=121
    )
    config["solver"] = _solver(backend, 0.52 + 0.58j)
    return config


BUILDERS = {
    "salpha-adiabatic-itg": lambda backend="cpu": salpha_adiabatic_itg(2.5, backend),
    "cbc-itg": lambda backend="cpu": cbc_kinetic_branch(0.30, "ITG", backend),
    "cbc-tem": lambda backend="cpu": cbc_kinetic_branch(0.30, "TEM", backend),
    "cbc-companion-tem": cbc_companion_tem,
    "strong-gradient-tem": strong_gradient_tem,
    "miller-triangularity-itg": (
        lambda backend="cpu": miller_triangularity_itg(0.2, backend)
    ),
    "miller-extended-tem": miller_extended_tem,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(BUILDERS))
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--json", action="store_true", help="print configuration only")
    args = parser.parse_args()
    config = copy.deepcopy(BUILDERS[args.case](args.backend))
    if args.json:
        print(json.dumps(config, indent=2, default=str))
        return
    result = mgk.solve(config)
    print(f"case={args.case} backend={args.backend}")
    print(f"omega R0/vti = {result.omega.real:+.12g}{result.omega.imag:+.12g}i")
    print(f"eigen residual = {result.eigenResidual:.6e}")
    print(f"field residual = {result.fieldConstraintResidual:.6e}")
    print(f"wall/runtime seconds = {result.runtime:.6f}")


if __name__ == "__main__":
    main()
