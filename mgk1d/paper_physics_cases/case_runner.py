"""Python runners and regression checks for the seven paper physics cases."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgk
from mgk.examples.validation import published_cases


HERE = Path(__file__).resolve().parent
REFERENCE_FILE = HERE / "reference_results.json"
CASE_DIRECTORIES = {
    "01": "01_salpha_itg_eta",
    "02": "02_cbc_kinetic_itg_tem",
    "03": "03_strong_gradient_tem",
    "04": "04_miller_triangularity_itg",
    "05": "05_miller_tem_domain",
    "06": "06_xie_em_cbc",
    "07": "07_shen_em_kbm",
}

ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_UNIT = 1.66053906660e-27
ELECTRON_MASS = 9.1093837139e-31


def _solver(backend: str, guess: complex, *, subspace: int = 64) -> dict[str, Any]:
    use_gpu = backend == "gpu"
    return {
        "useGpu": use_gpu,
        "blockPrecision": "double",
        "eigenBackend": "gpu_arnoldi" if use_gpu else "eigs",
        "modeSelection": "nearest",
        "frequencyGuess": guess,
        "eigenTolerance": 2e-6 if use_gpu else 1e-9,
        "eigenSubspaceDimension": subspace,
        "eigenMaxIterations": 1200,
        "gpuArnoldiMaxRestarts": 30,
        "singleShiftTimeLimit": 180.0,
        "enableFactorizationCache": True,
        "enableGpuResultCache": False,
        "compactResult": True,
    }


def _xie_config(beta: float, backend: str) -> dict[str, Any]:
    major_radius = 0.835
    temperature = 2200.0
    ion_mass = 1837 * ELECTRON_MASS
    vti = math.sqrt(temperature * ELEMENTARY_CHARGE / ion_mass)
    rho_i = vti / (ELEMENTARY_CHARGE * 2.0 / ion_mass)
    frequency_reference = vti / major_radius
    return {
        "physical": {
            "magneticField": 2.0,
            "majorRadius": major_radius,
            "minorRadius": 0.18 * major_radius,
            "ionMass": ion_mass,
            "ionChargeNumber": 1,
            "ionTemperature": temperature,
            "electronTemperature": temperature,
            "electronBeta": beta,
            "densityGradientLength": major_radius / 2.2,
            "ionTemperatureGradientLength": major_radius / 6.9,
            "electronTemperatureGradientLength": major_radius / 6.9,
            "binormalWavenumber": 0.22 / rho_i,
        },
        "geometry": {
            "model": "s-alpha",
            "q": 1.4,
            "magneticShear": 0.78,
            "alpha": 0.0,
            "ballooningAngle": 0.0,
            "mirrorConvention": "cgyro_s_alpha",
        },
        "model": {
            "magneticMirror": True,
            "aparallel": beta > 0.0,
            "bparallel": False,
            "parallelBoundary": "open-extrapolated",
            "boundaryCoordinateStretch": 2.5,
            "boundarySpongeStrength": 0.3,
            "boundarySpongeFraction": 0.30,
            "electronClosure": "kinetic",
        },
        "species": {
            "enabled": True,
            "items": {
                "kind": "electron",
                "kinetic": True,
                "mass": ELECTRON_MASS,
                "temperature": temperature,
                "densityGradientLength": major_radius / 2.2,
                "temperatureGradientLength": major_radius / 6.9,
            },
        },
        "grid": {
            "thetaMin": -4 * math.pi,
            "thetaMax": 4 * math.pi,
            "numTheta": 97,
            "energyMax": 8.0,
            "numEnergy": 12,
            "numPitch": 24,
            "numBouncePoints": 48,
        },
        "solver": _solver(
            backend, (-2.0 + 0.95j) * frequency_reference, subspace=80
        ),
    }


def _shen_config(beta: float, fields: int, backend: str,
                 parallel_boundary: str = "periodic") -> dict[str, Any]:
    temperature = 1000.0
    ion_mass = 1837 * ELECTRON_MASS
    vti = math.sqrt(temperature * ELEMENTARY_CHARGE / ion_mass)
    rho_i = vti / (ELEMENTARY_CHARGE * 2.0 / ion_mass)
    seed = (-2.239 + 1.280j) if fields == 2 else (-1.960 + 1.924j)
    return {
        "physical": {
            "magneticField": 2.0,
            "majorRadius": 1.0,
            "minorRadius": 0.0018,
            "ionMass": ion_mass,
            "ionChargeNumber": 1,
            "ionTemperature": temperature,
            "electronTemperature": temperature,
            "electronBeta": beta,
            "densityGradientLength": 0.2,
            "ionTemperatureGradientLength": 0.2,
            "electronTemperatureGradientLength": 0.2,
            "binormalWavenumber": 0.3 / rho_i,
        },
        "geometry": {
            "model": "s-alpha",
            "q": 2.0,
            "magneticShear": 1.0,
            "alpha": 80.0 * beta,
            "ballooningAngle": 0.0,
            "mirrorConvention": "cgyro_s_alpha",
        },
        "model": {
            "magneticMirror": True,
            "aparallel": True,
            "bparallel": fields == 3,
            "parallelBoundary": parallel_boundary,
            "electronClosure": "kinetic",
        },
        "species": {
            "enabled": True,
            "items": {
                "kind": "electron",
                "kinetic": True,
                "mass": ELECTRON_MASS,
                "temperature": temperature,
                "densityGradientLength": 0.2,
                "temperatureGradientLength": 0.2,
            },
        },
        "grid": {
            "thetaMin": -5 * math.pi,
            "thetaMax": 5 * math.pi,
            "numTheta": 241 if backend == "gpu" else 129,
            "energyMax": 12.5,
            "numEnergy": 16,
            "numPitch": 24,
            "numBouncePoints": 24,
        },
        "solver": _shen_solver(backend, seed * vti),
    }


def _shen_solver(backend: str, guess: complex) -> dict[str, Any]:
    if backend == "gpu":
        # Exact matched profile from mgk_beta_range_scan.m.
        return {
            "useGpu": True,
            "blockPrecision": "single",
            "eigenBackend": "gpu_arnoldi",
            "modeSelection": "nearest",
            "singleShiftTimeLimit": 20.0,
            "eigenTolerance": 1e-6,
            "eigenSubspaceDimension": 8,
            "gpuArnoldiMaxRestarts": 40,
            "enableFactorizationCache": True,
            "enableGpuResultCache": False,
            "compactResult": True,
            "frequencyGuess": guess,
        }
    # CPU is retained as a portable diagnostic profile only.
    solver = _solver(backend, guess, subspace=24)
    solver["eigenMaxIterations"] = 6000
    return solver


def _load_references() -> dict[str, Any]:
    with REFERENCE_FILE.open(encoding="utf-8") as stream:
        return json.load(stream)


def _expected(case_id: str, point: str) -> tuple[complex | None, float]:
    item = _load_references().get(case_id, {}).get("points", {}).get(point)
    if item is None:
        return None, math.nan
    return complex(*item["omega"]), float(item["relativeTolerance"])


def _solve(
    case_id: str,
    point: str,
    config: dict[str, Any],
    *,
    scale: float = 1.0,
    previous: Any = None,
) -> tuple[dict[str, Any], Any]:
    config = copy.deepcopy(config)
    if previous is not None:
        config["solver"]["frequencyGuess"] = previous.omegaPhysical
        config["solver"]["eigenInitialVector"] = previous.reducedMode
    result = mgk.solve(config)
    omega = result.omega * scale
    gpu_info = result.get("gpuArnoldiInfo", {})
    inverse_residual = gpu_info.get("inverseResidualEstimate", math.nan)
    expected, tolerance = _expected(case_id, point)
    relative_error = (
        abs(omega - expected) / max(abs(expected), 1e-30)
        if expected is not None
        else math.nan
    )
    residual_ok = (
        math.isfinite(result.eigenResidual)
        and math.isfinite(result.fieldConstraintResidual)
        and result.eigenResidual <= 1e-6
        and result.fieldConstraintResidual <= 1e-6
    )
    if config["solver"].get("useGpu"):
        residual_ok = (
            math.isfinite(inverse_residual)
            and inverse_residual <= 5e-6
            and math.isfinite(result.fieldConstraintResidual)
            and result.fieldConstraintResidual <= 5e-4
        )
    reference_ok = expected is None or relative_error <= tolerance
    row = {
        "case": case_id,
        "point": point,
        "omegaReal": float(omega.real),
        "omegaImag": float(omega.imag),
        "expectedReal": None if expected is None else float(expected.real),
        "expectedImag": None if expected is None else float(expected.imag),
        "relativeError": None if expected is None else float(relative_error),
        "relativeTolerance": None if expected is None else tolerance,
        "eigenResidual": float(result.eigenResidual),
        "fieldResidual": float(result.fieldConstraintResidual),
        "inverseResidual": float(inverse_residual),
        "gpuConverged": bool(gpu_info.get("converged", False)),
        "runtimeSeconds": float(result.runtime),
        "passed": bool(residual_ok and reference_ok),
    }
    status = "PASS" if row["passed"] else "FAIL"
    error_text = "n/a" if expected is None else f"{relative_error:.3e}"
    print(
        f"[{status}] {case_id} {point}: "
        f"omega={omega.real:+.10f}{omega.imag:+.10f}i "
        f"relerr={error_text} eig={result.eigenResidual:.2e} "
        f"field={result.fieldConstraintResidual:.2e} t={result.runtime:.3f}s",
        flush=True,
    )
    return row, result


def _case_01(backend: str, full: bool) -> list[dict[str, Any]]:
    values = [2.3, 2.4, 2.5, 2.6, 2.7] if full else [2.5]
    rows = []
    anchor_row, anchor = _solve(
        "01", "eta=2.5", published_cases.salpha_adiabatic_itg(2.5, backend)
    )
    rows.append(anchor_row)
    for direction in (
        sorted(value for value in values if value > 2.5),
        sorted((value for value in values if value < 2.5), reverse=True),
    ):
        previous = anchor
        for eta in direction:
            row, previous = _solve(
                "01", f"eta={eta:.1f}",
                published_cases.salpha_adiabatic_itg(eta, backend),
                previous=previous,
            )
            rows.append(row)
    return sorted(rows, key=lambda row: row["point"])


def _case_02(backend: str, full: bool) -> list[dict[str, Any]]:
    if full:
        ky_values = [round(0.1 + 0.05 * index, 3) for index in range(9)]
        ky_values += [round(0.525 + 0.025 * index, 3) for index in range(20)]
    else:
        ky_values = [0.30]
    rows = []
    for branch in ("ITG", "TEM"):
        row, anchor = _solve(
            "02", f"ky={0.30:.3f}:{branch}",
            published_cases.cbc_kinetic_branch(0.30, branch, backend),
        )
        rows.append(row)
        for direction in (
            sorted(value for value in ky_values if value > 0.30),
            sorted((value for value in ky_values if value < 0.30), reverse=True),
        ):
            previous = anchor
            for ky in direction:
                row, previous = _solve(
                    "02", f"ky={ky:.3f}:{branch}",
                    published_cases.cbc_kinetic_branch(ky, branch, backend),
                    previous=previous,
                )
                rows.append(row)
    if not full:
        row, _ = _solve(
            "02", "ky=0.335:companion-TEM",
            published_cases.cbc_companion_tem(backend),
        )
        rows.append(row)
    return rows


def _case_03(backend: str, full: bool) -> list[dict[str, Any]]:
    row, _ = _solve(
        "03", "eta_e=3.13", published_cases.strong_gradient_tem(backend)
    )
    return [row]


def _case_04(backend: str, full: bool) -> list[dict[str, Any]]:
    values = [-0.4, -0.2, 0.0, 0.2, 0.4] if full else [0.2]
    anchor_delta = 0.0 if full else 0.2
    rows = []
    row, anchor = _solve(
        "04", f"delta={anchor_delta:+.1f}",
        published_cases.miller_triangularity_itg(anchor_delta, backend),
        scale=1.0 / 3.0,
    )
    rows.append(row)
    for direction in (
        sorted(value for value in values if value > anchor_delta),
        sorted((value for value in values if value < anchor_delta), reverse=True),
    ):
        previous = anchor
        for delta in direction:
            row, previous = _solve(
                "04", f"delta={delta:+.1f}",
                published_cases.miller_triangularity_itg(delta, backend),
                scale=1.0 / 3.0,
                previous=previous,
            )
            rows.append(row)
    return rows


def _case_05(backend: str, full: bool) -> list[dict[str, Any]]:
    row, _ = _solve(
        "05", "extended-domain-TEM",
        published_cases.miller_extended_tem(backend),
        scale=0.36,
    )
    return [row]


def _case_06(backend: str, full: bool) -> list[dict[str, Any]]:
    values = [value / 1000 for value in range(12, 21)] if full else [0.015]
    rows = []
    row, anchor = _solve("06", "beta=0.015:KBM", _xie_config(0.015, backend))
    rows.append(row)
    for direction in (
        sorted(value for value in values if value > 0.015),
        sorted((value for value in values if value < 0.015), reverse=True),
    ):
        previous = anchor
        for beta in direction:
            row, previous = _solve(
                "06", f"beta={beta:.3f}:KBM", _xie_config(beta, backend),
                previous=previous,
            )
            rows.append(row)
    return rows


def _case_07(backend: str, full: bool,
             parallel_boundary: str = "periodic") -> list[dict[str, Any]]:
    values = [value / 1000 for value in range(8, 26)] if full else [0.020]
    rows = []
    num_theta = 241 if backend == "gpu" else 129
    for fields in (2, 3):
        anchor_point = (
            f"beta=0.020:{fields}-field:ntheta={num_theta}:"
            f"boundary={parallel_boundary}"
        )
        row, anchor = _solve(
            "07", anchor_point,
            _shen_config(0.020, fields, backend, parallel_boundary)
        )
        rows.append(row)
        for direction in (
            sorted(value for value in values if value > 0.020),
            sorted((value for value in values if value < 0.020), reverse=True),
        ):
            previous = anchor
            for beta in direction:
                row, previous = _solve(
                    "07",
                    f"beta={beta:.3f}:{fields}-field:ntheta={num_theta}:"
                    f"boundary={parallel_boundary}",
                    _shen_config(beta, fields, backend, parallel_boundary),
                    previous=previous,
                )
                rows.append(row)
    return rows


RUNNERS: dict[str, Callable[[str, bool], list[dict[str, Any]]]] = {
    "01": _case_01,
    "02": _case_02,
    "03": _case_03,
    "04": _case_04,
    "05": _case_05,
    "06": _case_06,
    "07": _case_07,
}


def _write_results(case_id: str, rows: list[dict[str, Any]]) -> None:
    output_dir = HERE / CASE_DIRECTORIES[case_id]
    csv_file = output_dir / "python_results.csv"
    json_file = output_dir / "python_validation.json"
    with csv_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "case": case_id,
        "passed": all(row["passed"] for row in rows),
        "points": rows,
    }
    with json_file.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_case(case_id: str, backend: str = "cpu", full: bool = False,
             parallel_boundary: str = "periodic") -> bool:
    if case_id not in RUNNERS:
        raise ValueError(f"unknown paper case: {case_id}")
    if backend not in {"cpu", "gpu"}:
        raise ValueError("backend must be cpu or gpu")
    if parallel_boundary not in {"periodic", "open"}:
        raise ValueError("parallel_boundary must be periodic or open")
    if case_id != "07" and parallel_boundary != "periodic":
        raise ValueError("parallel_boundary is only configurable for case 07")
    rows = (_case_07(backend, full, parallel_boundary)
            if case_id == "07" else RUNNERS[case_id](backend, full))
    _write_results(case_id, rows)
    return all(row["passed"] for row in rows)
