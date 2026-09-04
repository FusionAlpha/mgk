"""Minimal direct VMEC ``wout.nc`` reader and local field-line sampler.

The sampler intentionally produces the geometry-only profile consumed by the
orbit operator.  It does not build a global VMEC mesh or depend on GENE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VMECWout:
    """Fourier data and scalar metadata read from a VMEC output file."""

    filename: str
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnmax_nyq: int
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    rmnc: np.ndarray
    rmns: np.ndarray
    zmnc: np.ndarray
    zmns: np.ndarray
    lmnc: np.ndarray
    lmns: np.ndarray
    bmnc: np.ndarray
    bmns: np.ndarray
    bsupumnc: np.ndarray
    bsupumns: np.ndarray
    bsupvmnc: np.ndarray
    bsupvmns: np.ndarray
    iotaf: np.ndarray
    b0: float
    aminor: float
    rmajor: float

    def surface(self, s: float) -> dict[str, np.ndarray | float]:
        """Interpolate Fourier coefficients to normalized toroidal flux ``s``."""
        if not np.isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError("VMEC surface s must be finite and in [0, 1]")
        coordinate = s * (self.ns - 1)
        lower = min(int(np.floor(coordinate)), self.ns - 1)
        upper = min(lower + 1, self.ns - 1)
        weight = coordinate - lower

        def interpolate(values: np.ndarray) -> np.ndarray:
            return (1.0 - weight) * values[lower] + weight * values[upper]

        return {
            "rmnc": interpolate(self.rmnc), "rmns": interpolate(self.rmns),
            "zmnc": interpolate(self.zmnc), "zmns": interpolate(self.zmns),
            "lmnc": interpolate(self.lmnc), "lmns": interpolate(self.lmns),
            "bmnc": interpolate(self.bmnc), "bmns": interpolate(self.bmns),
            "bsupumnc": interpolate(self.bsupumnc),
            "bsupumns": interpolate(self.bsupumns),
            "bsupvmnc": interpolate(self.bsupvmnc),
            "bsupvmns": interpolate(self.bsupvmns),
            "iota": float((1.0 - weight) * self.iotaf[lower] + weight * self.iotaf[upper]),
        }


def _variable(variables: dict[str, Any], name: str, required: bool = True) -> np.ndarray:
    if name not in variables:
        if required:
            raise ValueError(f"VMEC file is missing variable '{name}'")
        return np.asarray([], dtype=float)
    variable = variables[name]
    value = variable.data if hasattr(variable, "data") else variable[...]
    return np.asarray(value).copy()


def _scalar(variables: dict[str, Any], name: str, default: float | None = None) -> float:
    value = _variable(variables, name, required=default is None).reshape(-1)
    return float(value[0]) if value.size else float(default)


def _surface_array(value: np.ndarray, ns: int, name: str) -> np.ndarray:
    """Normalize VMEC arrays to shape ``(ns, modes)``."""
    value = np.asarray(value, dtype=float)
    if value.ndim != 2:
        raise ValueError(f"VMEC variable '{name}' must be two-dimensional")
    if value.shape[0] == ns:
        return value
    if value.shape[1] == ns:
        return value.T
    raise ValueError(f"VMEC variable '{name}' has no surface dimension of length {ns}")


def _optional_surface_array(variables, name: str, ns: int, modes: int) -> np.ndarray:
    value = _variable(variables, name, required=False)
    if value.size == 0:
        return np.zeros((ns, modes), dtype=float)
    result = _surface_array(value, ns, name)
    if result.shape[1] != modes:
        raise ValueError(f"VMEC variable '{name}' has incompatible mode count")
    return result


def read_vmec_wout(filename: str | Path) -> VMECWout:
    """Read a VMEC NetCDF output without requiring GENE.

    ``netCDF4`` is used when available.  The required package dependencies
    already include SciPy, whose NetCDF3 reader handles standard VMEC output as
    a fallback, so the public MGK CPU installation does not require netCDF4.
    """
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"VMEC output does not exist: {path}")

    try:
        from netCDF4 import Dataset  # type: ignore
    except ImportError:
        Dataset = None

    if Dataset is not None:
        with Dataset(path, "r") as dataset:
            variables = dataset.variables
            return _read_vmec_variables(path, variables)

    from scipy.io import netcdf_file
    with netcdf_file(path, "r", mmap=False) as dataset:
        return _read_vmec_variables(path, dataset.variables)


def _read_vmec_variables(path: Path, variables: dict[str, Any]) -> VMECWout:
    ns = int(_scalar(variables, "ns"))
    nfp = int(_scalar(variables, "nfp"))
    mpol = int(_scalar(variables, "mpol"))
    ntor = int(_scalar(variables, "ntor"))
    mnmax = int(_scalar(variables, "mnmax"))
    mnmax_nyq = int(_scalar(variables, "mnmax_nyq"))

    def modes(name: str, count: int) -> np.ndarray:
        value = _variable(variables, name).reshape(-1).astype(float)
        if value.size != count:
            raise ValueError(f"VMEC mode array '{name}' has length {value.size}, expected {count}")
        return value

    def coeff(name: str, count: int) -> np.ndarray:
        value = _surface_array(_variable(variables, name), ns, name)
        if value.shape[1] != count:
            raise ValueError(
                f"VMEC coefficient array '{name}' has {value.shape[1]} modes, expected {count}"
            )
        return value

    result = VMECWout(
        filename=str(path), nfp=nfp, ns=ns, mpol=mpol, ntor=ntor,
        mnmax=mnmax, mnmax_nyq=mnmax_nyq,
        xm=modes("xm", mnmax), xn=modes("xn", mnmax),
        xm_nyq=modes("xm_nyq", mnmax_nyq), xn_nyq=modes("xn_nyq", mnmax_nyq),
        rmnc=coeff("rmnc", mnmax), rmns=_optional_surface_array(variables, "rmns", ns, mnmax),
        zmnc=_optional_surface_array(variables, "zmnc", ns, mnmax),
        zmns=coeff("zmns", mnmax),
        lmnc=_optional_surface_array(variables, "lmnc", ns, mnmax),
        lmns=coeff("lmns", mnmax),
        bmnc=coeff("bmnc", mnmax_nyq),
        bmns=_optional_surface_array(variables, "bmns", ns, mnmax_nyq),
        bsupumnc=coeff("bsupumnc", mnmax_nyq),
        bsupumns=_optional_surface_array(variables, "bsupumns", ns, mnmax_nyq),
        bsupvmnc=coeff("bsupvmnc", mnmax_nyq),
        bsupvmns=_optional_surface_array(variables, "bsupvmns", ns, mnmax_nyq),
        iotaf=_variable(variables, "iotaf").reshape(-1).astype(float),
        b0=abs(_scalar(variables, "b0", 1.0)),
        aminor=abs(_scalar(variables, "Aminor_p", 1.0)),
        rmajor=abs(_scalar(variables, "Rmajor_p", 1.0)),
    )
    if result.iotaf.size != ns:
        raise ValueError("VMEC iotaf has an incompatible surface count")
    if result.b0 <= 0 or result.aminor <= 0 or result.rmajor <= 0:
        raise ValueError("VMEC reference B, minor radius, and major radius must be positive")
    return result


def _fourier(coeff_cos, coeff_sin, xm, xn, theta, phi):
    phase = np.asarray(theta)[..., None] * xm - np.asarray(phi)[..., None] * xn
    cos_phase, sin_phase = np.cos(phase), np.sin(phase)
    value = np.sum(coeff_cos * cos_phase + coeff_sin * sin_phase, axis=-1)
    dtheta = np.sum(-xm * coeff_cos * sin_phase + xm * coeff_sin * cos_phase, axis=-1)
    dphi = np.sum(xn * coeff_cos * sin_phase - xn * coeff_sin * cos_phase, axis=-1)
    return value, dtheta, dphi


def _position(coefficients, theta, phi, eq: VMECWout):
    r, rt, rp = _fourier(coefficients["rmnc"], coefficients["rmns"], eq.xm, eq.xn, theta, phi)
    z, zt, zp = _fourier(coefficients["zmnc"], coefficients["zmns"], eq.xm, eq.xn, theta, phi)
    return r, rt, rp, z, zt, zp


def _cylindrical_vectors(r, rt, rp, z, zt, zp, phi):
    e_r = np.stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)), axis=-1)
    e_phi = np.stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)), axis=-1)
    e_z = np.zeros_like(e_r)
    e_z[..., 2] = 1.0
    e_theta = rt[..., None] * e_r + zt[..., None] * e_z
    e_toroidal = rp[..., None] * e_r + r[..., None] * e_phi + zp[..., None] * e_z
    return e_theta, e_toroidal


def _finite_derivative(values, spacing):
    values = np.asarray(values, dtype=float)
    return np.gradient(values, spacing, axis=0, edge_order=2)


def sample_vmec_fieldline(
    filename: str | Path,
    *,
    s: float = 0.64,
    alpha: float = 0.0,
    z_min: float = -4 * np.pi,
    z_max: float = 4 * np.pi,
    num_points: int = 513,
) -> dict[str, np.ndarray | float | str | bool]:
    """Sample a VMEC surface along a straight-field-line coordinate.

    ``z`` is a local field-line coordinate, while the VMEC toroidal angle is
    ``phi=z/nfp``.  The output is deliberately non-periodic by default because
    a generic stellarator field line does not close after one toroidal period.
    Use ``parallelBoundary='open'`` for this profile in MGK.
    """
    if not isinstance(num_points, (int, np.integer)) or num_points < 17:
        raise ValueError("num_points must be an integer >= 17")
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        raise ValueError("z_min must be smaller than z_max")
    eq = read_vmec_wout(filename)
    coefficients = eq.surface(float(s))
    iota = float(coefficients["iota"])
    if abs(iota) < 1e-12:
        raise ValueError("VMEC iota is too close to zero for a field-line sample")
    z = np.linspace(z_min, z_max, int(num_points))
    phi = z / eq.nfp
    target = float(alpha) + iota * phi
    theta = target.copy()
    for _ in range(30):
        lam, lam_theta, _ = _fourier(
            coefficients["lmnc"], coefficients["lmns"], eq.xm, eq.xn, theta, phi
        )
        correction = (theta + lam - target) / (1.0 + lam_theta)
        theta -= correction
        if np.max(np.abs(correction)) < 1e-12:
            break
    else:
        raise RuntimeError("VMEC straight-field-line angle solve did not converge")

    r, r_theta, r_phi, zz, z_theta, z_phi = _position(coefficients, theta, phi, eq)
    e_theta, e_phi = _cylindrical_vectors(r, r_theta, r_phi, zz, z_theta, z_phi, phi)
    gtt = np.sum(e_theta * e_theta, axis=-1)
    gtp = np.sum(e_theta * e_phi, axis=-1)
    gpp = np.sum(e_phi * e_phi, axis=-1)
    determinant = gtt * gpp - gtp * gtp
    if np.any(determinant <= 0):
        raise ValueError("VMEC surface metric is singular along the requested field line")

    lam, lam_theta, lam_phi = _fourier(
        coefficients["lmnc"], coefficients["lmns"], eq.xm, eq.xn, theta, phi
    )
    alpha_theta = 1.0 + lam_theta
    alpha_phi = lam_phi - iota
    grad_alpha_sq = (
        gpp * alpha_theta**2 - 2 * gtp * alpha_theta * alpha_phi
        + gtt * alpha_phi**2
    ) / determinant
    length_ref = eq.aminor
    kmetric = length_ref * np.sqrt(np.maximum(grad_alpha_sq, 0.0))

    b_raw, b_theta, b_phi = _fourier(
        coefficients["bmnc"], coefficients["bmns"], eq.xm_nyq, eq.xn_nyq, theta, phi
    )
    btheta, _, _ = _fourier(
        coefficients["bsupumnc"], coefficients["bsupumns"], eq.xm_nyq, eq.xn_nyq, theta, phi
    )
    bphi, _, _ = _fourier(
        coefficients["bsupvmnc"], coefficients["bsupvmns"], eq.xm_nyq, eq.xn_nyq, theta, phi
    )
    bvec = btheta[..., None] * e_theta + bphi[..., None] * e_phi
    bnorm = np.linalg.norm(bvec, axis=-1)
    bhat = bvec / np.maximum(bnorm[..., None], np.finfo(float).eps)

    theta_z = _finite_derivative(theta, z[1] - z[0])
    phi_z = np.full_like(z, 1.0 / eq.nfp)
    tangent = theta_z[..., None] * e_theta + phi_z[..., None] * e_phi
    dl_dz = np.linalg.norm(tangent, axis=-1)
    q = 1.0 / iota
    parallel_gradient = q / (dl_dz * length_ref)

    # Convert surface covariant derivatives into physical tangent gradients.
    inv_metric_00 = gpp / determinant
    inv_metric_01 = -gtp / determinant
    inv_metric_11 = gtt / determinant
    grad_b = (
        (inv_metric_00 * b_theta + inv_metric_01 * b_phi)[..., None] * e_theta
        + (inv_metric_01 * b_theta + inv_metric_11 * b_phi)[..., None] * e_phi
    )
    grad_alpha = (
        (inv_metric_00 * alpha_theta + inv_metric_01 * alpha_phi)[..., None] * e_theta
        + (inv_metric_01 * alpha_theta + inv_metric_11 * alpha_phi)[..., None] * e_phi
    )
    curvature = _finite_derivative(bhat, z[1] - z[0]) / np.maximum(dl_dz[..., None], 1e-14)
    drift = length_ref**2 * np.sum(
        np.cross(bhat, grad_b / np.maximum(np.abs(b_raw)[..., None], 1e-14) + curvature)
        * grad_alpha,
        axis=-1,
    )

    field = b_raw / eq.b0
    if np.any(field <= 0) or np.any(~np.isfinite(field)):
        raise ValueError("VMEC magnetic-field magnitude is non-positive or non-finite")
    profile: dict[str, np.ndarray | float | str | bool] = {
        "z": z, "period": float(z_max - z_min), "periodic": False,
        "B": field, "kperpMetric": np.maximum(kmetric, 1e-12),
        "driftCurvature": drift, "driftParallelCorrection": drift,
        "parallelGradient": parallel_gradient,
        "logFieldDerivative": _finite_derivative(np.log(field), z[1] - z[0]),
        "kperpMetricDerivative": _finite_derivative(kmetric, z[1] - z[0]),
        "fieldLineLabel": float(alpha), "vmecFile": str(Path(filename).resolve()),
        "vmecSurfaceS": float(s), "vmecIota": iota, "vmecQ": q,
        "vmecNfp": eq.nfp, "referenceB": eq.b0, "lengthReference": length_ref,
        "geometrySource": "direct VMEC Fourier field-line sample",
    }
    return profile


def write_vmec_profile(filename: str | Path, output: str | Path, **kwargs) -> Path:
    """Sample ``filename`` directly and write an MGK-compatible ``.npz``."""
    profile = sample_vmec_fieldline(filename, **kwargs)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **profile)
    return destination
