#!/usr/bin/env python3
"""Convert a GENE ``vmec.dat``/geometry table to an MGK1D profile.

The conversion is intentionally explicit about the dimensional reduction.  A
GENE table contains the full local metric and magnetic-field derivatives,
whereas MGK1D accepts one scalar binormal metric and two coefficients in its
one-dimensional drift model.  For a ky-only mode (kx=0) this utility uses

``kperpMetric = sqrt(g_yy)``
``parallelGradient = q R_ref/L_ref C_xy / (J B)``
``driftCurvature = R_ref/L_ref K_y / B``
``driftParallelCorrection = 0``

where ``K_y`` is the curvature coefficient constructed with the same formula
as GENE's ``set_curvature`` routine.  GENE normalizes time to ``L_ref/c_ref``
whereas MGK normalizes it to ``R_ref/c_ref``.  The explicit length ratio is
therefore part of the operator conversion, not a plotting correction.  GENE's
velocity variables give ``mu*B + 2*vp^2 = v_perp^2/2 + v_parallel^2`` in
MGK's ``sqrt(T/m)`` convention, so no extra parallel-energy correction is
required.
This is a ky-only local reduction, not a direct VMEC reader and not a proof of
three-dimensional code equivalence.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


_HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^!/&]+)")


def _periodic_derivative(values: np.ndarray, spacing: float) -> np.ndarray:
    return (
        -np.roll(values, -2) + 8 * np.roll(values, -1)
        - 8 * np.roll(values, 1) + np.roll(values, 2)
    ) / (12 * spacing)


def _read_table(path: Path) -> tuple[dict[str, float], np.ndarray]:
    header: dict[str, float] = {}
    rows: list[list[float]] = []
    in_header = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("&"):
            in_header = True
            continue
        if in_header and line == "/":
            in_header = False
            continue
        if in_header:
            match = _HEADER_RE.match(raw)
            if match:
                token = match.group(2).strip().strip("'\"")
                try:
                    header[match.group(1)] = float(token.replace("D", "E"))
                except ValueError:
                    pass
            continue
        if line.startswith("#"):
            continue
        values = line.split()
        if len(values) >= 16:
            try:
                rows.append([float(value.replace("D", "E")) for value in values[:16]])
            except ValueError:
                continue
    table = np.asarray(rows, dtype=float)
    if table.ndim != 2 or table.shape[1] != 16 or table.shape[0] < 8:
        raise ValueError(f"{path} does not contain a 16-column GENE geometry table")
    return header, table


def convert_gene_geometry(
    source: str | Path,
    output: str | Path,
    *,
    period: float | None = None,
    z_origin: float | None = None,
    normalize_b: bool = False,
    dpdx_pm: float = 0.0,
    reference_length_ratio: float | None = None,
    periodic: bool = True,
) -> Path:
    """Convert ``source`` and write an MGK1D ``.npz`` profile."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not np.isfinite(dpdx_pm):
        raise ValueError("dpdx_pm must be finite")
    header, table = _read_table(source_path)
    if period is None:
        # GENE's table omits z; its geometry span is encoded by n_pol.  Keep
        # the historical 2*pi default for generic tables without that header.
        n_pol = float(header.get("n_pol", 1.0))
        period = 2 * np.pi * abs(n_pol)
    if period <= 0 or not np.isfinite(period):
        raise ValueError("period must be positive and finite")
    if z_origin is None:
        z_origin = -0.5 * period
    if not np.isfinite(z_origin):
        raise ValueError("z_origin must be finite")
    gridpoints = int(header.get("gridpoints", table.shape[0]))
    if gridpoints != table.shape[0]:
        raise ValueError(
            f"GENE header declares {gridpoints} grid points but table has {table.shape[0]} rows"
        )

    # GENE columns: gxx,gxy,gxz,gyy,gyz,gzz,B,dBdx,dBdy,dBdz,J,R,phi,Z,dx/dR,dx/dZ.
    gxx, gxy, gxz, gyy, gyz, _gzz = table[:, 0:6].T
    field, d_bx, _d_by, d_bz, jacobian = table[:, 6:11].T
    if np.any(field <= 0) or np.any(jacobian <= 0):
        raise ValueError("GENE geometry requires positive B and J for this conversion")
    if np.any(gyy <= 0):
        raise ValueError("g_yy must be positive for the ky-only MGK reduction")

    c_xy = float(header.get("Cxy", 1.0))
    if not np.isfinite(c_xy) or c_xy == 0:
        raise ValueError("GENE header Cxy must be finite and nonzero")
    q = float(header.get("q0", np.nan))
    if not np.isfinite(q) or q == 0:
        raise ValueError("GENE header q0 must be finite and nonzero")
    if reference_length_ratio is None:
        reference_length_ratio = float(header.get("major_R", np.nan))
    if not np.isfinite(reference_length_ratio) or reference_length_ratio <= 0:
        raise ValueError(
            "reference_length_ratio must be positive and finite (R_ref/L_ref)"
        )

    ga1 = gxx * gyy - gxy * gxy
    ga3 = gxy * gyz - gyy * gxz
    if np.any(np.abs(ga1) <= np.finfo(float).eps):
        raise ValueError("GENE metric has a singular perpendicular 2x2 block")
    if normalize_b:
        field_scale = float(np.mean(field))
        field = field / field_scale
        d_bx = d_bx / field_scale
        d_bz = d_bz / field_scale
    # This is K_j after GENE's metric projection, before any species factor.
    k_y = (d_bx - ga3 / ga1 * d_bz) / c_xy

    # In MGK velocity units the gradB_eq_curv pressure contribution has the
    # same total-energy dependence as the projected GENE curvature term.
    pressure_drift = dpdx_pm / (2.0 * c_xy * field**2)
    kperp_metric = np.sqrt(gyy)
    drift_curvature = reference_length_ratio * (k_y / field + pressure_drift)
    gene_ky = k_y.copy()
    drift_parallel = np.zeros_like(drift_curvature)
    parallel_gradient = q * reference_length_ratio * c_xy / (jacobian * field)
    n = table.shape[0]
    dz = period / n
    z = z_origin + dz * np.arange(n, dtype=float)
    if periodic:
        profile_period = period
        log_field_derivative = _periodic_derivative(np.log(field), dz)
        kperp_metric_derivative = _periodic_derivative(kperp_metric, dz)
    else:
        # GENE geometry tables use cell-centered samples and omit the right
        # endpoint.  MGK's finite open interval needs both endpoints; append
        # a one-sided linear extrapolation instead of silently shortening the
        # domain by one grid cell.
        z = np.r_[z, z[-1] + dz]
        def append_endpoint(value):
            return np.r_[value, 2.0 * value[-1] - value[-2]]
        field = append_endpoint(field)
        kperp_metric = append_endpoint(kperp_metric)
        drift_curvature = append_endpoint(drift_curvature)
        gene_ky = append_endpoint(gene_ky)
        drift_parallel = append_endpoint(drift_parallel)
        parallel_gradient = append_endpoint(parallel_gradient)
        log_field_derivative = append_endpoint(np.gradient(np.log(field[:-1]), dz, edge_order=2))
        kperp_metric_derivative = append_endpoint(np.gradient(kperp_metric[:-1], dz, edge_order=2))
        profile_period = float(z[-1] - z[0])
    profile = {
        "z": z,
        "period": np.asarray(profile_period),
        "periodic": np.asarray(periodic),
        "B": field,
        "kperpMetric": kperp_metric,
        "driftCurvature": drift_curvature,
        "driftParallelCorrection": drift_parallel,
        "parallelGradient": parallel_gradient,
        "geneKyCurvatureCoefficient": gene_ky,
        "geneCxy": np.asarray(c_xy),
        "geneQ": np.asarray(q),
        "referenceLengthRatio": np.asarray(reference_length_ratio),
        "fieldLineLabel": np.asarray(0.0),
        "sourceFormat": np.asarray("GENE 16-column geometry table"),
        "sourceFile": np.asarray(str(source_path)),
        "driftConvention": np.asarray("GENE set_curvature, ky-only reduced"),
        "dpdxPm": np.asarray(dpdx_pm),
        "normalizationNote": np.asarray(
            "GENE Lref/c_ref operators converted to MGK Rref/c_ref convention"
        ),
    }
    # The MGK loader computes these when absent; saving them makes the profile
    # deterministic and avoids a second finite-difference convention later.
    profile["logFieldDerivative"] = log_field_derivative
    profile["kperpMetricDerivative"] = kperp_metric_derivative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **profile)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="GENE vmec.dat or geometry table")
    parser.add_argument("output", type=Path, help="MGK1D profile .npz")
    parser.add_argument(
        "--period", type=float, default=None,
        help="field-line span; defaults to 2*pi*abs(n_pol) for GENE tables",
    )
    parser.add_argument(
        "--z-origin", type=float, default=None,
        help="first coordinate; defaults to -period/2",
    )
    parser.add_argument(
        "--normalize-b", action="store_true",
        help="divide B by its table mean (off by default; GENE B is normally already normalized)",
    )
    parser.add_argument(
        "--dpdx-pm", type=float, default=0.0,
        help="GENE pressure-gradient coefficient for gradB_eq_curv (default: 0)",
    )
    parser.add_argument(
        "--reference-length-ratio", type=float, default=None,
        help="MGK R_ref divided by GENE L_ref (default: GENE header major_R)",
    )
    parser.add_argument(
        "--open-boundary", action="store_true",
        help="write a non-periodic field-line profile and use one-sided endpoint derivatives",
    )
    args = parser.parse_args()
    path = convert_gene_geometry(
        args.source, args.output, period=args.period,
        z_origin=args.z_origin, normalize_b=args.normalize_b,
        dpdx_pm=args.dpdx_pm,
        reference_length_ratio=args.reference_length_ratio,
        periodic=not args.open_boundary,
    )
    print(f"wrote MGK1D profile: {path}")


if __name__ == "__main__":
    main()
