"""s-alpha and Miller geometry and mirror-field conventions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from mgk._struct import Struct


_MILLER_TABLE_CACHE = {}
_STELLARATOR_PROFILE_CACHE = {}


def _periodic_derivative(values, spacing):
    """Fourth-order derivative for a uniformly sampled periodic profile."""
    values = np.asarray(values, dtype=float)
    return (
        -np.roll(values, -2) + 8 * np.roll(values, -1)
        - 8 * np.roll(values, 1) + np.roll(values, 2)
    ) / (12 * spacing)


def _profile_array(data, name, required=True):
    if name not in data:
        if required:
            raise ValueError(f"stellarator profile is missing '{name}'")
        return None
    value = np.asarray(data[name], dtype=float).reshape(-1)
    if value.size < 8 or not np.all(np.isfinite(value)):
        raise ValueError(f"stellarator profile '{name}' must be a finite 1D array with >=8 points")
    return value


def load_stellarator_profile(filename):
    """Load a preprocessed local stellarator field-line profile.

    The profile is deliberately geometry-only.  ``kperpMetric`` is the
    coefficient multiplying the normalized binormal wavenumber, so that
    ``k_perp**2 = (kt*kperpMetric)**2``.  The remaining arrays are the
    already-normalized coefficients consumed by the orbit operator.
    """
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"stellarator geometry profile does not exist: {path}")
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _STELLARATOR_PROFILE_CACHE.get(key)
    if cached is not None:
        return cached
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "stellarator geometry currently accepts a preprocessed .npz profile; "
            "use mgk.write_vmec_profile or a profile conversion tool first"
        )
    with np.load(path, allow_pickle=False) as data:
        z = _profile_array(data, "z")
        field = _profile_array(data, "B")
        kmetric = _profile_array(data, "kperpMetric")
        drift_curvature = _profile_array(data, "driftCurvature")
        drift_parallel = _profile_array(data, "driftParallelCorrection")
        parallel_gradient = _profile_array(data, "parallelGradient", required=False)
        log_field = _profile_array(data, "logFieldDerivative", required=False)
        kmetric_derivative = _profile_array(data, "kperpMetricDerivative", required=False)
        period_data = np.asarray(data["period"]).reshape(-1) if "period" in data else None
        periodic_data = np.asarray(data["periodic"]).reshape(-1) if "periodic" in data else None
        q_data = np.asarray(data["vmecQ"]).reshape(-1) if "vmecQ" in data else None
        label_data = np.asarray(data["fieldLineLabel"]).reshape(-1) if "fieldLineLabel" in data else None

    arrays = [field, kmetric, drift_curvature, drift_parallel]
    if any(array.size != z.size for array in arrays):
        raise ValueError("all stellarator profile arrays must have the same length as z")
    if parallel_gradient is not None and parallel_gradient.size != z.size:
        raise ValueError("parallelGradient must have the same length as z")
    if log_field is not None and log_field.size != z.size:
        raise ValueError("logFieldDerivative must have the same length as z")
    if kmetric_derivative is not None and kmetric_derivative.size != z.size:
        raise ValueError("kperpMetricDerivative must have the same length as z")
    if np.any(field <= 0) or np.any(kmetric <= 0):
        raise ValueError("stellarator profile requires positive B and kperpMetric")
    if not np.all(np.diff(z) > 0):
        raise ValueError("stellarator profile z must be strictly increasing")

    dz = np.diff(z)
    if np.max(dz) - np.min(dz) > 1e-8 * max(1.0, float(np.mean(dz))):
        raise ValueError("stellarator profile z must be uniformly sampled")
    spacing = float(np.mean(dz))
    period = float(period_data[0]) if period_data is not None and period_data.size else float(z[-1] - z[0] + spacing)
    periodic = bool(periodic_data[0]) if periodic_data is not None and periodic_data.size else True
    if not np.isfinite(period) or period <= 0:
        raise ValueError("stellarator profile period must be positive")

    # Accept either a periodic grid without a duplicate endpoint or the
    # common endpoint-inclusive convention.  The latter is stripped so that
    # periodic interpolation never double-counts the first point.
    if periodic and abs((z[-1] - z[0]) - period) <= 1e-8 * max(1.0, period):
        z = z[:-1]
        field, kmetric, drift_curvature, drift_parallel = [item[:-1] for item in arrays]
        if parallel_gradient is not None:
            parallel_gradient = parallel_gradient[:-1]
        if log_field is not None:
            log_field = log_field[:-1]
        if kmetric_derivative is not None:
            kmetric_derivative = kmetric_derivative[:-1]
        spacing = float(np.mean(np.diff(z)))
    expected_range = z[-1] - z[0] + spacing if periodic else z[-1] - z[0]
    if abs(expected_range - period) > 1e-6 * max(1.0, period):
        raise ValueError("stellarator profile z range and period are inconsistent")

    if parallel_gradient is None:
        parallel_gradient = np.ones_like(field)
    if np.any(~np.isfinite(parallel_gradient)) or np.any(parallel_gradient == 0):
        raise ValueError("parallelGradient must be finite and nonzero")
    if log_field is None:
        log_field = (_periodic_derivative(np.log(field), spacing) if periodic
                     else np.gradient(np.log(field), spacing, edge_order=2))
    if kmetric_derivative is None:
        kmetric_derivative = (_periodic_derivative(kmetric, spacing) if periodic
                              else np.gradient(kmetric, spacing, edge_order=2))
    if label_data is not None and label_data.size:
        field_line_label = float(label_data[0])
    else:
        field_line_label = 0.0
    vmec_q = float(q_data[0]) if q_data is not None and q_data.size else None

    profile = Struct(
        filename=str(path), z=z, period=period, periodic=periodic, spacing=spacing,
        field=field, kperpMetric=kmetric,
        kperpMetricDerivative=kmetric_derivative,
        logFieldDerivative=log_field,
        driftCurvature=drift_curvature,
        driftParallelCorrection=drift_parallel,
        parallelGradient=parallel_gradient,
        fieldMinimum=float(np.min(field)), fieldMaximum=float(np.max(field)),
        fieldLineLabel=field_line_label, vmecQ=vmec_q,
    )
    _STELLARATOR_PROFILE_CACHE[key] = profile
    # Keep stale versions from holding large profiles indefinitely.
    for old_key in tuple(_STELLARATOR_PROFILE_CACHE):
        if old_key[0] == str(path) and old_key != key:
            del _STELLARATOR_PROFILE_CACHE[old_key]
    return profile


def _periodic_sample(profile, values, theta):
    theta = np.asarray(theta, dtype=float)
    if not profile.periodic:
        tolerance = 100 * np.finfo(float).eps * max(1.0, abs(profile.z[0]), abs(profile.z[-1]))
        if np.any(theta < profile.z[0] - tolerance) or np.any(theta > profile.z[-1] + tolerance):
            raise ValueError(
                f"non-periodic stellarator profile only covers [{profile.z[0]}, {profile.z[-1]}]"
            )
        return np.interp(theta.reshape(-1), profile.z, values).reshape(theta.shape)
    wrapped = profile.z[0] + np.mod(theta - profile.z[0], profile.period)
    extended_z = np.r_[profile.z, profile.z[0] + profile.period]
    extended_values = np.r_[values, values[0]]
    return np.interp(wrapped.reshape(-1), extended_z, extended_values).reshape(theta.shape)


def stellarator_profile(cfg):
    return load_stellarator_profile(cfg.geometryProfileFile)


def stellarator_well_centers(cfg):
    """Return local magnetic-well centers replicated over the requested z box."""
    profile = stellarator_profile(cfg)
    values = profile.field
    if not profile.periodic:
        minima = np.flatnonzero(
            (values[1:-1] <= values[:-2]) & (values[1:-1] < values[2:])
        ) + 1
        centers = profile.z[minima]
        return centers[(centers >= cfg.thmin) & (centers <= cfg.thmax)]
    previous = np.roll(values, 1)
    following = np.roll(values, -1)
    minima = np.flatnonzero((values <= previous) & (values < following))
    if minima.size == 0:
        minima = np.array([int(np.argmin(values))])
    base = profile.z[minima]
    lower, upper = float(cfg.thmin), float(cfg.thmax)
    tolerance = 100 * np.finfo(float).eps * max(
        1.0, abs(lower), abs(upper), profile.period
    )
    # Keep one representative for each physical well in the requested
    # coordinate box.  Adding an extra period on either side is tempting for
    # wrapped interpolation, but duplicates the same periodic well whenever
    # the domain spans one field period.  Bounce intervals are wrapped by the
    # path builder, so no outside copy is needed here.
    centers = []
    for value in base:
        first = int(np.ceil((lower - value - tolerance) / profile.period))
        last = int(np.floor((upper - value + tolerance) / profile.period))
        for period_index in range(first, last + 1):
            candidate = float(value + period_index * profile.period)
            # Treat the right endpoint as the same periodic representative as
            # the left endpoint.  For wider domains this also avoids a
            # duplicated endpoint while retaining all interior wells.
            if candidate >= upper - tolerance and abs(candidate - upper) <= tolerance:
                if abs(lower - (value + period_index * profile.period - profile.period)) <= tolerance:
                    continue
            if candidate >= lower - tolerance and candidate < upper - tolerance:
                centers.append(candidate)
    if not centers:
        # A profile whose minimum falls exactly on the boundary still needs a
        # representative well for a periodic domain.
        for value in base:
            distance = np.mod(lower - value, profile.period)
            if min(distance, profile.period - distance) <= tolerance:
                centers.append(lower)
                break
    return np.sort(np.unique(np.round(np.asarray(centers), 14)))


def stellarator_bounce_interval(cfg, target, center):
    """Find the nearest left/right roots of B(z)=target around a well center."""
    profile = stellarator_profile(cfg)
    center = float(center)
    if float(_periodic_sample(profile, profile.field, center)) >= target:
        raise ValueError("requested orbit is not trapped in the selected stellarator magnetic well")
    sample_count = max(256, 8 * profile.z.size)
    if profile.periodic:
        left_grid = np.linspace(center - 0.5 * profile.period, center, sample_count)
        right_grid = np.linspace(center, center + 0.5 * profile.period, sample_count)
    else:
        left_grid = np.linspace(profile.z[0], center, sample_count)
        right_grid = np.linspace(center, profile.z[-1], sample_count)
    left_values = _periodic_sample(profile, profile.field, left_grid) - target
    right_values = _periodic_sample(profile, profile.field, right_grid) - target
    left_indices = np.flatnonzero((left_values[:-1] >= 0) & (left_values[1:] <= 0))
    right_indices = np.flatnonzero((right_values[:-1] <= 0) & (right_values[1:] >= 0))
    if left_indices.size == 0 or right_indices.size == 0:
        raise ValueError("could not bracket stellarator trapped-orbit bounce points")
    li = int(left_indices[-1])
    ri = int(right_indices[0])
    left = brentq(lambda value: float(_periodic_sample(profile, profile.field, value) - target),
                  left_grid[li], left_grid[li + 1])
    right = brentq(lambda value: float(_periodic_sample(profile, profile.field, value) - target),
                   right_grid[ri], right_grid[ri + 1])
    return left, right


def evaluate_geometry(theta, cfg) -> Struct:
    theta = np.asarray(theta)
    if getattr(cfg, "geometryModel", "s-alpha") == "stellarator":
        profile = stellarator_profile(cfg)
        field = _periodic_sample(profile, profile.field, theta)
        metric = _periodic_sample(profile, profile.kperpMetric, theta)
        metric_derivative = _periodic_sample(profile, profile.kperpMetricDerivative, theta)
        k_perpendicular = cfg.kt * metric
        return Struct(
            theta=theta, shearCoordinate=np.zeros_like(theta),
            shearDerivative=np.zeros_like(theta), inverseAspectRatio=np.zeros_like(theta),
            fieldStrength=field,
            logFieldDerivative=_periodic_sample(profile, profile.logFieldDerivative, theta),
            localGyroradiusScale=1 / field,
            metricWaveNumberSquared=k_perpendicular**2,
            kPerpendicularSquared=k_perpendicular**2,
            kPerpendicularDerivative=cfg.kt * metric_derivative,
            driftCurvature=_periodic_sample(profile, profile.driftCurvature, theta),
            driftParallelCorrection=_periodic_sample(profile, profile.driftParallelCorrection, theta),
            parallelGradient=_periodic_sample(profile, profile.parallelGradient, theta),
            fieldLineLabel=profile.fieldLineLabel,
        )
    if getattr(cfg, "geometryModel", "s-alpha") == "miller":
        return _evaluate_miller(theta, cfg)
    shifted = theta - cfg.tk
    shear = cfg.s * shifted - cfg.alpha * np.sin(theta)
    inverse_field = 1 + cfg.epsilon * np.cos(theta)
    field = 1 / inverse_field
    log_derivative = cfg.epsilon * np.sin(theta) / inverse_field
    metric_k2 = cfg.kt**2 * (1 + shear**2)
    local_k2 = metric_k2 * inverse_field**2
    local_k = np.sqrt(local_k2)
    shear_derivative = cfg.s - cfg.alpha * np.cos(theta)
    wave_derivative = local_k * (
        shear * shear_derivative / (1 + shear**2) - log_derivative
    )
    # In s-alpha geometry, the MHD pressure gradient changes grad-B drift
    # without changing curvature drift.  Splitting the common energy factor
    # in the same convention as Miller/GACODE gives
    #   K_gradB = K_d - alpha/(2 q^2),
    #   K_curv  = K_gradB + alpha/(2 q^2).
    pressure_correction = cfg.alpha / (2 * cfg.q**2)
    return Struct(
        theta=theta, shearCoordinate=shear, shearDerivative=shear_derivative,
        inverseAspectRatio=cfg.epsilon, fieldStrength=field,
        logFieldDerivative=log_derivative, localGyroradiusScale=inverse_field,
        metricWaveNumberSquared=metric_k2, kPerpendicularSquared=local_k2,
        kPerpendicularDerivative=wave_derivative,
        driftCurvature=(np.cos(theta) + shear * np.sin(theta)
                        - pressure_correction),
        driftParallelCorrection=np.full_like(theta, pressure_correction),
        parallelGradient=np.ones_like(theta),
    )


def evaluate_trapping_field(theta, cfg):
    geometry = evaluate_geometry(theta, cfg)
    if (getattr(cfg, "geometryModel", "s-alpha") in {"miller", "stellarator"}
            or cfg.mirrorConvention == "physical"):
        return geometry.fieldStrength, geometry.logFieldDerivative
    cosine = np.cos(theta)
    field_minimum = 1 / (1 + cfg.epsilon)
    field = field_minimum * np.exp(
        cfg.epsilon * (1 - cosine) + 0.5 * cfg.epsilon**2 * (1 - cosine**2)
    )
    log_derivative = cfg.epsilon * np.sin(theta) * (1 + cfg.epsilon * cosine)
    return field, log_derivative


def _cumtrapz(values, coordinate):
    result = np.zeros_like(values)
    result[1:] = np.cumsum(
        0.5 * (values[1:] + values[:-1]) * np.diff(coordinate)
    )
    return result


def _periodic_derivative(values, spacing, jump=0.0):
    core = np.asarray(values[:-1])
    extended = np.r_[
        core[-2:] - jump, core, core[:2] + jump,
    ]
    n = core.size
    derivative = (
        -extended[4:n + 4] + 8 * extended[3:n + 3]
        - 8 * extended[1:n + 1] + extended[:n]
    ) / (12 * spacing)
    return np.r_[derivative, derivative[0]]


def _build_miller_table(cfg):
    key = (
        cfg.minorRadius, cfg.majorRadius, cfg.shiftDerivative, cfg.q,
        cfg.s, cfg.elongation, cfg.elongationShear, cfg.triangularity,
        cfg.triangularityShear, cfg.betaStar, cfg.tableResolution,
    )
    cached = _MILLER_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    n = int(cfg.tableResolution)
    theta = np.linspace(-np.pi, np.pi, n)
    dtheta = 2 * np.pi / (n - 1)
    r, r0, q = cfg.minorRadius, cfg.majorRadius, cfg.q
    x = np.arcsin(cfg.triangularity)
    angle = theta + x * np.sin(theta)
    angle_theta = 1 + x * np.cos(theta)
    angle_theta_theta = -x * np.sin(theta)

    major = r0 + r * np.cos(angle)
    major_radius = cfg.shiftDerivative + np.cos(angle) - np.sin(angle) * (
        cfg.triangularityShear / np.cos(x) * np.sin(theta)
    )
    major_theta = -r * angle_theta * np.sin(angle)
    major_theta_theta = (
        -r * angle_theta**2 * np.cos(angle)
        - r * angle_theta_theta * np.sin(angle)
    )
    elongation = cfg.elongation
    vertical = elongation * r * np.sin(theta)
    vertical_radius = elongation * (1 + cfg.elongationShear) * np.sin(theta)
    vertical_theta = elongation * r * np.cos(theta)
    vertical_theta_theta = -elongation * r * np.sin(theta)

    metric_theta = major_theta**2 + vertical_theta**2
    jacobian_radius = major * (major_radius * vertical_theta - major_theta * vertical_radius)
    grad_radius = major * np.sqrt(metric_theta) / jacobian_radius
    arc_theta = np.sqrt(metric_theta)
    curvature_radius = arc_theta**3 / (
        major_theta * vertical_theta_theta - vertical_theta * major_theta_theta
    )
    cos_normal = vertical_theta / arc_theta
    minus_sin_normal = major_theta / arc_theta
    arc_radius = cos_normal * vertical_radius + minus_sin_normal * major_radius
    normal_shear = (major_radius * major_theta + vertical_radius * vertical_theta) / arc_theta

    closed = slice(0, n - 1)
    f = r / (np.sum(arc_theta[closed] / (major[closed] * grad_radius[closed])) * dtheta / (2 * np.pi))
    toroidal = f / major
    poloidal = (r / q) * grad_radius / major
    field = np.sqrt(toroidal**2 + poloidal**2)

    periodic = field[closed]
    indices = np.arange(n - 1)
    plus2 = periodic[(indices + 2) % (n - 1)]
    plus1 = periodic[(indices + 1) % (n - 1)]
    minus1 = periodic[(indices - 1) % (n - 1)]
    minus2 = periodic[(indices - 2) % (n - 1)]
    field_theta = np.r_[
        (-plus2 + 8 * plus1 - 8 * minus1 + minus2) / (12 * dtheta), 0.0
    ]
    field_theta[-1] = field_theta[0]
    field_theta_theta = np.r_[
        (-plus2 + 16 * plus1 - 30 * periodic + 16 * minus1 - minus2)
        / (12 * dtheta**2), 0.0
    ]
    field_theta_theta[-1] = field_theta_theta[0]

    field_arc = field_theta / arc_theta
    generalized_sine = toroidal * r0 * field_arc / field**2
    generalized_cosine1 = (
        toroidal**2 / major * cos_normal
        + poloidal**2 / curvature_radius
    ) * r0 / field**2
    generalized_cosine2 = -0.5 * r0 * cfg.betaStar * grad_radius / field**2
    gtheta = major * field * arc_theta / (r * r0 * grad_radius)
    gq = r * field / (q * major * poloidal)
    c = dtheta * arc_theta / (major * grad_radius)
    toroidal_poloidal = toroidal / poloidal
    integrands = np.column_stack((
        c * 2 * toroidal_poloidal * (r / curvature_radius - r * cos_normal / major),
        c * field**2 / poloidal**2,
        c * grad_radius * 0.5 / poloidal**2 * toroidal_poloidal * cfg.betaStar,
        -c * grad_radius * toroidal_poloidal,
    ))
    integrals = np.zeros((n, 4))
    center = (n - 1) // 2
    for i in range(center + 1, n):
        integrals[i] = integrals[i - 1] + 0.5 * (integrands[i - 1] + integrands[i])
    for i in range(center - 1, -1, -1):
        integrals[i] = integrals[i + 1] - 0.5 * (integrands[i + 1] + integrands[i])
    loops = integrals[-1] - integrals[0]
    fprime = (2 * np.pi * q * cfg.s / r - loops[0] / r + loops[2]) / loops[1]
    nu = integrals[:, 3]
    cap_theta = poloidal / field * grad_radius * major * (
        integrals[:, 0] / r + integrals[:, 1] * fprime - integrals[:, 2]
    )
    theta_nc = _cumtrapz(gtheta, theta)
    theta_nc = -np.pi + 2 * np.pi * theta_nc / theta_nc[-1]
    theta_straight = _cumtrapz(jacobian_radius / major**2, theta)
    theta_straight = -np.pi + 2 * np.pi * theta_straight / theta_straight[-1]
    chi2 = (
        (poloidal * cos_normal - poloidal * major / curvature_radius
         + 0.5 * q / r * cfg.betaStar * major**2 - f * fprime)
        / (major * poloidal)**2
    ) / q + cfg.s / r**2

    matrix = np.column_stack((
        field, field_theta, field_theta_theta, poloidal, toroidal,
        generalized_sine, generalized_cosine1, generalized_cosine2,
        gtheta, grad_radius, gq, cap_theta, nu, arc_radius, arc_theta,
        normal_shear, -minus_sin_normal, toroidal / field * cos_normal,
        major, vertical, major_radius, major_theta, vertical_radius,
        vertical_theta, theta_nc, theta_straight, chi2,
        _periodic_derivative(gq, dtheta),
        _periodic_derivative(grad_radius, dtheta),
        _periodic_derivative(cap_theta, dtheta, cap_theta[-1] - cap_theta[0]),
    ))
    table = Struct(
        theta=theta, matrix=matrix,
        scalars=Struct(f=f, fPrime=fprime, ffPrime=f * fprime,
                       volumePrime=2 * np.pi * np.sum(arc_theta[closed] * major[closed] / grad_radius[closed]) * dtheta),
    )
    _MILLER_TABLE_CACHE[key] = table
    return table


def _sample_miller_table(table, theta):
    shape = np.shape(theta)
    values = np.asarray(theta).reshape(-1)
    periods = np.floor((values + np.pi) / (2 * np.pi)).astype(int)
    wrapped = values - periods * 2 * np.pi
    dtheta = table.theta[1] - table.theta[0]
    position = (wrapped - table.theta[0]) / dtheta
    left = np.floor(position).astype(int)
    left = np.clip(left, 0, len(table.theta) - 1)
    right = np.minimum(left + 1, len(table.theta) - 1)
    weight = position - left
    sampled = table.matrix[left] + (table.matrix[right] - table.matrix[left]) * weight[:, None]
    for index in (11, 12, 24, 25):
        sampled[:, index] += periods * (table.matrix[-1, index] - table.matrix[0, index])
    names = (
        "fieldStrength fieldTheta fieldThetaTheta poloidalField toroidalField "
        "generalizedSine generalizedCosine1 generalizedCosine2 gTheta gradRadius "
        "gQ capTheta nu arcRadius arcTheta normalShear sinNormal cosFieldNormal "
        "majorRadius verticalPosition majorRadiusRadialDerivative "
        "majorRadiusThetaDerivative verticalPositionRadialDerivative "
        "verticalPositionThetaDerivative thetaNC thetaStraight chi2 "
        "gQThetaDerivative gradRadiusThetaDerivative capThetaDerivative"
    ).split()
    return Struct(
        theta=np.asarray(theta),
        **{name: sampled[:, i].reshape(shape) for i, name in enumerate(names)},
        scalars=table.scalars,
    )


def _evaluate_miller(theta, cfg):
    raw = _sample_miller_table(_build_miller_table(cfg), theta)
    radial_wave = raw.gQ * raw.capTheta - cfg.s * cfg.tk * raw.gradRadius
    radial_derivative = (
        raw.gQThetaDerivative * raw.capTheta
        + raw.gQ * raw.capThetaDerivative
        - cfg.s * cfg.tk * raw.gradRadiusThetaDerivative
    )
    metric = np.sqrt(raw.gQ**2 + radial_wave**2)
    metric_derivative = (raw.gQ * raw.gQThetaDerivative + radial_wave * radial_derivative) / metric
    log_field = raw.fieldTheta / raw.fieldStrength
    wave = cfg.kt * metric / raw.fieldStrength
    wave_derivative = cfg.kt / raw.fieldStrength * (metric_derivative - metric * log_field)
    return Struct(
        theta=np.asarray(theta), fieldStrength=raw.fieldStrength,
        logFieldDerivative=log_field, kPerpendicularSquared=wave**2,
        kPerpendicularDerivative=wave_derivative,
        driftCurvature=(raw.gQ / raw.fieldStrength) * (
            raw.generalizedCosine1 + raw.generalizedCosine2
            + raw.capTheta * raw.generalizedSine
        ) - cfg.s * cfg.tk * raw.gradRadius / raw.fieldStrength * raw.generalizedSine,
        driftParallelCorrection=-raw.gQ / raw.fieldStrength * raw.generalizedCosine2,
        parallelGradient=1 / raw.gTheta,
    )
