"""Quadrature, LGL, velocity, and boundary discretizations."""

from __future__ import annotations

from math import pi, sqrt

import numpy as np
from scipy.optimize import brentq

from mgk._struct import Struct


def gauss_legendre(num_nodes: int, lower: float, upper: float):
    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    half = (upper - lower) / 2
    return (upper + lower) / 2 + half * nodes, half * weights


def _lgl(num_nodes: int):
    order = num_nodes - 1
    nodes = np.cos(pi * np.arange(num_nodes) / order)
    previous = np.full_like(nodes, 2.0)
    values = np.empty((num_nodes, num_nodes))
    while np.max(np.abs(nodes - previous)) > 1e-14:
        previous = nodes.copy()
        values[:, 0] = 1
        values[:, 1] = nodes
        for degree in range(2, order + 1):
            values[:, degree] = ((2 * degree - 1) * nodes * values[:, degree - 1]
                                 - (degree - 1) * values[:, degree - 2]) / degree
        nodes = previous - (nodes * values[:, order] - values[:, order - 1]) / (num_nodes * values[:, order])
    nodes = nodes[::-1]
    highest = values[:, order][::-1]
    weights = 2 / (order * num_nodes * highest**2)
    bary = np.empty(num_nodes)
    for row in range(num_nodes):
        differences = nodes[row] - nodes
        differences[row] = 1
        bary[row] = 1 / np.prod(differences)
    derivative = np.zeros((num_nodes, num_nodes))
    for row in range(num_nodes):
        mask = np.arange(num_nodes) != row
        derivative[row, mask] = bary[mask] / (bary[row] * (nodes[row] - nodes[mask]))
        derivative[row, row] = -np.sum(derivative[row, mask])
    return nodes, weights, derivative


def build_theta_operators(cfg) -> Struct:
    if cfg.parallelBoundary == "open-dtn":
        return _build_tail_element_operators(cfg)
    nodes, weights, reference_derivative = _lgl(cfg.nth)
    half = (cfg.thmax - cfg.thmin) / 2
    center = (cfg.thmax + cfg.thmin) / 2
    coordinate = center + half * nodes
    map_jacobian = np.ones_like(coordinate)
    if cfg.thetaMapAlpha != 0:
        normalized = (coordinate - center) / half
        alpha = cfg.thetaMapAlpha
        coordinate = center + half * np.sinh(alpha * normalized) / np.sinh(alpha)
        map_jacobian = alpha * np.cosh(alpha * normalized) / np.sinh(alpha)
    theta = coordinate.copy()
    jacobian = map_jacobian.copy()
    if cfg.boundaryCoordinateStretch != 1 and cfg.parallelBoundary != "periodic":
        normalized = (coordinate - center) / half
        radius = np.abs(normalized)
        start = 1 - cfg.boundarySpongeFraction
        ramp = np.maximum((radius - start) / cfg.boundarySpongeFraction, 0)
        theta = center + half * np.sign(normalized) * (radius + (cfg.boundaryCoordinateStretch - 1) * ramp**4)
        jacobian *= 1 + 4 * (cfg.boundaryCoordinateStretch - 1) / cfg.boundarySpongeFraction * ramp**3
    mass_diag = half * weights * jacobian
    ctheta = weights[:, None] * reference_derivative
    dtheta = ctheta / mass_diag[:, None]
    ctheta[np.abs(ctheta) < 1e-14] = 0
    dtheta[np.abs(dtheta) < 1e-14] = 0
    return Struct(theta=theta, massDiag=mass_diag, cTheta=ctheta, dTheta=dtheta, nth=cfg.nth)


def _build_tail_element_operators(cfg) -> Struct:
    """Assemble continuous LGL elements for the real-coordinate DtN tails."""
    num_tail = cfg.boundaryTailPeriods
    element_points = [cfg.boundaryTailPoints] * num_tail
    breaks = list(cfg.thmin + np.arange(num_tail + 1) * 2 * pi)
    if cfg.boundaryLeftTransitionPoints > 0:
        element_points.append(cfg.boundaryLeftTransitionPoints)
        breaks.append(cfg.boundaryCoreMin)
    element_points.append(cfg.boundaryCoreNumTheta)
    breaks.append(cfg.boundaryCoreMax)
    if cfg.boundaryRightTransitionPoints > 0:
        element_points.append(cfg.boundaryRightTransitionPoints)
        breaks.append(cfg.boundaryRightTailInterface)
    element_points.extend([cfg.boundaryTailPoints] * num_tail)
    breaks.extend(cfg.boundaryRightTailInterface + np.arange(1, num_tail + 1) * 2 * pi)
    num_elements = len(element_points)
    num_nodes = sum(element_points) - num_elements + 1
    theta = np.zeros(num_nodes)
    mass_diag = np.zeros(num_nodes)
    ctheta = np.zeros((num_nodes, num_nodes))
    elements = []
    first = 0
    for count, left, right in zip(element_points, breaks[:-1], breaks[1:]):
        nodes, weights, derivative = _lgl(count)
        half = 0.5 * (right - left)
        center = 0.5 * (right + left)
        local_theta = center + half * nodes
        indices = np.arange(first, first + count)
        theta[indices] = local_theta
        mass_diag[indices] += half * weights
        ctheta[np.ix_(indices, indices)] += weights[:, None] * derivative
        bary = np.empty(count)
        for row in range(count):
            differences = local_theta[row] - local_theta
            differences[row] = 1
            bary[row] = 1 / np.prod(differences)
        bary /= np.max(np.abs(bary))
        elements.append(Struct(
            indices=indices, theta=local_theta, barycentricWeights=bary,
            derivative=derivative / half,
        ))
        first += count - 1
    dtheta = ctheta / mass_diag[:, None]
    ctheta[np.abs(ctheta) < 1e-14] = 0
    dtheta[np.abs(dtheta) < 1e-14] = 0
    tolerance = 100 * np.finfo(float).eps * max(1.0, np.max(np.abs(theta)))
    core = ((theta >= cfg.boundaryCoreMin - tolerance)
            & (theta <= cfg.boundaryCoreMax + tolerance))
    return Struct(
        theta=theta, massDiag=mass_diag, cTheta=ctheta, dTheta=dtheta,
        nth=num_nodes, elements=elements, coreIndices=np.flatnonzero(core),
        tailIndices=np.flatnonzero(~core),
    )


def build_orbit_boundary(cfg, theta_ops) -> Struct:
    nfull = theta_ops.nth
    if cfg.parallelBoundary == "periodic":
        nfield = nfull - 1
        prolongation = np.zeros((nfull, nfield))
        prolongation[np.arange(nfull), np.r_[np.arange(nfield), 0]] = 1
        weights = prolongation.T @ theta_ops.massDiag
        derivative = (prolongation.T @ theta_ops.cTheta @ prolongation) / weights[:, None]
        moment = (
            (prolongation.T * theta_ops.massDiag) @ prolongation
        ) / weights[:, None]
        branch = Struct(active=np.arange(nfield), derivative=derivative,
                        fieldDerivative=derivative, fieldValue=np.eye(nfield), momentMap=moment)
        branches = [branch, branch]
        indices = np.arange(nfield)
    else:
        indices = np.arange(1, nfull - 1)
        nfield = len(indices)
        prolongation = np.zeros((nfull, nfield))
        prolongation[indices, np.arange(nfield)] = 1
        if cfg.parallelBoundary in {"open-extrapolated", "open-dtn"}:
            left = (theta_ops.theta[0] - theta_ops.theta[1]) / (theta_ops.theta[2] - theta_ops.theta[1])
            right = (theta_ops.theta[-1] - theta_ops.theta[-2]) / (theta_ops.theta[-3] - theta_ops.theta[-2])
            prolongation[0, :2] = [1 - left, left]
            prolongation[-1, -2:] = [right, 1 - right]
        weights = prolongation.T @ theta_ops.massDiag
        branches = []
        for active in (np.arange(nfull - 1), np.arange(1, nfull)):
            moment = (prolongation[active, :].T * theta_ops.massDiag[active]) / weights[:, None]
            branches.append(Struct(
                active=active, derivative=theta_ops.dTheta[np.ix_(active, active)],
                fieldDerivative=theta_ops.dTheta[np.ix_(active, np.arange(nfull))] @ prolongation,
                fieldValue=prolongation[active, :], momentMap=moment,
            ))
    core_indices = np.arange(len(indices))
    if cfg.parallelBoundary == "open-dtn":
        tolerance = 100 * np.finfo(float).eps * max(1.0, np.max(np.abs(theta_ops.theta)))
        field_theta = theta_ops.theta[indices]
        core_indices = np.flatnonzero(
            (field_theta >= cfg.boundaryCoreMin - tolerance)
            & (field_theta <= cfg.boundaryCoreMax + tolerance)
        )
    return Struct(fieldIndices=indices, fieldTheta=theta_ops.theta[indices], fieldWeights=weights,
                  fieldProlongation=prolongation, numFieldTheta=len(indices),
                  numPassingTheta=nfull - 1, coreFieldIndices=core_indices,
                  passing=branches)


def build_orbit_kinematics(cfg, theta, _species=None) -> Struct:
    from .geometry import (
        evaluate_geometry, evaluate_trapping_field, stellarator_profile,
        stellarator_well_centers,
    )

    if cfg.geometryModel == "stellarator":
        profile = stellarator_profile(cfg)
        physical_minimum = profile.fieldMinimum
        physical_maximum = profile.fieldMaximum
        field_minimum, field_maximum = physical_minimum, physical_maximum
    else:
        physical_minimum = 1 / (1 + cfg.epsilon)
        physical_maximum = 1 / (1 - cfg.epsilon)
        extrema, _ = evaluate_trapping_field(np.array([0.0, pi]), cfg)
        field_minimum, field_maximum = extrema
    if cfg.geometryModel == "miller":
        if field_minimum >= field_maximum:
            raise ValueError("Miller orbit geometry requires B(0) < B(pi)")
        physical_minimum = float(field_minimum)
        physical_maximum = float(field_maximum)
    separatrix = sqrt(max(0.0, 1 - field_minimum / field_maximum))
    pitch_nodes, pitch_weights, pitch_region = _split_pitch_quadrature(
        cfg, field_minimum, separatrix
    )
    speed_nodes, speed_weights = gauss_legendre(cfg.numEnergy, 0, sqrt(2 * cfg.energyMax))
    speed, pitch = np.meshgrid(speed_nodes, pitch_nodes, indexing="ij")
    speed_weight, pitch_weight = np.meshgrid(speed_weights, pitch_weights, indexing="ij")
    flatten = lambda value: np.asarray(value).reshape(-1, order="F")
    energy = flatten(speed) ** 2 / 2
    pitch_at_minimum = flatten(pitch)
    reference_weight = 2 * pi * flatten(speed) ** 2 * flatten(speed_weight) * flatten(pitch_weight)
    lambda_value = (1 - pitch_at_minimum**2) / field_minimum
    passing = lambda_value * field_maximum < 1
    trapped = ~passing
    theta = np.asarray(theta).reshape(-1)
    geometry = evaluate_geometry(theta, cfg)
    orbit_field, _ = evaluate_trapping_field(theta, cfg)
    accessible = 1 - lambda_value[:, None] * orbit_field[None, :] >= 0
    parallel_speed = np.sqrt(2 * energy[:, None] * np.maximum(1 - lambda_value[:, None] * orbit_field[None, :], 0))
    perpendicular_speed = np.sqrt(2 * energy[:, None] * lambda_value[:, None] * orbit_field[None, :])
    reference_parallel = np.sqrt(2 * energy) * pitch_at_minimum
    safe_parallel = np.maximum(parallel_speed, sqrt(np.finfo(float).eps))
    weight_ratio = orbit_field[None, :] / field_minimum * reference_parallel[:, None] / safe_parallel
    weight_ratio[~accessible] = 0
    bounce_angle = np.full(len(energy), np.nan)
    if cfg.geometryModel == "miller" and np.any(trapped):
        trapped_indices = np.flatnonzero(trapped)
        for orbit_index in trapped_indices:
            target = 1 / lambda_value[orbit_index]
            bounce_angle[orbit_index] = brentq(
                lambda angle: float(
                    evaluate_trapping_field(np.asarray(angle), cfg)[0] - target
                ),
                0.0, pi,
            )
    elif cfg.epsilon > 0 and np.any(trapped):
        bounce_log = np.log(1 / (lambda_value[trapped] * field_minimum))
        if cfg.mirrorConvention == "physical":
            argument = (lambda_value[trapped] - 1) / cfg.epsilon
        else:
            argument = (-1 + np.sqrt((1 + cfg.epsilon) ** 2 - 2 * bounce_log)) / cfg.epsilon
        bounce_angle[trapped] = np.arccos(np.clip(argument, -1, 1))
    elif cfg.geometryModel == "stellarator" and np.any(trapped):
        # Exact turning points are found separately for every magnetic well.
        # This half-period bound is used only for conservative topology
        # containment before those roots are evaluated.
        bounce_angle[trapped] = 0.5 * profile.period
    first_well = int(np.ceil(cfg.thmin / (2 * pi)))
    last_well = int(np.floor(cfg.thmax / (2 * pi)))
    result = Struct(
        energy=energy, pitchCosineAtFieldMinimum=pitch_at_minimum,
        referenceIntegrationWeight=reference_weight, passing=passing, trapped=trapped,
        bounceAngle=bounce_angle,
        wellCenters=(stellarator_well_centers(cfg)
                     if cfg.geometryModel == "stellarator"
                     else 2 * pi * np.arange(first_well, last_well + 1)),
        theta=theta, fieldStrength=geometry.fieldStrength,
        orbitFieldStrength=orbit_field, accessible=accessible,
        parallelSpeed=parallel_speed, parallelVelocityPositive=parallel_speed,
        parallelVelocityNegative=-parallel_speed, perpendicularSpeed=perpendicular_speed,
        integrationWeightPerSign=reference_weight[:, None] * weight_ratio,
        fieldMinimum=field_minimum, fieldMaximum=field_maximum,
        physicalFieldMinimum=physical_minimum, physicalFieldMaximum=physical_maximum,
        separatrixPitch=separatrix, pitchCosineNodes=pitch_nodes,
        pitchWeights=pitch_weights, pitchRegion=pitch_region,
        numPitchNodes=len(pitch_nodes), numOrbits=len(energy),
    )
    result["lambda"] = lambda_value
    return result


def _split_pitch_quadrature(cfg, field_minimum, separatrix):
    if separatrix == 0:
        nodes, weights = gauss_legendre(cfg.numPitch, 0, 1)
        return nodes, weights, np.full(cfg.numPitch, "passing")
    ntrapped = cfg.numPitch // 2
    npassing = cfg.numPitch - ntrapped
    if cfg.geometryModel == "stellarator":
        # A general 3-D field has no monotonic half-period from which to use
        # the s-alpha cosine Jacobian.  Integrate directly in the pitch cosine
        # at the global field minimum; the orbit roots for each magnetic well
        # are found later from B(z)=1/lambda.  This keeps all quadrature
        # weights positive even when a field line contains multiple wells.
        trapped_nodes, trapped_weights = gauss_legendre(
            ntrapped, 0.0, separatrix
        )
        passing_nodes, passing_weights = gauss_legendre(
            npassing, separatrix, 1.0
        )
        return (
            np.r_[trapped_nodes, passing_nodes],
            np.r_[trapped_weights, passing_weights],
            np.r_[np.full(ntrapped, "trapped"), np.full(npassing, "passing")],
        )
    bounce_nodes, bounce_weights = gauss_legendre(ntrapped, 0, pi)
    from .geometry import evaluate_trapping_field
    bounce_field, bounce_log_derivative = evaluate_trapping_field(bounce_nodes, cfg)
    trapped_nodes = np.sqrt(np.maximum(0, 1 - field_minimum / bounce_field))
    jacobian = field_minimum * bounce_log_derivative / (2 * bounce_field * trapped_nodes)
    trapped_weights = bounce_weights * jacobian
    trapped_weights *= separatrix / np.sum(trapped_weights)
    passing_nodes, passing_weights = gauss_legendre(npassing, separatrix, 1)
    return (np.r_[trapped_nodes, passing_nodes], np.r_[trapped_weights, passing_weights],
            np.r_[np.full(ntrapped, "trapped"), np.full(npassing, "passing")])


def build_orbit_layout(cfg, orbits) -> Struct:
    passing_indices = np.flatnonzero(orbits.passing)
    trapped_indices = np.flatnonzero(orbits.trapped)
    if cfg.geometryModel == "stellarator":
        from .geometry import (
            stellarator_bounce_interval, stellarator_profile,
            stellarator_well_centers,
        )
        profile = stellarator_profile(cfg)
        candidate_centers = stellarator_well_centers(cfg)
        trapped_pairs = []
        for center in candidate_centers:
            for orbit_index in trapped_indices:
                target = 1.0 / orbits["lambda"][orbit_index]
                try:
                    left, right = stellarator_bounce_interval(
                        cfg, target, center
                    )
                except ValueError:
                    # A trapped pitch orbit need not be trapped in every
                    # local magnetic well of a 3-D field.
                    continue
                if cfg.parallelBoundary != "periodic":
                    if left < cfg.thmin or right > cfg.thmax:
                        continue
                trapped_pairs.append((center, orbit_index, left, right))
        npassing = 2 * len(passing_indices)
        ntrapped = len(trapped_pairs)
        nblocks = npassing + ntrapped
        block_type = np.empty(nblocks, dtype="U7")
        orbit_index = np.empty(nblocks, dtype=int)
        velocity_sign = np.zeros(nblocks)
        well_center = np.full(nblocks, np.nan)
        theta_start = np.empty(nblocks)
        theta_end = np.empty(nblocks)
        branches = np.ones(nblocks, dtype=int)
        block = 0
        for sign in (-1, 1):
            span = slice(block, block + len(passing_indices))
            block_type[span] = "passing"
            orbit_index[span] = passing_indices
            velocity_sign[span] = sign
            theta_start[span], theta_end[span] = cfg.thmin, cfg.thmax
            block += len(passing_indices)
        for center, orbit, left, right in trapped_pairs:
            block_type[block] = "trapped"
            orbit_index[block] = orbit
            well_center[block] = center
            theta_start[block], theta_end[block] = left, right
            branches[block] = 2
            block += 1
        used_centers = np.unique([item[0] for item in trapped_pairs]) if trapped_pairs else np.empty(0)
        return Struct(
            blockType=block_type, orbitIndex=orbit_index,
            velocitySign=velocity_sign, wellCenter=well_center,
            thetaStart=theta_start, thetaEnd=theta_end,
            numBranches=branches, wrapsBoundary=(theta_start < cfg.thmin) | (theta_end > cfg.thmax),
            wellCenters=used_centers, numPassingBlocks=npassing,
            numTrappedBlocks=ntrapped, numBlocks=nblocks,
        )
    first_well = int(np.ceil(cfg.thmin / (2 * pi)))
    last_well = int(np.ceil(cfg.thmax / (2 * pi))) - 1
    well_centers = 2 * pi * np.arange(first_well, last_well + 1)
    if cfg.parallelBoundary != "periodic" and trapped_indices.size:
        # An open field line cannot connect a trapped bounce orbit through
        # the opposite end of the computational interval.  Keep only wells
        # whose widest trapped orbit is fully contained in the domain.  The
        # old Python path retained wrapped wells here, producing extra blocks
        # and a subtly different open-boundary problem.
        maximum_bounce_angle = float(np.max(orbits.bounceAngle[trapped_indices]))
        tolerance = 100 * np.finfo(float).eps * max(
            1.0, abs(float(cfg.thmin)), abs(float(cfg.thmax))
        )
        contained = (
            well_centers - maximum_bounce_angle >= cfg.thmin - tolerance
        ) & (
            well_centers + maximum_bounce_angle <= cfg.thmax + tolerance
        )
        well_centers = well_centers[contained]
    npassing = 2 * len(passing_indices)
    ntrapped = len(trapped_indices) * len(well_centers)
    nblocks = npassing + ntrapped
    block_type = np.empty(nblocks, dtype="U7")
    orbit_index = np.empty(nblocks, dtype=int)
    velocity_sign = np.zeros(nblocks)
    well_center = np.full(nblocks, np.nan)
    theta_start = np.empty(nblocks)
    theta_end = np.empty(nblocks)
    branches = np.ones(nblocks, dtype=int)
    block = 0
    for sign in (-1, 1):
        span = slice(block, block + len(passing_indices))
        block_type[span] = "passing"
        orbit_index[span] = passing_indices
        velocity_sign[span] = sign
        theta_start[span], theta_end[span] = cfg.thmin, cfg.thmax
        block += len(passing_indices)
    for center in well_centers:
        span = slice(block, block + len(trapped_indices))
        angle = orbits.bounceAngle[trapped_indices]
        block_type[span] = "trapped"
        orbit_index[span] = trapped_indices
        well_center[span] = center
        theta_start[span], theta_end[span] = center - angle, center + angle
        branches[span] = 2
        block += len(trapped_indices)
    return Struct(
        blockType=block_type, orbitIndex=orbit_index, velocitySign=velocity_sign,
        wellCenter=well_center, thetaStart=theta_start, thetaEnd=theta_end,
        numBranches=branches, wrapsBoundary=(theta_start < cfg.thmin) | (theta_end > cfg.thmax),
        wellCenters=well_centers, numPassingBlocks=npassing,
        numTrappedBlocks=ntrapped, numBlocks=nblocks,
    )


def _fourier_derivative(num_points):
    indices = np.arange(num_points)
    difference = indices[:, None] - indices[None, :]
    derivative = np.zeros((num_points, num_points))
    mask = difference != 0
    derivative[mask] = 0.5 * (-1.0) ** difference[mask] / np.tan(pi * difference[mask] / num_points)
    return derivative


def build_trapped_path(cfg, orbits, orbit_index, well_center, num_points=64, theta_ops=None) -> Struct:
    from .geometry import evaluate_geometry, evaluate_trapping_field

    if theta_ops is None:
        theta_ops = build_theta_operators(cfg)
    if not orbits.trapped[orbit_index]:
        raise ValueError("The selected orbit is not trapped")
    if num_points < 8 or num_points % 2:
        raise ValueError("num_points must be an even integer of at least 8")
    energy, lambda_value = orbits.energy[orbit_index], orbits["lambda"][orbit_index]
    # s-alpha/Miller use a symmetric angle around the well center.  A
    # stellarator well is generally asymmetric, so obtain its actual turning
    # points and map the periodic bounce coordinate onto that interval.
    if cfg.geometryModel == "stellarator":
        from .geometry import stellarator_bounce_interval
        left, right = stellarator_bounce_interval(
            cfg, 1.0 / lambda_value, well_center
        )
        path_center = 0.5 * (left + right)
        half_width = 0.5 * (right - left)
    else:
        left = right = None
        path_center = float(well_center)
        half_width = float(orbits.bounceAngle[orbit_index])
    angle = half_width
    reference_weight = orbits.referenceIntegrationWeight[orbit_index]
    reference_parallel = sqrt(2 * energy) * orbits.pitchCosineAtFieldMinimum[orbit_index]
    chi = -pi / 2 + np.arange(num_points) * 2 * pi / num_points
    theta_unwrapped = path_center + half_width * np.sin(chi)
    from .geometry import stellarator_profile
    profile_is_periodic = (
        stellarator_profile(cfg).periodic if cfg.geometryModel == "stellarator" else True
    )
    if cfg.geometryModel == "stellarator" and not profile_is_periodic:
        theta_wrapped = theta_unwrapped
    else:
        theta_wrapped = cfg.thmin + np.mod(theta_unwrapped - cfg.thmin, cfg.thmax - cfg.thmin)
    geometry = evaluate_geometry(theta_unwrapped, cfg)
    orbit_field, orbit_log_derivative = evaluate_trapping_field(theta_unwrapped, cfg)
    parallel_speed = np.sqrt(np.maximum(2 * energy * (1 - lambda_value * orbit_field), 0))
    cosine = np.cos(chi)
    stream = np.zeros_like(chi)
    regular = np.abs(cosine) > 100 * np.finfo(float).eps
    stream[regular] = parallel_speed[regular] / (half_width * np.abs(cosine[regular]))
    bounce = ~regular
    field_derivative = orbit_field * orbit_log_derivative
    stream[bounce] = np.sqrt(
        energy * lambda_value * np.abs(field_derivative[bounce]) / half_width
    )
    parallel_velocity = stream * half_width * cosine
    perpendicular = np.sqrt(2 * energy * lambda_value * orbit_field)
    path_weight = (2 * pi / num_points * reference_weight * orbit_field / orbits.fieldMinimum
                   * reference_parallel / stream)
    boundary = build_orbit_boundary(cfg, theta_ops)
    nodes = theta_ops.theta
    interpolation_full = np.zeros((num_points, theta_ops.nth))
    derivative_full = np.zeros_like(interpolation_full)
    if "elements" in theta_ops:
        tolerance = 100 * np.finfo(float).eps * max(1.0, np.max(np.abs(nodes)))
        for row, target in enumerate(theta_wrapped):
            element = next((item for item in theta_ops.elements
                            if target >= item.theta[0] - tolerance
                            and target <= item.theta[-1] + tolerance), None)
            if element is None:
                raise ValueError("trapped-path interpolation target lies outside the theta domain")
            differences = target - element.theta
            local_node = int(np.argmin(np.abs(differences)))
            if abs(differences[local_node]) < 1e-13:
                node = element.indices[local_node]
                interpolation_full[row, node] = 1
                derivative_full[row, :] = theta_ops.dTheta[node, :]
            else:
                values = element.barycentricWeights / differences
                values /= np.sum(values)
                interpolation_full[row, element.indices] = values
                derivative_full[row, element.indices] = values @ element.derivative
    else:
        bary = (-1.0) ** np.arange(theta_ops.nth) * np.sqrt(theta_ops.massDiag)
        for row, target in enumerate(theta_wrapped):
            differences = target - nodes
            node = np.argmin(np.abs(differences))
            if abs(differences[node]) < 1e-13:
                interpolation_full[row, node] = 1
            else:
                values = bary / differences
                interpolation_full[row, :] = values / np.sum(values)
        derivative_full = interpolation_full @ theta_ops.dTheta
    interpolation = interpolation_full @ boundary.fieldProlongation
    derivative_interpolation = derivative_full @ boundary.fieldProlongation
    deposition = (interpolation.T * path_weight) / boundary.fieldWeights[:, None]
    derivative = _fourier_derivative(num_points)
    result = Struct(
        orbitIndex=orbit_index, referenceIntegrationWeight=reference_weight,
        wellCenter=well_center, pathCenter=path_center,
        leftBounce=left, rightBounce=right,
        energy=energy, bounceAngle=angle, chi=chi,
        theta=theta_wrapped, thetaUnwrapped=theta_unwrapped, geometry=geometry,
        fieldStrength=geometry.fieldStrength, orbitFieldStrength=orbit_field,
        parallelVelocity=parallel_velocity, perpendicularSpeed=perpendicular,
        streamCoefficient=stream, phaseSpaceThetaWeight=path_weight,
        derivative=derivative, fieldInterpolation=interpolation,
        orbitDerivativeInterpolation=derivative @ interpolation,
        fieldDerivativeInterpolation=derivative_interpolation,
        fieldDeposition=deposition, numPoints=num_points,
    )
    result["lambda"] = lambda_value
    return result
