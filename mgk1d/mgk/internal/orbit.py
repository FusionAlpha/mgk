"""Grouped passing/trapped orbit assembly and Schur-complement solve."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from math import pi, sqrt
from time import perf_counter

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import j0, j1

from mgk._struct import Struct
from .discretization import (
    build_orbit_boundary,
    build_orbit_kinematics,
    build_orbit_layout,
    build_theta_operators,
    build_trapped_path,
)
from .eigensolver import solve_eigenproblem
from .geometry import evaluate_geometry
from .physics import resolve_kinetic_species, two_j1_over_argument
from .selection import solve_selected
from .cache import TRANSIENT_FIELDS, freeze

_ORBIT_ASSEMBLY_CACHE = None
_ORBIT_FACTORIZATION_CACHE = None
_ORBIT_TOPOLOGY_CACHE = None
_ORBIT_WARM_START_CACHE = {}
_ORBIT_CONTINUATION_CACHE = {}
_SHARED_PAGE_EXECUTORS = {}
_SHARED_GROUP_EXECUTORS = {}


class _PageWorkerPool:
    """Run independent CPU page operations with deterministic chunk ordering."""

    def __init__(self, workers, shared=False):
        self.workers = max(1, int(workers))
        self.owns_executor = False
        if self.workers <= 1:
            self.executor = None
        elif shared:
            if self.workers not in _SHARED_PAGE_EXECUTORS:
                _SHARED_PAGE_EXECUTORS[self.workers] = ThreadPoolExecutor(
                    max_workers=self.workers
                )
            self.executor = _SHARED_PAGE_EXECUTORS[self.workers]
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.workers)
            self.owns_executor = True

    def close(self):
        if self.executor is not None and self.owns_executor:
            self.executor.shutdown()

    def _slices(self, count):
        pieces = min(count, self.workers)
        if pieces <= 1:
            return [slice(0, count)]
        edges = np.linspace(0, count, pieces + 1, dtype=int)
        return [slice(int(start), int(stop))
                for start, stop in zip(edges[:-1], edges[1:])
                if start < stop]

    def inverse(self, pages):
        # Page-first moveaxis views must be contiguous to reach batched BLAS.
        pages = np.ascontiguousarray(pages)
        if self.executor is None or len(pages) < 2:
            return np.linalg.inv(pages)
        result = np.empty_like(pages)

        def task(part):
            result[part] = np.linalg.inv(pages[part])

        list(self.executor.map(task, self._slices(len(pages))))
        return result

    def matmul(self, left, right):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        if self.executor is None or len(left) < 2:
            return np.matmul(left, right)
        result = np.empty(
            (left.shape[0], left.shape[1], right.shape[2]),
            dtype=np.result_type(left, right),
        )

        def task(part):
            np.matmul(left[part], right[part], out=result[part])

        list(self.executor.map(task, self._slices(len(left))))
        return result

    def matmul_shared_right_pair(self, left, other, right):
        left = np.ascontiguousarray(left)
        other = np.ascontiguousarray(other)
        right = np.ascontiguousarray(right)
        if self.executor is None or len(left) < 2:
            return np.matmul(left, right), np.matmul(other, right)
        shape = (left.shape[0], left.shape[1], right.shape[1])
        first = np.empty(shape, dtype=np.result_type(left, right))
        second = np.empty(shape, dtype=np.result_type(other, right))

        def task(part):
            np.matmul(left[part], right, out=first[part])
            np.matmul(other[part], right, out=second[part])

        list(self.executor.map(task, self._slices(len(left))))
        return first, second

    def matmul_shared_right(self, left, right):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        if self.executor is None or len(left) < 2:
            return np.matmul(left, right)
        shape = (left.shape[0], left.shape[1], right.shape[1])
        result = np.empty(shape, dtype=np.result_type(left, right))

        def task(part):
            np.matmul(left[part], right, out=result[part])

        list(self.executor.map(task, self._slices(len(left))))
        return result

    def contract_shared_vector_pair(self, left, other, right):
        """Contract page-last tensors with the same field vectors."""
        shape = (left.shape[0], left.shape[2], right.shape[1])
        if self.executor is None or len(left) < 2:
            return (
                np.einsum("ifb,fk->ibk", left, right, optimize=False),
                np.einsum("ifb,fk->ibk", other, right, optimize=False),
            )
        first = np.empty(shape, dtype=np.result_type(left, right))
        second = np.empty(shape, dtype=np.result_type(other, right))

        def task(part):
            np.einsum(
                "ifb,fk->ibk", left[part], right,
                out=first[part], optimize=False,
            )
            np.einsum(
                "ifb,fk->ibk", other[part], right,
                out=second[part], optimize=False,
            )

        list(self.executor.map(task, self._slices(len(left))))
        return first, second

    def sum_matmul(self, left, right):
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        if self.executor is None or len(left) < 2:
            return np.sum(np.matmul(left, right), axis=0)

        def task(part):
            return np.sum(np.matmul(left[part], right[part]), axis=0)

        partial = list(self.executor.map(task, self._slices(len(left))))
        return np.sum(partial, axis=0)


def _orbit_solver_cache_keys(cfg, num_bounce):
    cache_enabled = (
        cfg.enableGpuFactorizationCache if cfg.useGpu
        else cfg.enableFactorizationCache
    )
    if not cache_enabled:
        return None, None
    factor_key = freeze(
        cfg,
        TRANSIENT_FIELDS | {"etai", "temperatureGradientLength", "temperatureGradientRatio"},
    )
    # Use the same complete static signature as the factorization cache for
    # eigenvector warm starts.  The previous abbreviated key could collide
    # across geometry/physics changes (for example different ballooning
    # angles), causing an unrelated mode to seed a new solve.
    warm_key = factor_key
    return factor_key, warm_key


def _unit_mode(value):
    value = np.asarray(value, dtype=complex).reshape(-1)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        return None
    return value / norm


def _continuation_parameters(cfg, kinetic_species):
    return np.asarray(
        [cfg.kt]
        + [item.temperatureGradientRatio for item in kinetic_species],
        dtype=float,
    )


def _warm_ritz_predictor(system, cfg, parameters):
    """Return a safe phase-aligned linear or quadratic continuation predictor."""
    if (not cfg.enableWarmRitz or cfg.modeSelection != "nearest"
            or cfg.eigsInitialVector is None or system.warmStartKey is None):
        return None, None
    history = list(_ORBIT_CONTINUATION_CACHE.get(system.warmStartKey, []))
    initial = _unit_mode(cfg.eigsInitialVector)
    if initial is None:
        return None, None
    matching = [
        index for index, entry in enumerate(history)
        if initial.size == np.asarray(entry.mode).size
        and abs(np.vdot(initial, _unit_mode(entry.mode))) >= 1 - 1e-10
    ]
    if not matching:
        return None, None
    history = history[:matching[-1] + 1]
    if len(history) < 2:
        first_step = np.asarray(parameters) - np.asarray(
            history[-1].parameters
        )
        k_scan = abs(first_step[0]) > 10 * np.finfo(float).eps
        eta_unchanged = np.linalg.norm(first_step[1:]) <= (
            10 * np.finfo(float).eps
            * max(1, np.linalg.norm(parameters[1:]))
        )
        if k_scan and eta_unchanged:
            return initial, 15
        return None, None
    previous, latest = history[-2], history[-1]
    latest_mode = _unit_mode(latest.mode)
    previous_mode = _unit_mode(previous.mode)
    if latest_mode is None or previous_mode is None:
        return None, None
    if initial.size != latest_mode.size or initial.size != previous_mode.size:
        return None, None
    latest_overlap = abs(np.vdot(initial, latest_mode))
    if latest_overlap < 1 - 1e-10:
        return None, None
    previous_step = np.asarray(latest.parameters) - np.asarray(previous.parameters)
    current_step = np.asarray(parameters) - np.asarray(latest.parameters)
    denominator = float(previous_step @ previous_step)
    if denominator <= np.finfo(float).eps:
        return None, None
    scale = float(current_step @ previous_step) / denominator
    orthogonal_step = current_step - scale * previous_step
    parameter_scale = max(
        np.linalg.norm(current_step), np.linalg.norm(previous_step),
        np.finfo(float).eps,
    )
    if (np.linalg.norm(orthogonal_step) > 1e-10 * parameter_scale
            or abs(scale) > 2 or abs(scale) < 1e-12):
        return None, None
    phase = np.vdot(initial, previous_mode)
    if phase == 0:
        return None, None
    previous_mode *= np.exp(-1j * np.angle(phase))
    linear_predictor = initial + scale * (initial - previous_mode)
    k_scan = abs(current_step[0]) > 10 * np.finfo(float).eps
    linear_dimension = 8 if k_scan else 3
    if len(history) < 3:
        return linear_predictor, linear_dimension

    older = history[-3]
    older_mode = _unit_mode(older.mode)
    if older_mode is None or older_mode.size != initial.size:
        return linear_predictor, linear_dimension
    older_offset = np.asarray(older.parameters) - np.asarray(latest.parameters)
    older_coordinate = float(older_offset @ previous_step) / denominator
    older_orthogonal = older_offset - older_coordinate * previous_step
    if (np.linalg.norm(older_orthogonal) > 1e-10 * parameter_scale
            or older_coordinate >= -1 - 1e-8):
        return linear_predictor, linear_dimension
    nodes = np.asarray([older_coordinate, -1.0, 0.0])
    if np.min(np.abs(np.diff(nodes))) <= 1e-8:
        return linear_predictor, linear_dimension
    weights = np.ones(3)
    for index in range(3):
        for other in range(3):
            if index != other:
                weights[index] *= (
                    (scale - nodes[other]) / (nodes[index] - nodes[other])
                )
    if not np.all(np.isfinite(weights)) or np.max(np.abs(weights)) > 10:
        return linear_predictor, linear_dimension
    phase = np.vdot(initial, older_mode)
    if phase == 0:
        return linear_predictor, linear_dimension
    older_mode *= np.exp(-1j * np.angle(phase))
    quadratic_predictor = (
        weights[0] * older_mode
        + weights[1] * previous_mode
        + weights[2] * initial
    )
    return quadratic_predictor, 8 if k_scan else 2


def _cached_continuation_mode(system, cfg):
    """Return the latest automatic warm-start mode.

    The orbit solver keeps one eigenvector per warm-start topology and uses it
    whenever the caller does not provide an explicit initial vector. Python
    previously exposed the cache only through the optional
    ``eigenInitialVector`` input, making otherwise identical eta scans cold.
    """
    if (cfg.modeSelection != "nearest" or system.warmStartKey is None
            or cfg.eigsInitialVector is not None):
        return None
    history = _ORBIT_CONTINUATION_CACHE.get(system.warmStartKey, [])
    if not history:
        return None
    return np.asarray(history[-1].mode, dtype=complex).copy()


def _record_continuation(system, cfg, parameters, solution):
    if system.warmStartKey is None or cfg.modeSelection != "nearest":
        return
    converged = bool(solution.get("eigsConverged", True))
    # Store every converged Arnoldi mode in the persistent warm-start cache. A
    # hard 1e-9 gate discarded otherwise valid modes
    # returned at the configured 1e-8 tolerance, leaving the next point with
    # no continuation history and forcing a cold ARPACK solve.  Let the
    # eigensolver convergence flag decide validity, while still rejecting
    # non-finite residuals so the predictor cannot be poisoned.
    if (not converged or not np.isfinite(solution.omega)
            or not np.isfinite(solution.eigenResidual)):
        _ORBIT_CONTINUATION_CACHE.pop(system.warmStartKey, None)
        return
    mode = _unit_mode(solution.modeReduced)
    if mode is None:
        return
    entry = Struct(
        parameters=np.asarray(parameters, dtype=float).copy(),
        mode=mode,
        omega=complex(solution.omega),
    )
    history = list(_ORBIT_CONTINUATION_CACHE.get(system.warmStartKey, []))
    extends_history = False
    if cfg.eigsInitialVector is not None and history:
        initial = _unit_mode(cfg.eigsInitialVector)
        if initial is not None:
            matching = [
                index for index, item in enumerate(history)
                if initial.size == np.asarray(item.mode).size
                and abs(np.vdot(initial, _unit_mode(item.mode))) >= 1 - 1e-10
            ]
            if matching:
                history = history[:matching[-1] + 1]
                extends_history = True
    if not extends_history:
        history = []
    if history and np.array_equal(history[-1].parameters, entry.parameters):
        history[-1] = entry
    else:
        history.append(entry)
    _ORBIT_CONTINUATION_CACHE[system.warmStartKey] = history[-16:]


def _derivative_product(derivative, values):
    if derivative.ndim == 2:
        return derivative @ values
    return np.einsum("ijb,jb->ib", derivative, values, optimize=True)


def _as_pages(value, reference):
    value = np.asarray(value)
    if value.ndim == 0:
        return value
    if value.ndim == 1 and reference.ndim == 2 and value.size == reference.shape[1]:
        return value[None, None, :]
    if value.ndim == 1:
        return value[:, None, None]
    return value[:, None, :]


def _page_matmul(left, right):
    """Multiply matching matrix pages through batched BLAS."""
    return np.moveaxis(
        np.matmul(np.moveaxis(left, 2, 0), np.moveaxis(right, 2, 0)),
        0, 2,
    )


def _sum_page_matmul(left, right):
    """Sum matching page products as one two-dimensional BLAS product."""
    rows, inner, pages = left.shape
    if right.shape[0] != inner or right.shape[2] != pages:
        raise ValueError("page-product dimensions are incompatible")
    columns = right.shape[1]
    left_flat = left.reshape(rows, inner * pages)
    right_flat = right.transpose(0, 2, 1).reshape(inner * pages, columns)
    return left_flat @ right_flat


def build_kinetic_orbit_coefficients(cfg, species, geometry, parallel_velocity,
                                     perpendicular_speed, energy, stream_coefficient,
                                     moment_weight, coordinate_derivative) -> Struct:
    field = geometry.fieldStrength
    gyro_squared = getattr(geometry, "gyroWaveNumberSquared",
                           geometry.kPerpendicularSquared)
    gyro_derivative = getattr(geometry, "gyroWaveNumberDerivative",
                              geometry.kPerpendicularDerivative)
    k_perpendicular = np.sqrt(gyro_squared)
    # Use the geometry-independent k_perpendicular representation.  For
    # circular s-alpha this reduces algebraically to the legacy metric-k/ B
    # expression, while it also matches the Miller/GACODE convention.
    gyro_argument = species.gyroradiusRatio * k_perpendicular * perpendicular_speed
    jzero = j0(gyro_argument)
    gyro_derivative = gyro_argument * (
        gyro_derivative / k_perpendicular
        + 0.5 * geometry.logFieldDerivative
    )
    jzero_derivative = -j1(gyro_argument) * gyro_derivative
    drift_energy = parallel_velocity**2 + perpendicular_speed**2 / 2
    drift = species.driftFrequencyScale * (
        geometry.driftCurvature * drift_energy
        + geometry.driftParallelCorrection * parallel_velocity**2
    )
    gradient = species.gradientFrequency * (1 + species.temperatureGradientRatio * (energy - 1.5))
    maxwellian = species.maxwellianScale * np.exp(-energy) / sqrt((2 * pi) ** 3)
    parallel_factor = -1j * species.streamingScale / cfg.q
    stream_coefficient = stream_coefficient * geometry.parallelGradient
    factors = [jzero]
    derivatives = [jzero_derivative]
    if cfg.aparallel:
        a_factor = -species.thermalSpeedRatio * parallel_velocity * jzero
        factors.append(a_factor)
        derivatives.append(_derivative_product(coordinate_derivative, a_factor))
    if cfg.bparallel:
        b_factor = (0.5 * perpendicular_speed**2
                    * two_j1_over_argument(gyro_argument)
                    / field / species.fieldDriveScale)
        factors.append(b_factor)
        derivatives.append(_derivative_product(coordinate_derivative, b_factor))
    potential_factors = np.stack(factors, axis=1)
    potential_derivatives = np.stack(derivatives, axis=1)
    stream_pages = _as_pages(stream_coefficient, parallel_velocity)
    maxwell_pages = _as_pages(maxwellian, parallel_velocity)
    drive_pages = _as_pages(-(gradient - drift) * maxwellian, parallel_velocity)
    temperature_pages = _as_pages(-species.gradientFrequency * (energy - 1.5) * maxwellian,
                                  parallel_velocity)
    field_derivative = (species.fieldDriveScale * parallel_factor
                        * _as_pages(stream_coefficient * maxwellian, parallel_velocity)
                        * potential_factors)
    field_local = species.fieldDriveScale * (
        drive_pages * potential_factors
        + parallel_factor * stream_pages * maxwell_pages * potential_derivatives
    )
    moment = species.momentScale * _as_pages(moment_weight, parallel_velocity) * potential_factors
    return Struct(
        stream=parallel_factor * stream_coefficient, kineticLocal=drift,
        fieldDerivative=field_derivative, fieldLocal=field_local,
        fieldLocalTemperatureGradient=species.fieldDriveScale * temperature_pages * potential_factors,
        fieldEquilibriumResponse=species.fieldDriveScale * maxwell_pages * potential_factors,
        moment=moment,
    )


def _passing_template(cfg, theta_ops, boundary, orbits, layout):
    n, nf, nb = boundary.numPassingTheta, boundary.numFieldTheta, layout.numPassingBlocks
    orbit_indices = layout.orbitIndex[:nb]
    parallel = np.zeros((n, nb))
    perpendicular = np.zeros((n, nb))
    weights = np.zeros((n, nb))
    theta = np.zeros((n, nb))
    derivative = np.zeros((n, n, nb))
    field_derivative = np.zeros((n, nf, nb))
    field_value = np.zeros((n, nf, nb))
    moment_map = np.zeros((nf, n, nb))
    for page in range(nb):
        sign = layout.velocitySign[page]
        branch = boundary.passing[1 if sign > 0 else 0]
        active = branch.active
        orbit = orbit_indices[page]
        parallel[:, page] = sign * orbits.parallelSpeed[orbit, active]
        perpendicular[:, page] = orbits.perpendicularSpeed[orbit, active]
        weights[:, page] = orbits.integrationWeightPerSign[orbit, active]
        theta[:, page] = theta_ops.theta[active]
        derivative[:, :, page] = branch.derivative
        field_derivative[:, :, page] = branch.fieldDerivative
        field_value[:, :, page] = branch.fieldValue
        moment_map[:, :, page] = branch.momentMap
    return Struct(
        numPassing=n, numField=nf, numBlocks=nb, parallelVelocity=parallel,
        perpendicularSpeed=perpendicular, integrationWeight=weights,
        energy=orbits.energy[orbit_indices], theta=theta, derivative=derivative,
        fieldDerivative=field_derivative, fieldValue=field_value,
        momentMap=moment_map, identity=np.eye(n),
    )


def _passing_sponge(cfg, theta, stream):
    if cfg.boundarySpongeStrength == 0 or cfg.parallelBoundary == "periodic":
        return np.zeros_like(theta)
    center = 0.5 * (cfg.thmin + cfg.thmax)
    base_half = 0.5 * (cfg.thmax - cfg.thmin)
    layer_start = base_half * (1 - cfg.boundarySpongeFraction)
    physical_half = np.max(np.abs(theta - center))
    layer_width = physical_half - layer_start
    ramp = np.maximum((np.abs(theta - center) - layer_start) / layer_width, 0)
    return 5 * cfg.boundarySpongeStrength / layer_width * np.abs(stream) * ramp**4


def _build_passing_group(cfg, species, theta_ops, boundary, orbits, layout, offset,
                         template=None):
    if template is None:
        template = _passing_template(cfg, theta_ops, boundary, orbits, layout)
    geometry = evaluate_geometry(template.theta, cfg)
    coefficients = build_kinetic_orbit_coefficients(
        cfg, species, geometry, template.parallelVelocity, template.perpendicularSpeed,
        template.energy, template.parallelVelocity, template.integrationWeight,
        template.derivative,
    )
    n, nf, nb = template.numPassing, template.numField, template.numBlocks
    sponge = _passing_sponge(cfg, template.theta, coefficients.stream)
    kinetic = template.derivative * coefficients.stream[:, None, :]
    diagonal = np.arange(n)
    kinetic[diagonal, diagonal, :] += coefficients.kineticLocal - 1j * sponge
    ncomponents, total_fields = cfg.fields.count, nf * cfg.fields.count
    coupling = np.zeros((n, total_fields, nb), dtype=complex)
    coupling_t = np.zeros_like(coupling)
    moment = np.zeros((total_fields, n, nb), dtype=complex)
    field_response = np.zeros((total_fields, total_fields), dtype=complex)
    for component in range(ncomponents):
        indices = slice(component * nf, (component + 1) * nf)
        coupling[:, indices, :] = (
            template.fieldDerivative * coefficients.fieldDerivative[:, component, :][:, None, :]
            + template.fieldValue * coefficients.fieldLocal[:, component, :][:, None, :]
        )
        coupling_t[:, indices, :] = template.fieldValue * coefficients.fieldLocalTemperatureGradient[:, component, :][:, None, :]
        moment[indices, :, :] = template.momentMap * coefficients.moment[:, component, :][None, :, :]
    for row in range(ncomponents):
        row_indices = slice(row * nf, (row + 1) * nf)
        for column in range(ncomponents):
            column_indices = slice(column * nf, (column + 1) * nf)
            response_map = (
                template.fieldValue
                * coefficients.fieldEquilibriumResponse[:, column, :][:, None, :]
            )
            field_response[row_indices, column_indices] = _sum_page_matmul(
                moment[row_indices, :, :], response_map,
            )
    coupling -= species.temperatureGradientRatio * coupling_t
    permutation = np.r_[0, np.arange(n - 1, 0, -1)] if cfg.parallelBoundary == "periodic" else np.arange(n - 1, -1, -1)
    return Struct(
        kinetic=kinetic, fieldCouplingBase=coupling,
        fieldCouplingTemperatureGradient=coupling_t, momentCoupling=moment,
        kineticDerivativeGroups=np.stack(
            (boundary.passing[0].derivative, boundary.passing[1].derivative)
        ),
        kineticDerivativeGroupEdges=np.array([0, nb // 2, nb]),
        kineticStream=coefficients.stream,
        kineticLocal=coefficients.kineticLocal - 1j * sponge,
        fieldResponse=field_response, temperatureGradientRatio=species.temperatureGradientRatio,
        blockSize=n, numBlocks=nb, flatIndices=np.arange(offset, offset + n * nb),
        reflectionHalfFactorization=(cfg.geometryModel != "stellarator"
                                     and abs(cfg.tk) <= 10 * np.finfo(float).eps
                                     and abs(cfg.thmin + cfg.thmax) <= 100 * np.finfo(float).eps
                                     * max(1, abs(cfg.thmin), abs(cfg.thmax))),
        reflectionPermutation=permutation,
    )


def _rescale_path(template, orbits, orbit_index):
    scale = sqrt(orbits.energy[orbit_index] / template.energy)
    weight_scale = orbits.referenceIntegrationWeight[orbit_index] / template.referenceIntegrationWeight
    result = template.deepcopy()
    result.orbitIndex = orbit_index
    result.energy = orbits.energy[orbit_index]
    result.parallelVelocity = scale * template.parallelVelocity
    result.perpendicularSpeed = scale * template.perpendicularSpeed
    result.streamCoefficient = scale * template.streamCoefficient
    result.phaseSpaceThetaWeight = weight_scale * template.phaseSpaceThetaWeight
    result.fieldDeposition = weight_scale * template.fieldDeposition
    result.referenceIntegrationWeight = orbits.referenceIntegrationWeight[orbit_index]
    return result


def _trapped_template(cfg, theta_ops, orbits, layout, num_bounce):
    boundary = build_orbit_boundary(cfg, theta_ops)
    nf, nb = boundary.numFieldTheta, layout.numTrappedBlocks
    interpolation = np.zeros((num_bounce, nf, nb))
    orbit_derivative = np.zeros_like(interpolation)
    deposition = np.zeros((nf, num_bounce, nb))
    parallel = np.zeros((num_bounce, nb))
    perpendicular = np.zeros((num_bounce, nb))
    parallel_coefficient = np.zeros((num_bounce, nb))
    energy = np.zeros(nb)
    theta_unwrapped = np.zeros((num_bounce, nb))
    cache = {}
    derivative = np.zeros((num_bounce, num_bounce))
    for page in range(nb):
        block = layout.numPassingBlocks + page
        orbit_index = layout.orbitIndex[block]
        pitch_index = orbit_index // cfg.numEnergy
        center = layout.wellCenter[block]
        well_index = int(np.flatnonzero(layout.wellCenters == center)[0])
        key = (pitch_index, well_index)
        if key not in cache:
            path = build_trapped_path(cfg, orbits, orbit_index, center, num_bounce, theta_ops)
            cache[key] = path
        else:
            path = _rescale_path(cache[key], orbits, orbit_index)
        interpolation[:, :, page] = path.fieldInterpolation
        orbit_derivative[:, :, page] = path.orbitDerivativeInterpolation
        deposition[:, :, page] = path.fieldDeposition
        parallel_coefficient[:, page] = path.streamCoefficient
        parallel[:, page] = path.parallelVelocity
        perpendicular[:, page] = path.perpendicularSpeed
        energy[page] = path.energy
        theta_unwrapped[:, page] = path.thetaUnwrapped
        derivative = path.derivative
    return Struct(
        fieldInterpolation=interpolation, orbitDerivativeInterpolation=orbit_derivative,
        deposition=deposition, parallelCoefficient=parallel_coefficient,
        parallelVelocity=parallel, perpendicularSpeed=perpendicular,
        energy=energy, thetaUnwrapped=theta_unwrapped, derivative=derivative,
        identity=np.eye(num_bounce),
    )


def _build_trapped_group(cfg, species, theta_ops, orbits, layout, num_bounce,
                         offset, template=None):
    if template is None:
        template = _trapped_template(cfg, theta_ops, orbits, layout, num_bounce)
    geometry = evaluate_geometry(template.thetaUnwrapped, cfg)
    coefficients = build_kinetic_orbit_coefficients(
        cfg, species, geometry, template.parallelVelocity, template.perpendicularSpeed,
        template.energy, template.parallelCoefficient, 1, template.derivative,
    )
    nb = layout.numTrappedBlocks
    kinetic = template.derivative[:, :, None] * coefficients.stream[:, None, :]
    # Add the local diagonal directly instead of materializing diagonal pages.
    diagonal = np.arange(num_bounce)
    kinetic[diagonal, diagonal, :] += coefficients.kineticLocal
    nf, nc = template.fieldInterpolation.shape[1], cfg.fields.count
    total_fields = nf * nc
    coupling = np.zeros((num_bounce, total_fields, nb), dtype=complex)
    coupling_t = np.zeros_like(coupling)
    moment = np.zeros((total_fields, num_bounce, nb), dtype=complex)
    field_response = np.zeros((total_fields, total_fields), dtype=complex)
    for component in range(nc):
        indices = slice(component * nf, (component + 1) * nf)
        coupling[:, indices, :] = (
            template.orbitDerivativeInterpolation * coefficients.fieldDerivative[:, component, :][:, None, :]
            + template.fieldInterpolation * coefficients.fieldLocal[:, component, :][:, None, :]
        )
        coupling_t[:, indices, :] = template.fieldInterpolation * coefficients.fieldLocalTemperatureGradient[:, component, :][:, None, :]
        # Right multiplication by diag(d) is exactly column scaling:
        # A @ diag(d) = A * d.
        moment[indices, :, :] = (
            template.deposition
            * coefficients.moment[:, component, :][None, :, :]
        )
    if nb:
        for row in range(nc):
            row_indices = slice(row * nf, (row + 1) * nf)
            for column in range(nc):
                column_indices = slice(column * nf, (column + 1) * nf)
                response_map = (
                    template.fieldInterpolation
                    * coefficients.fieldEquilibriumResponse[:, column, :][:, None, :]
                )
                field_response[row_indices, column_indices] = _sum_page_matmul(
                    moment[row_indices, :, :], response_map,
                )
    coupling -= species.temperatureGradientRatio * coupling_t
    return Struct(
        kinetic=kinetic, fieldCouplingBase=coupling,
        fieldCouplingTemperatureGradient=coupling_t, momentCoupling=moment,
        kineticDerivativeGroups=template.derivative[None, :, :],
        kineticDerivativeGroupEdges=np.array([0, nb]),
        kineticStream=coefficients.stream,
        kineticLocal=coefficients.kineticLocal,
        fieldResponse=field_response, temperatureGradientRatio=species.temperatureGradientRatio,
        blockSize=num_bounce, numBlocks=nb,
        flatIndices=np.arange(offset, offset + num_bounce * nb),
    )


def _field_base(cfg, theta_ops, boundary):
    geometry = evaluate_geometry(theta_ops.theta, cfg)
    polarization = sum(item.polarizationScale for item in cfg.species.items)
    projection = ((boundary.fieldProlongation.T * theta_ops.massDiag)
                  / boundary.fieldWeights[:, None])
    num_fields = cfg.fields.count
    local = np.zeros((theta_ops.nth, num_fields, num_fields), dtype=complex)
    phi = cfg.fields.names.index("phi")
    local[:, phi, phi] = -polarization
    if "aparallel" in cfg.fields.names:
        aparallel = cfg.fields.names.index("aparallel")
        local[:, aparallel, aparallel] = (
            2 * geometry.kPerpendicularSquared / cfg.betaElectron
        )
    if "bparallel" in cfg.fields.names:
        bparallel = cfg.fields.names.index("bparallel")
        local[:, bparallel, bparallel] = 2 / cfg.betaElectron
    blocks = []
    for row in range(num_fields):
        blocks.append([
            projection @ np.diag(local[:, row, column])
            @ boundary.fieldProlongation
            for column in range(num_fields)
        ])
    return np.block(blocks).astype(complex)


def _orbit_topology(cfg, num_bounce):
    """Return the k-independent orbit geometry and interpolation templates."""
    global _ORBIT_TOPOLOGY_CACHE
    topology_key = (
        cfg.thmin, cfg.thmax, cfg.nth,
        cfg.energyMax, cfg.numEnergy, cfg.numPitch, num_bounce,
        cfg.epsilon, cfg.mirrorConvention, cfg.parallelBoundary,
        cfg.boundaryCoordinateStretch, cfg.boundarySpongeFraction,
        cfg.thetaMapAlpha, cfg.boundaryCoreMin, cfg.boundaryCoreMax,
        cfg.boundaryCoreNumTheta, cfg.boundaryTailPeriods,
        cfg.boundaryTailPoints,
        cfg.geometryProfileKey,
    )
    cache_hit = (
        _ORBIT_TOPOLOGY_CACHE is not None
        and _ORBIT_TOPOLOGY_CACHE[0] == topology_key
    )
    if cache_hit:
        return _ORBIT_TOPOLOGY_CACHE[1]
    theta_ops = build_theta_operators(cfg)
    boundary = build_orbit_boundary(cfg, theta_ops)
    orbits = build_orbit_kinematics(cfg, theta_ops.theta)
    layout = build_orbit_layout(cfg, orbits)
    topology = Struct(
        thetaOps=theta_ops,
        boundary=boundary,
        orbits=orbits,
        layout=layout,
        passingTemplate=_passing_template(
            cfg, theta_ops, boundary, orbits, layout
        ),
        trappedTemplate=_trapped_template(
            cfg, theta_ops, orbits, layout, num_bounce
        ),
    )
    _ORBIT_TOPOLOGY_CACHE = (topology_key, topology)
    return topology


def assemble_orbit_grouped_system(cfg, num_bounce, kinetic_species):
    global _ORBIT_ASSEMBLY_CACHE
    assembly_key = freeze(cfg, TRANSIENT_FIELDS | {"etai", "temperatureGradientLength", "temperatureGradientRatio"})
    if _ORBIT_ASSEMBLY_CACHE is not None and _ORBIT_ASSEMBLY_CACHE[0] == assembly_key:
        system, metadata = _ORBIT_ASSEMBLY_CACHE[1], _ORBIT_ASSEMBLY_CACHE[2]
        ratios = [species.temperatureGradientRatio for species in kinetic_species]
        for index, group in enumerate(system.groups):
            group.temperatureGradientRatio = ratios[index // 2]
        system.factorizationKey, system.warmStartKey = _orbit_solver_cache_keys(
            cfg, num_bounce)
        return system, metadata
    # A mismatched full-system cache can never be reused for this k. Drop it
    # before allocating the replacement so both large page sets do not
    # coexist during assembly.
    _ORBIT_ASSEMBLY_CACHE = None
    groups, metadata = [], []
    topology = _orbit_topology(cfg, num_bounce)
    theta_ops = topology.thetaOps
    boundary = topology.boundary
    orbits = topology.orbits
    layout = topology.layout
    passing_size = (
        topology.passingTemplate.numPassing * layout.numPassingBlocks
    )
    trapped_size = num_bounce * layout.numTrappedBlocks
    species_size = passing_size + trapped_size

    def build_species_groups(item):
        species_index, species = item
        kinetic_offset = species_index * species_size
        passing = _build_passing_group(
            cfg, species, theta_ops, boundary, orbits, layout, kinetic_offset,
            template=topology.passingTemplate,
        )
        kinetic_offset += passing.blockSize * passing.numBlocks
        trapped = _build_trapped_group(
            cfg, species, theta_ops, orbits, layout, num_bounce,
            kinetic_offset, template=topology.trappedTemplate,
        )
        species_metadata = Struct(
            theta=boundary.fieldTheta, thetaWeights=boundary.fieldWeights,
            thetaFull=theta_ops.theta, fieldProlongation=boundary.fieldProlongation,
            thetaDerivative=boundary.passing[1].derivative, orbits=orbits,
            layout=layout, numBouncePoints=num_bounce,
            blockOffset=species_index * layout.numBlocks,
        )
        return passing, trapped, species_metadata

    indexed_species = list(enumerate(kinetic_species))
    if len(indexed_species) > 1:
        with ThreadPoolExecutor(max_workers=len(indexed_species)) as executor:
            built_species = list(executor.map(
                build_species_groups, indexed_species
            ))
    else:
        built_species = [build_species_groups(indexed_species[0])]
    for passing, trapped, species_metadata in built_species:
        groups.extend((passing, trapped))
        metadata.append(species_metadata)
    field = _field_base(cfg, theta_ops, boundary)
    for group in groups:
        field += group.fieldResponse
        del group["fieldResponse"]
    factor_key, warm_key = _orbit_solver_cache_keys(cfg, num_bounce)
    system = Struct(groups=groups, field=field, factorizationKey=factor_key, warmStartKey=warm_key)
    if cfg.parallelBoundary == "open-dtn":
        num_field = boundary.numFieldTheta
        core = np.concatenate([
            boundary.coreFieldIndices + component * num_field
            for component in range(cfg.fields.count)
        ])
        mask = np.ones(cfg.fields.count * num_field, dtype=bool)
        mask[core] = False
        system.fieldSchurPartition = Struct(core=core, tail=np.flatnonzero(mask))
    _ORBIT_ASSEMBLY_CACHE = (assembly_key, system, metadata)
    return system, metadata


class GroupedOrbitShiftSolver:
    def __init__(self, system, sigma, precision="double",
                 cpu_factorization_workers=1, cpu_operator_workers=1):
        global _ORBIT_FACTORIZATION_CACHE
        self.system, self.sigma = system, sigma
        self.groups = []
        self.nf = system.field.shape[0]
        self.orbit_sizes = np.concatenate([
            np.full(group.numBlocks, group.blockSize, dtype=int) for group in system.groups
        ])
        self.nd = int(np.sum(self.orbit_sizes))
        self.num_unknowns = self.nd + self.nf
        dtype = np.complex64 if precision == "single" else np.complex128
        self.field = np.asarray(system.field, dtype=dtype)
        schur = self.field.copy()
        factorized = 0
        cache_key = ((system.factorizationKey, complex(sigma), precision)
                     if system.factorizationKey is not None else None)
        cache_hit = (cache_key is not None and _ORBIT_FACTORIZATION_CACHE is not None
                     and _ORBIT_FACTORIZATION_CACHE[0] == cache_key)
        cached_groups = _ORBIT_FACTORIZATION_CACHE[1] if cache_hit else []
        if not cache_hit:
            # Release a stale k-specific factorization before materializing
            # the next several-hundred-megabyte set of inverse/response pages.
            _ORBIT_FACTORIZATION_CACHE = None
        new_cache = []
        total_factor_workers = max(1, int(cpu_factorization_workers))

        def factor_group(source):
            group_count = max(1, len(system.groups))
            inner_workers = max(1, total_factor_workers // group_count)
            group_pool = _PageWorkerPool(inner_workers)
            try:
                ratio = source.temperatureGradientRatio
                moment = np.asarray(source.momentCoupling, dtype=dtype)
                base = np.asarray(source.fieldCouplingBase, dtype=dtype)
                temperature = np.asarray(
                    source.fieldCouplingTemperatureGradient, dtype=dtype
                )
                shifted_pages = np.ascontiguousarray(
                    np.moveaxis(
                        np.asarray(source.kinetic, dtype=dtype), 2, 0
                    )
                    - dtype(sigma)
                    * np.eye(source.blockSize, dtype=dtype)[None, :, :]
                )
                use_reflection = (
                    source.get("reflectionHalfFactorization", False)
                    and source.numBlocks % 2 == 0
                )
                if source.numBlocks and use_reflection:
                    num_independent = source.numBlocks // 2
                    independent_inverse = group_pool.inverse(
                        shifted_pages[:num_independent]
                    )
                    permutation = np.asarray(source.reflectionPermutation)
                    inverse_pages = np.empty_like(shifted_pages)
                    inverse_pages[:num_independent] = independent_inverse
                    inverse_pages[num_independent:] = independent_inverse[
                        :, permutation, :
                    ][:, :, permutation]
                    factorized_count = num_independent
                elif source.numBlocks:
                    inverse_pages = group_pool.inverse(shifted_pages)
                    factorized_count = source.numBlocks
                else:
                    inverse_pages = shifted_pages.copy()
                    factorized_count = 0
                # The current physical coupling is affine in the
                # temperature-gradient ratio. Apply the inverse once to the
                # combined coupling; retain the affine pieces for an exact
                # ratio-only cache update.
                combined = base + dtype(ratio) * temperature
                response_reference_pages = group_pool.matmul(
                    inverse_pages, np.moveaxis(combined, 2, 0)
                )
                if source.numBlocks:
                    schur_reference = group_pool.sum_matmul(
                        np.moveaxis(moment, 2, 0),
                        response_reference_pages,
                    )
                else:
                    schur_reference = np.zeros_like(schur)
                cached = Struct(
                    inversePages=inverse_pages,
                    responseReferencePages=response_reference_pages,
                    responseTemperaturePages=None,
                    schurReference=schur_reference,
                    schurTemperature=None,
                    referenceRatio=ratio,
                    moment=moment,
                    momentFlat=moment.reshape(self.nf, -1),
                    fieldCouplingBase=base,
                    fieldCouplingTemperature=temperature,
                    kineticDerivativeGroups=np.asarray(
                        source.kineticDerivativeGroups, dtype=dtype
                    ),
                    kineticDerivativeGroupEdges=np.asarray(
                        source.kineticDerivativeGroupEdges, dtype=int
                    ),
                    kineticStream=np.asarray(
                        source.kineticStream, dtype=dtype
                    ),
                    kineticLocal=np.asarray(
                        source.kineticLocal, dtype=dtype
                    ),
                )
                return cached, factorized_count
            finally:
                group_pool.close()

        if not cache_hit:
            outer_workers = min(len(system.groups), total_factor_workers)
            if outer_workers > 1:
                with ThreadPoolExecutor(max_workers=outer_workers) as executor:
                    factored = list(executor.map(
                        factor_group, system.groups
                    ))
            else:
                factored = [factor_group(source) for source in system.groups]
            cached_groups = [item[0] for item in factored]
            factorized = sum(item[1] for item in factored)
            new_cache = cached_groups
        page_pool = _PageWorkerPool(cpu_factorization_workers)
        try:
            for group_index, source in enumerate(system.groups):
                ratio = source.temperatureGradientRatio
                cached = cached_groups[group_index]
                ratio_delta = ratio - cached.referenceRatio
                if ratio_delta == 0:
                    response_pages = cached.responseReferencePages
                    schur_response = cached.schurReference
                else:
                    if cached.responseTemperaturePages is None:
                        cached.responseTemperaturePages = page_pool.matmul(
                            cached.inversePages,
                            np.moveaxis(
                                cached.fieldCouplingTemperature, 2, 0
                            ),
                        )
                        if source.numBlocks:
                            cached.schurTemperature = page_pool.sum_matmul(
                                np.moveaxis(cached.moment, 2, 0),
                                cached.responseTemperaturePages,
                            )
                        else:
                            cached.schurTemperature = np.zeros_like(schur)
                    response_pages = (
                        cached.responseReferencePages
                        + dtype(ratio_delta)
                        * cached.responseTemperaturePages
                    )
                    schur_response = (
                        cached.schurReference
                        + dtype(ratio_delta) * cached.schurTemperature
                    )
                schur -= schur_response
                group = Struct(
                    blockSize=source.blockSize, numBlocks=source.numBlocks,
                    flatIndices=source.flatIndices, temperatureGradientRatio=ratio,
                    inversePages=cached.inversePages,
                    responsePages=response_pages,
                    momentCoupling=cached.moment,
                    momentFlat=cached.momentFlat,
                    fieldCouplingBase=cached.fieldCouplingBase,
                    fieldCouplingTemperature=cached.fieldCouplingTemperature,
                    kineticDerivativeGroups=cached.kineticDerivativeGroups,
                    kineticDerivativeGroupEdges=(
                        cached.kineticDerivativeGroupEdges
                    ),
                    kineticStream=cached.kineticStream,
                    kineticLocal=cached.kineticLocal,
                )
                self.groups.append(group)
        finally:
            page_pool.close()
        self.operator_pool = _PageWorkerPool(cpu_operator_workers, shared=True)
        group_workers = min(len(self.groups), max(1, cpu_operator_workers))
        if group_workers > 1:
            if group_workers not in _SHARED_GROUP_EXECUTORS:
                _SHARED_GROUP_EXECUTORS[group_workers] = ThreadPoolExecutor(
                    max_workers=group_workers
                )
            self.group_executor = _SHARED_GROUP_EXECUTORS[group_workers]
        else:
            self.group_executor = None
        if cache_key is not None and not cache_hit:
            _ORBIT_FACTORIZATION_CACHE = (cache_key, new_cache)
        self.schur = schur
        partition = system.get("fieldSchurPartition")
        if partition is not None and len(partition.tail):
            self.dtn_core = np.asarray(partition.core, dtype=int)
            self.dtn_tail = np.asarray(partition.tail, dtype=int)
            tail_block = schur[np.ix_(self.dtn_tail, self.dtn_tail)]
            self.tail_lu = lu_factor(tail_block, check_finite=False)
            self.tail_to_core = schur[np.ix_(self.dtn_core, self.dtn_tail)]
            self.core_to_tail = schur[np.ix_(self.dtn_tail, self.dtn_core)]
            tail_inverse_core = lu_solve(
                self.tail_lu, self.core_to_tail, check_finite=False
            )
            self.dtn_operator = (
                schur[np.ix_(self.dtn_core, self.dtn_core)]
                - self.tail_to_core @ tail_inverse_core
            )
            self.dtn_lu = lu_factor(self.dtn_operator, check_finite=False)
            self.schur_lu = None
        else:
            self.dtn_core = self.dtn_tail = None
            self.schur_lu = lu_factor(schur, check_finite=False)
        self.info = Struct(
            numBlocks=len(self.orbit_sizes), orbitSizes=self.orbit_sizes,
            groupSizes=np.array([group.blockSize for group in system.groups]),
            numFields=self.nf, numUnknowns=self.num_unknowns,
            field=np.asarray(self.field, dtype=complex), schur=np.asarray(schur, dtype=complex), usedGpu=False,
            kineticCacheHit=cache_hit, numFactorizedPages=factorized,
            cpuFactorizationWorkers=int(cpu_factorization_workers),
            cpuOperatorWorkers=int(cpu_operator_workers),
            dtn=Struct(
                enabled=self.dtn_core is not None,
                numCoreFields=(self.nf if self.dtn_core is None else len(self.dtn_core)),
                numTailFields=(0 if self.dtn_tail is None else len(self.dtn_tail)),
                coreIndices=self.dtn_core, tailIndices=self.dtn_tail,
                operator=(None if self.dtn_core is None else np.asarray(self.dtn_operator, dtype=complex)),
            ),
        )

    @staticmethod
    def _matrix(value):
        value = np.asarray(value)
        return (value[:, None], True) if value.ndim == 1 else (value, False)

    def solve(self, rhs):
        rhs, vector = self._matrix(rhs)
        count = rhs.shape[1]
        field_rhs = rhs[self.nd:, :].astype(complex, copy=True)

        def solve_free(group):
            selected_pages = rhs[group.flatIndices, :].reshape(
                (group.numBlocks, group.blockSize, count)
            )
            free = np.matmul(
                group.inversePages, np.ascontiguousarray(selected_pages)
            )
            free_flat = free.transpose(1, 0, 2).reshape((-1, count))
            return free, group.momentFlat @ free_flat

        if self.group_executor is None:
            free_groups = [solve_free(group) for group in self.groups]
        else:
            free_groups = list(self.group_executor.map(
                solve_free, self.groups
            ))
        for _, moment_free in free_groups:
            field_rhs -= moment_free
        if self.dtn_core is None:
            field_solution = lu_solve(self.schur_lu, field_rhs, check_finite=False)
        else:
            free_tail = lu_solve(
                self.tail_lu, field_rhs[self.dtn_tail, :], check_finite=False
            )
            core_solution = lu_solve(
                self.dtn_lu,
                field_rhs[self.dtn_core, :] - self.tail_to_core @ free_tail,
                check_finite=False,
            )
            tail_solution = free_tail - lu_solve(
                self.tail_lu, self.core_to_tail @ core_solution,
                check_finite=False,
            )
            field_solution = np.zeros_like(field_rhs)
            field_solution[self.dtn_core, :] = core_solution
            field_solution[self.dtn_tail, :] = tail_solution
        result = np.zeros((self.num_unknowns, count), dtype=complex)

        def finish_group(item):
            group, (free, _) = item
            orbit = free - np.matmul(group.responsePages, field_solution)
            return group, orbit

        group_data = zip(self.groups, free_groups)
        if self.group_executor is None:
            finished_groups = map(finish_group, group_data)
        else:
            finished_groups = self.group_executor.map(
                finish_group, group_data
            )
        for group, orbit in finished_groups:
            result[group.flatIndices, :] = orbit.reshape((-1, count))
        result[self.nd:, :] = field_solution
        return result[:, 0] if vector else result

    def apply_shifted(self, value):
        value, vector = self._matrix(value)
        count = value.shape[1]
        field_vector = value[self.nd:, :]
        field_result = self.field @ field_vector
        result = np.zeros((self.num_unknowns, count), dtype=complex)
        for group in self.groups:
            orbit_pages = value[group.flatIndices, :].reshape(
                (group.numBlocks, group.blockSize, count)
            )
            orbit = orbit_pages.transpose(1, 0, 2)
            differentiated = np.empty_like(orbit)
            for derivative, start, stop in zip(
                    group.kineticDerivativeGroups,
                    group.kineticDerivativeGroupEdges[:-1],
                    group.kineticDerivativeGroupEdges[1:]):
                columns = orbit[:, start:stop, :].reshape(
                    (group.blockSize, -1)
                )
                differentiated[:, start:stop, :] = (
                    derivative @ columns
                ).reshape((group.blockSize, stop - start, count))
            pages = (
                group.kineticStream[:, :, None] * differentiated
                + (group.kineticLocal[:, :, None] - self.sigma) * orbit
            )
            base_pages, temperature_pages = (
                self.operator_pool.contract_shared_vector_pair(
                    group.fieldCouplingBase,
                    group.fieldCouplingTemperature,
                    field_vector,
                )
            )
            pages += base_pages
            pages += group.temperatureGradientRatio * temperature_pages
            orbit_flat = orbit.reshape((-1, count))
            field_result += group.momentFlat @ orbit_flat
            result[group.flatIndices, :] = pages.transpose(1, 0, 2).reshape(
                (-1, count)
            )
        result[self.nd:, :] = field_result
        return result[:, 0] if vector else result


def _finalize(solution, metadata, cfg, info):
    field_start = info.numUnknowns - info.numFields
    ntheta_fields = info.numFields // cfg.fields.count
    reduced = solution.modeReduced[field_start:].reshape((ntheta_fields, cfg.fields.count), order="F")
    if cfg.parallelBoundary == "periodic":
        field_full = np.vstack((reduced, reduced[0, :]))
        boundary = "passing-periodic/trapped-bounce-periodic"
    elif cfg.parallelBoundary == "open":
        field_full = np.zeros((ntheta_fields + 2, cfg.fields.count), dtype=complex)
        field_full[1:-1, :] = reduced
        boundary = "field-decay/passing-inflow/trapped-bounce-periodic"
    elif cfg.parallelBoundary == "open-extrapolated":
        field_full = metadata[0].fieldProlongation @ reduced
        boundary = "field-extrapolated/passing-inflow/trapped-bounce-periodic"
    else:
        field_full = metadata[0].fieldProlongation @ reduced
        boundary = "real-tail-dtn/passing-inflow/trapped-bounce-periodic"
    center = (field_full.shape[0] - 1) // 2
    scale = field_full[center, 0]
    field_full /= scale
    kinetic = solution.modeReduced[:field_start] / scale
    offsets = np.r_[0, np.cumsum(info.orbitSizes)]
    orbit_mode = [kinetic[offsets[i]:offsets[i + 1]] for i in range(len(info.orbitSizes))]
    result = Struct(
        mode=np.r_[kinetic, field_full.reshape(-1, order="F")],
        orbitMode=orbit_mode, phi=field_full[:, 0],
        aparallel=(field_full[:, cfg.fields.names.index("aparallel")]
                   if "aparallel" in cfg.fields.names else None),
        bparallel=(field_full[:, cfg.fields.names.index("bparallel")]
                   if "bparallel" in cfg.fields.names else None),
        theta=metadata[0].thetaFull, energy=np.concatenate([item.orbits.energy for item in metadata]),
        orbitMetadata=metadata, discretization="orbit", thetaBoundary=boundary,
    )
    result["lambda"] = np.concatenate([item.orbits["lambda"] for item in metadata])
    return result


def solve_orbit(cfg, total_start=None):
    if cfg.useGpu:
        from .gpu import solve_orbit_gpu
        return solve_orbit_gpu(cfg, total_start)
    if total_start is None:
        total_start = perf_counter()
    assembly_start = perf_counter()
    kinetic_species = resolve_kinetic_species(cfg)
    parameters = _continuation_parameters(cfg, kinetic_species)
    system, metadata = assemble_orbit_grouped_system(
        cfg, cfg.numBouncePoints, kinetic_species
    )
    assembly_time = perf_counter() - assembly_start
    factory = lambda shift: GroupedOrbitShiftSolver(
        system, shift, cfg.blockPrecision, cfg.cpuFactorizationWorkers,
        cfg.cpuOperatorWorkers,
    )
    # An explicit initial vector always wins; otherwise reuse the latest
    # compatible orbit mode.
    effective_cfg = cfg
    automatic_initial = _cached_continuation_mode(system, cfg)
    if automatic_initial is not None:
        effective_cfg = cfg.deepcopy()
        effective_cfg.eigsInitialVector = automatic_initial
    warm_ritz_vector, warm_ritz_dimension = _warm_ritz_predictor(
        system, effective_cfg, parameters
    )
    solution, shifted, timing, mode_scan = solve_selected(
        effective_cfg, factory, warm_ritz_vector=warm_ritz_vector,
        warm_ritz_dimension=warm_ritz_dimension,
    )
    _record_continuation(system, effective_cfg, parameters, solution)
    result = Struct() if cfg.compactResult else _finalize(solution, metadata, cfg, shifted.info)
    result.update(
        runtime=perf_counter() - total_start, matrix=None, massMatrix=None,
        eigenvalues=solution.omega, omega=solution.omega,
        reducedMode=solution.modeReduced, eigenResidual=solution.eigenResidual,
        fieldConstraintResidual=solution.fieldConstraintResidual, usedGpu=False,
        timing=Struct(assembly=assembly_time, factorization=timing.factorization,
                      probe=timing.probe, eigensolve=timing.eigensolve),
        blockSolverInfo=shifted.info,
    )
    if mode_scan is not None:
        result.modeScan = mode_scan
    if "eigsInfo" in solution:
        result.eigsInfo = solution.eigsInfo
    result.blockSolverInfo.warmStartUsed = bool(
        solution.get("eigsInfo", {}).get("warmStartUsed", False)
    )
    return result
