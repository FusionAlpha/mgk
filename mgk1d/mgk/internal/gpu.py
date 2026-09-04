"""Optional CuPy page solvers and restarted GPU Arnoldi backend."""

from __future__ import annotations

import os
import tempfile
from time import perf_counter

os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mgk-cupy-cache"))

import cupy as cp
import numpy as np
from scipy.special import j0, j1

from mgk._struct import Struct
from .geometry import evaluate_geometry

_GPU_ORBIT_CACHE = None
_GPU_ORBIT_TEMPLATE_CACHE = None
_GPU_ORBIT_ASSEMBLY_CACHE = None


def _dtype(precision):
    return cp.complex64 if precision == "single" else cp.complex128


def _real_dtype(precision):
    return cp.float32 if precision == "single" else cp.float64


def _return(value, device):
    return value if device else cp.asnumpy(value)


def _device_struct(source, names, dtype):
    return Struct(**{
        name: cp.asarray(source[name], dtype=dtype)
        for name in names
    })


def _gpu_orbit_templates(topology, precision):
    """Keep the k-independent orbit interpolation tensors on the GPU."""
    global _GPU_ORBIT_TEMPLATE_CACHE
    if (_GPU_ORBIT_TEMPLATE_CACHE is not None
            and _GPU_ORBIT_TEMPLATE_CACHE[0] is topology
            and _GPU_ORBIT_TEMPLATE_CACHE[1] == precision):
        return _GPU_ORBIT_TEMPLATE_CACHE[2]
    dtype = _real_dtype(precision)
    passing_names = (
        "parallelVelocity", "perpendicularSpeed", "integrationWeight",
        "energy", "theta", "derivative", "fieldDerivative", "fieldValue",
        "momentMap", "identity",
    )
    trapped_names = (
        "fieldInterpolation", "orbitDerivativeInterpolation", "deposition",
        "parallelCoefficient", "parallelVelocity", "perpendicularSpeed",
        "energy", "thetaUnwrapped", "derivative", "identity",
    )
    templates = Struct(
        passing=_device_struct(topology.passingTemplate, passing_names, dtype),
        trapped=_device_struct(topology.trappedTemplate, trapped_names, dtype),
    )
    for name in ("numPassing", "numField", "numBlocks"):
        templates.passing[name] = topology.passingTemplate[name]
    _GPU_ORBIT_TEMPLATE_CACHE = (topology, precision, templates)
    return templates


def _gpu_geometry(cpu_geometry, precision):
    dtype = _real_dtype(precision)
    return Struct(**{
        name: cp.asarray(value, dtype=dtype)
        for name, value in cpu_geometry.items()
        if isinstance(value, np.ndarray)
    })


def _gpu_kinetic_orbit_coefficients(
        cfg, species, cpu_geometry, geometry, cpu_perpendicular_speed,
        parallel_velocity, perpendicular_speed,
        energy, stream_coefficient, moment_weight, coordinate_derivative,
        precision):
    """Build k-dependent coefficients on device, with CPU Bessel fallback."""
    real_dtype = _real_dtype(precision)
    complex_dtype = _dtype(precision)
    cpu_gyro_squared = getattr(cpu_geometry, "gyroWaveNumberSquared",
                               cpu_geometry.kPerpendicularSquared)
    cpu_gyro_derivative = getattr(cpu_geometry, "gyroWaveNumberDerivative",
                                  cpu_geometry.kPerpendicularDerivative)
    cpu_k_perpendicular = np.sqrt(cpu_gyro_squared)
    cpu_argument = (
        species.gyroradiusRatio * cpu_k_perpendicular
        * cpu_perpendicular_speed
    )
    cpu_gyro_derivative = cpu_argument * (
        cpu_gyro_derivative / cpu_k_perpendicular
        + 0.5 * cpu_geometry.logFieldDerivative
    )
    jzero = cp.asarray(j0(cpu_argument), dtype=real_dtype)
    jzero_derivative = cp.asarray(
        -j1(cpu_argument) * cpu_gyro_derivative, dtype=real_dtype
    )

    stream_coefficient = stream_coefficient * geometry.parallelGradient
    drift_energy = parallel_velocity**2 + perpendicular_speed**2 / real_dtype(2)
    drift = real_dtype(species.driftFrequencyScale) * (
        geometry.driftCurvature * drift_energy
        + geometry.driftParallelCorrection * parallel_velocity**2
    )
    gradient = real_dtype(species.gradientFrequency) * (
        1 + real_dtype(species.temperatureGradientRatio) * (energy - real_dtype(1.5))
    )
    maxwellian = (real_dtype(species.maxwellianScale)
                  * cp.exp(-energy) / real_dtype(np.sqrt((2 * np.pi) ** 3)))
    parallel_factor = complex_dtype(
        -1j * species.streamingScale / cfg.q
    )
    # Keep the same field ordering and gyro-averaged factors as the CPU
    # builder.  This enables the native path for A_parallel/B_parallel while
    # retaining CPU-evaluated Bessel values for Blackwell compatibility.
    factors = [jzero]
    derivatives = [jzero_derivative]
    if cfg.aparallel:
        a_factor = (-real_dtype(species.thermalSpeedRatio)
                    * parallel_velocity * jzero)
        if coordinate_derivative.ndim == 2:
            a_derivative = coordinate_derivative @ a_factor
        else:
            a_derivative = cp.einsum(
                "ijb,jb->ib", coordinate_derivative, a_factor, optimize=True
            )
        factors.append(a_factor)
        derivatives.append(a_derivative)
    if cfg.bparallel:
        from .physics import two_j1_over_argument

        gyro_average = two_j1_over_argument(cpu_argument)
        b_factor = cp.asarray(
            0.5 * cpu_perpendicular_speed**2 * gyro_average
            / cpu_geometry.fieldStrength / species.fieldDriveScale,
            dtype=real_dtype,
        )
        if coordinate_derivative.ndim == 2:
            b_derivative = coordinate_derivative @ b_factor
        else:
            b_derivative = cp.einsum(
                "ijb,jb->ib", coordinate_derivative, b_factor, optimize=True
            )
        factors.append(b_factor)
        derivatives.append(b_derivative)
    potential_factors = cp.stack(factors, axis=1)
    potential_derivatives = cp.stack(derivatives, axis=1)
    stream_pages = stream_coefficient[:, None, :]
    maxwell_pages = maxwellian[None, None, :]
    drive_pages = (
        -(gradient[None, :] - drift) * maxwellian[None, :]
    )[:, None, :]
    temperature_pages = (
        -real_dtype(species.gradientFrequency)
        * (energy - real_dtype(1.5)) * maxwellian
    )[None, None, :]
    moment_pages = (moment_weight[:, None, :]
                    if isinstance(moment_weight, cp.ndarray) else moment_weight)
    field_derivative = (
        real_dtype(species.fieldDriveScale) * parallel_factor
        * (stream_coefficient * maxwellian[None, :])[:, None, :]
        * potential_factors
    )
    field_local = real_dtype(species.fieldDriveScale) * (
        drive_pages * potential_factors
        + parallel_factor * stream_pages * maxwell_pages
        * potential_derivatives
    )
    moment = (real_dtype(species.momentScale) * moment_pages
              * potential_factors)
    return Struct(
        stream=parallel_factor * stream_coefficient,
        kineticLocal=drift,
        fieldDerivative=field_derivative,
        fieldLocal=field_local,
        fieldLocalTemperatureGradient=(
            real_dtype(species.fieldDriveScale)
            * temperature_pages * potential_factors
        ),
        fieldEquilibriumResponse=(
            real_dtype(species.fieldDriveScale)
            * maxwell_pages * potential_factors
        ),
        moment=moment,
    )


def _gpu_passing_sponge(cfg, theta, stream, precision):
    if cfg.boundarySpongeStrength == 0 or cfg.parallelBoundary == "periodic":
        return cp.zeros_like(theta)
    dtype = _real_dtype(precision)
    center = dtype(0.5 * (cfg.thmin + cfg.thmax))
    base_half = dtype(0.5 * (cfg.thmax - cfg.thmin))
    layer_start = base_half * dtype(1 - cfg.boundarySpongeFraction)
    physical_half = cp.max(cp.abs(theta - center))
    layer_width = physical_half - layer_start
    ramp = cp.maximum((cp.abs(theta - center) - layer_start) / layer_width, 0)
    return (dtype(5 * cfg.boundarySpongeStrength) / layer_width
            * cp.abs(stream) * ramp**4)


def _gpu_sum_page_matmul(left, right):
    rows, inner, pages = left.shape
    columns = right.shape[1]
    left_flat = left.reshape(rows, inner * pages)
    right_flat = right.transpose(0, 2, 1).reshape(inner * pages, columns)
    return left_flat @ right_flat


def _build_passing_group_gpu(cfg, species, cpu_template, template, offset, precision):
    cpu_geometry = evaluate_geometry(cpu_template.theta, cfg)
    geometry = _gpu_geometry(cpu_geometry, precision)
    coefficients = _gpu_kinetic_orbit_coefficients(
        cfg, species, cpu_geometry, geometry,
        cpu_template.perpendicularSpeed,
        template.parallelVelocity, template.perpendicularSpeed,
        template.energy, template.parallelVelocity,
        template.integrationWeight, template.derivative, precision,
    )
    n, nf, nb = template.numPassing, template.numField, template.numBlocks
    sponge = _gpu_passing_sponge(cfg, template.theta, coefficients.stream, precision)
    kinetic = template.derivative * coefficients.stream[:, None, :]
    diagonal = cp.arange(n)
    kinetic[diagonal, diagonal, :] += coefficients.kineticLocal - 1j * sponge
    total_fields = nf * cfg.fields.count
    dtype = _dtype(precision)
    coupling = cp.zeros((n, total_fields, nb), dtype=dtype)
    coupling_t = cp.zeros_like(coupling)
    moment = cp.zeros((total_fields, n, nb), dtype=dtype)
    field_response = cp.zeros((total_fields, total_fields), dtype=dtype)
    for component in range(cfg.fields.count):
        indices = slice(component * nf, (component + 1) * nf)
        coupling[:, indices, :] = (
            template.fieldDerivative
            * coefficients.fieldDerivative[:, component, :][:, None, :]
            + template.fieldValue
            * coefficients.fieldLocal[:, component, :][:, None, :]
        )
        coupling_t[:, indices, :] = (
            template.fieldValue
            * coefficients.fieldLocalTemperatureGradient[:, component, :][:, None, :]
        )
        moment[indices, :, :] = (
            template.momentMap
            * coefficients.moment[:, component, :][None, :, :]
        )
    for row in range(cfg.fields.count):
        row_indices = slice(row * nf, (row + 1) * nf)
        for column in range(cfg.fields.count):
            column_indices = slice(column * nf, (column + 1) * nf)
            response_map = (
                template.fieldValue
                * coefficients.fieldEquilibriumResponse[:, column, :][:, None, :]
            )
            field_response[row_indices, column_indices] = _gpu_sum_page_matmul(
                moment[row_indices, :, :], response_map
            )
    ratio = _real_dtype(precision)(species.temperatureGradientRatio)
    coupling -= ratio * coupling_t
    permutation = (np.r_[0, np.arange(n - 1, 0, -1)]
                   if cfg.parallelBoundary == "periodic"
                   else np.arange(n - 1, -1, -1))
    return Struct(
        kinetic=kinetic, fieldCouplingBase=coupling,
        fieldCouplingTemperatureGradient=coupling_t,
        momentCoupling=moment, fieldResponse=field_response,
        temperatureGradientRatio=species.temperatureGradientRatio,
        blockSize=n, numBlocks=nb,
        flatIndices=cp.arange(offset, offset + n * nb),
        reflectionHalfFactorization=(
            cfg.geometryModel != "stellarator"
            and abs(cfg.tk) <= 10 * np.finfo(float).eps
            and abs(cfg.thmin + cfg.thmax) <= 100 * np.finfo(float).eps
            * max(1, abs(cfg.thmin), abs(cfg.thmax))
        ),
        reflectionPermutation=cp.asarray(permutation),
    )


def _build_trapped_group_gpu(
        cfg, species, cpu_template, template, layout, num_bounce,
        offset, precision):
    cpu_geometry = evaluate_geometry(cpu_template.thetaUnwrapped, cfg)
    geometry = _gpu_geometry(cpu_geometry, precision)
    coefficients = _gpu_kinetic_orbit_coefficients(
        cfg, species, cpu_geometry, geometry,
        cpu_template.perpendicularSpeed,
        template.parallelVelocity, template.perpendicularSpeed,
        template.energy, template.parallelCoefficient,
        _real_dtype(precision)(1), template.derivative, precision,
    )
    nb = layout.numTrappedBlocks
    kinetic = template.derivative[:, :, None] * coefficients.stream[:, None, :]
    diagonal = cp.arange(num_bounce)
    kinetic[diagonal, diagonal, :] += coefficients.kineticLocal
    nf, total_fields = cpu_template.fieldInterpolation.shape[1], (
        cpu_template.fieldInterpolation.shape[1] * cfg.fields.count
    )
    dtype = _dtype(precision)
    coupling = cp.zeros((num_bounce, total_fields, nb), dtype=dtype)
    coupling_t = cp.zeros_like(coupling)
    moment = cp.zeros((total_fields, num_bounce, nb), dtype=dtype)
    field_response = cp.zeros((total_fields, total_fields), dtype=dtype)
    for component in range(cfg.fields.count):
        indices = slice(component * nf, (component + 1) * nf)
        coupling[:, indices, :] = (
            template.orbitDerivativeInterpolation
            * coefficients.fieldDerivative[:, component, :][:, None, :]
            + template.fieldInterpolation
            * coefficients.fieldLocal[:, component, :][:, None, :]
        )
        coupling_t[:, indices, :] = (
            template.fieldInterpolation
            * coefficients.fieldLocalTemperatureGradient[:, component, :][:, None, :]
        )
        moment[indices, :, :] = (
            template.deposition
            * coefficients.moment[:, component, :][None, :, :]
        )
    if nb:
        for row in range(cfg.fields.count):
            row_indices = slice(row * nf, (row + 1) * nf)
            for column in range(cfg.fields.count):
                column_indices = slice(column * nf, (column + 1) * nf)
                response_map = (
                    template.fieldInterpolation
                    * coefficients.fieldEquilibriumResponse[:, column, :][:, None, :]
                )
                field_response[row_indices, column_indices] = _gpu_sum_page_matmul(
                    moment[row_indices, :, :], response_map
                )
    ratio = _real_dtype(precision)(species.temperatureGradientRatio)
    coupling -= ratio * coupling_t
    return Struct(
        kinetic=kinetic, fieldCouplingBase=coupling,
        fieldCouplingTemperatureGradient=coupling_t,
        momentCoupling=moment, fieldResponse=field_response,
        temperatureGradientRatio=species.temperatureGradientRatio,
        blockSize=num_bounce, numBlocks=nb,
        flatIndices=cp.arange(offset, offset + num_bounce * nb),
    )


def assemble_orbit_grouped_system_gpu(cfg, num_bounce, kinetic_species):
    """Assemble electrostatic or electromagnetic orbit pages on the GPU."""
    global _GPU_ORBIT_ASSEMBLY_CACHE
    from .cache import TRANSIENT_FIELDS, freeze
    from .orbit import _field_base, _orbit_solver_cache_keys, _orbit_topology

    assembly_key = freeze(
        cfg,
        TRANSIENT_FIELDS | {
            "etai", "temperatureGradientLength", "temperatureGradientRatio",
        },
    )
    if (_GPU_ORBIT_ASSEMBLY_CACHE is not None
            and _GPU_ORBIT_ASSEMBLY_CACHE[0] == assembly_key):
        system, metadata = _GPU_ORBIT_ASSEMBLY_CACHE[1:]
        ratios = [species.temperatureGradientRatio for species in kinetic_species]
        for index, group in enumerate(system.groups):
            group.temperatureGradientRatio = ratios[index // 2]
        system.factorizationKey, system.warmStartKey = _orbit_solver_cache_keys(
            cfg, num_bounce
        )
        return system, metadata
    _GPU_ORBIT_ASSEMBLY_CACHE = None
    topology = _orbit_topology(cfg, num_bounce)
    templates = _gpu_orbit_templates(topology, cfg.blockPrecision)
    boundary, layout = topology.boundary, topology.layout
    passing_size = topology.passingTemplate.numPassing * layout.numPassingBlocks
    trapped_size = num_bounce * layout.numTrappedBlocks
    species_size = passing_size + trapped_size
    groups, metadata = [], []
    for species_index, species in enumerate(kinetic_species):
        offset = species_index * species_size
        passing = _build_passing_group_gpu(
            cfg, species, topology.passingTemplate, templates.passing,
            offset, cfg.blockPrecision,
        )
        offset += passing.blockSize * passing.numBlocks
        trapped = _build_trapped_group_gpu(
            cfg, species, topology.trappedTemplate, templates.trapped,
            layout, num_bounce, offset, cfg.blockPrecision,
        )
        groups.extend((passing, trapped))
        metadata.append(Struct(
            theta=boundary.fieldTheta, thetaWeights=boundary.fieldWeights,
            thetaFull=topology.thetaOps.theta,
            fieldProlongation=boundary.fieldProlongation,
            thetaDerivative=boundary.passing[1].derivative,
            orbits=topology.orbits, layout=layout,
            numBouncePoints=num_bounce,
            blockOffset=species_index * layout.numBlocks,
        ))
    field = cp.asarray(
        _field_base(cfg, topology.thetaOps, boundary),
        dtype=_dtype(cfg.blockPrecision),
    )
    for group in groups:
        field += group.fieldResponse
        del group["fieldResponse"]
    factor_key, warm_key = _orbit_solver_cache_keys(cfg, num_bounce)
    system = Struct(
        groups=groups, field=field,
        factorizationKey=factor_key, warmStartKey=warm_key,
    )
    _GPU_ORBIT_ASSEMBLY_CACHE = (assembly_key, system, metadata)
    return system, metadata


class GpuGroupedOrbitShiftSolver:
    def __init__(self, system, sigma, precision="single"):
        global _GPU_ORBIT_CACHE
        self.system, self.sigma = system, sigma
        self.nf = system.field.shape[0]
        self.orbit_sizes = np.concatenate([
            np.full(group.numBlocks, group.blockSize, dtype=int) for group in system.groups
        ])
        self.nd = int(np.sum(self.orbit_sizes))
        self.num_unknowns = self.nd + self.nf
        dtype = _dtype(precision)
        self.groups = []
        factorized = 0
        cache_key = ((system.factorizationKey, complex(sigma), precision)
                     if system.factorizationKey is not None else None)
        cache_hit = (_GPU_ORBIT_CACHE is not None and cache_key is not None
                     and _GPU_ORBIT_CACHE[0] == cache_key)
        if cache_hit:
            cached_state = _GPU_ORBIT_CACHE[1]
            self.field = cached_state.field
            cached_groups = cached_state.groups
        else:
            self.field = cp.asarray(system.field, dtype=dtype)
            cached_groups = []
        schur = self.field.copy()
        new_cache = []
        for group_index, source in enumerate(system.groups):
            ratio = source.temperatureGradientRatio
            if cache_hit:
                fixed = cached_groups[group_index]
            else:
                kinetic = cp.asarray(source.kinetic, dtype=dtype)
                base = cp.asarray(source.fieldCouplingBase, dtype=dtype)
                temperature = cp.asarray(source.fieldCouplingTemperatureGradient, dtype=dtype)
                moment = cp.asarray(source.momentCoupling, dtype=dtype)
                shifted_pages = cp.moveaxis(kinetic, 2, 0) - dtype(sigma) * cp.eye(source.blockSize, dtype=dtype)[None]
                use_reflection = (source.get("reflectionHalfFactorization", False)
                                  and source.numBlocks % 2 == 0)
                if source.numBlocks and use_reflection:
                    num_independent = source.numBlocks // 2
                    independent_inverse = cp.linalg.inv(
                        shifted_pages[:num_independent])
                    permutation = cp.asarray(source.reflectionPermutation)
                    inverse_pages = cp.empty_like(shifted_pages)
                    inverse_pages[:num_independent] = independent_inverse
                    inverse_pages[num_independent:] = independent_inverse[
                        :, permutation, :][:, :, permutation]
                    factorized += num_independent
                elif source.numBlocks:
                    inverse_pages = cp.linalg.inv(shifted_pages)
                    factorized += source.numBlocks
                else:
                    inverse_pages = shifted_pages.copy()
                combined = base + dtype(ratio) * temperature
                response_reference = cp.matmul(
                    inverse_pages, cp.moveaxis(combined, 2, 0)
                )
                if source.numBlocks:
                    schur_reference = cp.sum(cp.matmul(
                        cp.moveaxis(moment, 2, 0), response_reference
                    ), axis=0)
                else:
                    schur_reference = cp.zeros_like(schur)
                fixed = Struct(
                    shifted=cp.moveaxis(shifted_pages, 0, 2),
                    inverse=cp.moveaxis(inverse_pages, 0, 2),
                    responseReference=cp.moveaxis(response_reference, 0, 2),
                    responseTemperature=None,
                    schurReference=schur_reference, schurTemperature=None,
                    referenceRatio=ratio,
                    base=base, temperature=temperature, moment=moment,
                    momentFlat=moment.reshape(self.nf, -1),
                )
                new_cache.append(fixed)
            ratio_delta = ratio - fixed.referenceRatio
            if ratio_delta == 0:
                response = fixed.responseReference
                schur_response = fixed.schurReference
            else:
                if fixed.responseTemperature is None:
                    response_temperature = cp.matmul(
                        cp.moveaxis(fixed.inverse, 2, 0),
                        cp.moveaxis(fixed.temperature, 2, 0),
                    )
                    fixed.responseTemperature = cp.moveaxis(
                        response_temperature, 0, 2
                    )
                    if source.numBlocks:
                        fixed.schurTemperature = cp.sum(cp.matmul(
                            cp.moveaxis(fixed.moment, 2, 0),
                            response_temperature,
                        ), axis=0)
                    else:
                        fixed.schurTemperature = cp.zeros_like(schur)
                response = (
                    fixed.responseReference
                    + dtype(ratio_delta) * fixed.responseTemperature
                )
                schur_response = (
                    fixed.schurReference
                    + dtype(ratio_delta) * fixed.schurTemperature
                )
            coupling = fixed.base + dtype(ratio) * fixed.temperature
            schur -= schur_response
            self.groups.append(Struct(
                blockSize=source.blockSize, numBlocks=source.numBlocks,
                flatIndices=source.flatIndices,
                shifted=fixed.shifted, inverse=fixed.inverse,
                response=response, coupling=coupling,
                moment=fixed.moment, momentFlat=fixed.momentFlat,
            ))
        if cache_key is not None and not cache_hit:
            _GPU_ORBIT_CACHE = (cache_key, Struct(field=self.field, groups=new_cache))
        self.schur = schur
        self.info = Struct(
            numBlocks=len(self.orbit_sizes), orbitSizes=self.orbit_sizes,
            groupSizes=np.array([group.blockSize for group in system.groups]),
            numFields=self.nf, numUnknowns=self.num_unknowns,
            field=cp.asnumpy(self.field), schur=cp.asnumpy(schur), usedGpu=True,
            kineticCacheHit=cache_hit, numFactorizedPages=factorized,
        )

    def _matrix(self, value):
        device = isinstance(value, cp.ndarray)
        value = cp.asarray(value, dtype=self.field.dtype)
        vector = value.ndim == 1
        return (value[:, None] if vector else value), vector, device

    def solve(self, rhs):
        rhs, vector, device = self._matrix(rhs)
        count = rhs.shape[1]
        field_rhs = rhs[self.nd:].copy()
        free_groups = []
        for group in self.groups:
            pages = rhs[group.flatIndices].reshape((group.blockSize, group.numBlocks, count), order="F").transpose(0, 2, 1)
            free = cp.einsum("ijb,jkb->ikb", group.inverse, pages, optimize=True)
            free_flat = free.transpose(1, 0, 2).reshape(count, -1).T
            field_rhs -= group.momentFlat @ free_flat
            free_groups.append(free)
        field_solution = cp.linalg.solve(self.schur, field_rhs)
        result = cp.zeros((self.num_unknowns, count), dtype=self.field.dtype)
        for group, free in zip(self.groups, free_groups):
            orbit = free - cp.einsum(
                "ifb,fk->ikb", group.response,
                field_solution, optimize=True,
            )
            result[group.flatIndices] = orbit.transpose(0, 2, 1).reshape((-1, count), order="F")
        result[self.nd:] = field_solution
        result = result[:, 0] if vector else result
        return _return(result, device)

    def apply_shifted(self, value):
        value, vector, device = self._matrix(value)
        count = value.shape[1]
        field_vector = value[self.nd:]
        field_result = self.field @ field_vector
        result = cp.zeros((self.num_unknowns, count), dtype=self.field.dtype)
        for group in self.groups:
            orbit = value[group.flatIndices].reshape((group.blockSize, group.numBlocks, count), order="F").transpose(0, 2, 1)
            pages = (cp.einsum("ijb,jkb->ikb", group.shifted, orbit, optimize=True)
                     + cp.einsum("ifb,fk->ikb", group.coupling,
                                 field_vector, optimize=True))
            orbit_flat = orbit.transpose(1, 0, 2).reshape(count, -1).T
            field_result += group.momentFlat @ orbit_flat
            result[group.flatIndices] = pages.transpose(0, 2, 1).reshape((-1, count), order="F")
        result[self.nd:] = field_result
        result = result[:, 0] if vector else result
        return _return(result, device)


def _mass(vector, num_fields):
    result = vector.copy()
    result[-num_fields:] = 0
    return result


def solve_eigenproblem_gpu(solver, cfg, sigma, initial_vector=None):
    start = perf_counter()
    n = solver.num_unknowns
    dimension = min(max(2, cfg.eigsSubspaceDimension), n - 1)
    target = max(cfg.eigsTolerance, 5e-6) if cfg.blockPrecision == "single" else cfg.eigsTolerance
    dtype = _dtype(cfg.blockPrecision)
    if initial_vector is None:
        index = cp.arange(1, n + 1, dtype=dtype)
        basis_start = cp.sin(index * dtype(np.sqrt(2))) + 1j * cp.cos(index * dtype(np.sqrt(3)))
    else:
        if np.asarray(initial_vector).size != n:
            raise ValueError("cfg.eigenInitialVector has an incompatible length")
        basis_start = cp.asarray(initial_vector, dtype=dtype).reshape(-1)
    basis_start /= cp.linalg.norm(basis_start)
    ritz_value, residual, iterations = np.nan, np.inf, 0
    for restart in range(1, cfg.gpuArnoldiMaxRestarts + 1):
        basis = cp.zeros((n, dimension + 1), dtype=dtype)
        hessenberg = cp.zeros((dimension + 1, dimension), dtype=dtype)
        basis[:, 0] = basis_start
        for column in range(dimension):
            if perf_counter() - start > cfg.singleShiftTimeLimit:
                raise TimeoutError(f"Single-shift eigensolve exceeded {cfg.singleShiftTimeLimit:.3g} seconds")
            work = solver.solve(_mass(basis[:, column], solver.nf))
            projection = basis[:, :column + 1].conj().T @ work
            work -= basis[:, :column + 1] @ projection
            correction = basis[:, :column + 1].conj().T @ work
            work -= basis[:, :column + 1] @ correction
            hessenberg[:column + 1, column] = projection + correction
            next_norm = cp.linalg.norm(work)
            hessenberg[column + 1, column] = next_norm
            iterations += 1
            if column < dimension - 1:
                basis[:, column + 1] = work / cp.maximum(next_norm, cp.finfo(dtype).tiny)
        host_h = cp.asnumpy(hessenberg)
        subdiagonal = np.diag(host_h[1:, :])
        breakdown = np.flatnonzero(np.abs(subdiagonal) <= np.finfo(np.float32 if dtype == cp.complex64 else np.float64).eps)
        effective = int(breakdown[0] + 1) if len(breakdown) else dimension
        values, vectors = np.linalg.eig(host_h[:effective, :effective])
        selected = np.argmax(np.abs(values))
        ritz_value = values[selected]
        coefficient = vectors[:, selected]
        basis_start = basis[:, :effective] @ cp.asarray(coefficient, dtype=dtype)
        basis_start /= cp.linalg.norm(basis_start)
        residual = (0.0 if effective < dimension else
                    abs(host_h[effective, effective - 1] * coefficient[-1])
                    / max(abs(ritz_value), np.finfo(float).eps))
        if residual <= target:
            break
    mode_device = basis_start
    omega = sigma + 1 / ritz_value
    mass_mode = _mass(mode_device, solver.nf)
    shifted_mode = solver.apply_shifted(mode_device)
    standard_mode = shifted_mode + dtype(sigma) * mass_mode
    eigen_residual = cp.linalg.norm(shifted_mode - dtype(omega - sigma) * mass_mode) / cp.maximum(cp.linalg.norm(standard_mode), cp.finfo(dtype).eps)
    field_residual = shifted_mode[-solver.nf:]
    field_mode = mode_device[-solver.nf:]
    field_term = cp.asarray(solver.system.field, dtype=dtype) @ field_mode
    moment_term = field_residual - field_term
    field_constraint = cp.linalg.norm(field_residual) / cp.maximum(
        cp.linalg.norm(field_term) + cp.linalg.norm(moment_term), cp.finfo(dtype).eps
    )
    return Struct(
        omega=complex(omega), modeReduced=cp.asnumpy(mode_device),
        eigenResidual=float(eigen_residual.get()),
        fieldConstraintResidual=float(field_constraint.get()),
        eigensolveTime=perf_counter() - start,
        gpuArnoldiInfo=Struct(iterations=iterations, restarts=restart,
                              inverseResidualEstimate=float(residual),
                              converged=bool(residual <= target), warmStartUsed=False),
    )


def solve_orbit_gpu(cfg, total_start=None):
    from .orbit import _finalize, _ORBIT_WARM_START_CACHE
    from .physics import resolve_kinetic_species
    if total_start is None:
        total_start = perf_counter()
    assembly_start = perf_counter()
    kinetic_species = resolve_kinetic_species(cfg)
    system, metadata = assemble_orbit_grouped_system_gpu(
        cfg, cfg.numBouncePoints, kinetic_species
    )
    cp.cuda.get_current_stream().synchronize()
    assembly_time = perf_counter() - assembly_start
    from .selection import solve_selected
    solve_cfg = cfg.deepcopy()
    warm_used = False
    if (system.warmStartKey is not None and solve_cfg.eigsInitialVector is None
            and system.warmStartKey in _ORBIT_WARM_START_CACHE):
        solve_cfg.eigsInitialVector = _ORBIT_WARM_START_CACHE[system.warmStartKey]
        warm_used = True
    factory = lambda shift: GpuGroupedOrbitShiftSolver(system, shift, cfg.blockPrecision)
    solution, shifted, timing, mode_scan = solve_selected(solve_cfg, factory)
    if system.warmStartKey is not None:
        _ORBIT_WARM_START_CACHE[system.warmStartKey] = solution.modeReduced.copy()
    result = Struct() if cfg.compactResult else _finalize(solution, metadata, cfg, shifted.info)
    result.update(
        runtime=perf_counter() - total_start, matrix=None, massMatrix=None,
        eigenvalues=solution.omega, omega=solution.omega,
        reducedMode=solution.modeReduced, eigenResidual=solution.eigenResidual,
        fieldConstraintResidual=solution.fieldConstraintResidual, usedGpu=True,
        timing=Struct(assembly=assembly_time, factorization=timing.factorization,
                      probe=timing.probe, eigensolve=timing.eigensolve),
        blockSolverInfo=shifted.info,
    )
    if "gpuArnoldiInfo" in solution:
        result.gpuArnoldiInfo = solution.gpuArnoldiInfo
        result.gpuArnoldiInfo.warmStartUsed = warm_used
    if mode_scan is not None:
        result.modeScan = mode_scan
    result.blockSolverInfo.warmStartUsed = warm_used
    result.blockSolverInfo.gpuAssembly = True
    return result
