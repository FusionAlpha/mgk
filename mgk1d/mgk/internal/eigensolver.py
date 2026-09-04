"""Shared shift-invert eigenvalue solve."""

from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.sparse.linalg import LinearOperator, ArpackNoConvergence, eigs

from mgk._struct import Struct


def apply_mass(vector, num_fields):
    result = np.array(vector, dtype=complex, copy=True)
    result[-num_fields:] = 0
    return result


def _candidate_diagnostics(shift_solver, sigma, omega, mode) -> Struct:
    mass_mode = (shift_solver.apply_mass(mode)
                 if hasattr(shift_solver, "apply_mass")
                 else apply_mass(mode, shift_solver.nf))
    shifted_mode = shift_solver.apply_shifted(mode)
    standard_mode = shifted_mode + sigma * mass_mode
    eigen_residual = np.linalg.norm(
        shifted_mode - (omega - sigma) * mass_mode
    ) / max(np.linalg.norm(standard_mode), np.finfo(float).eps)
    field_residual = shifted_mode[-shift_solver.nf:]
    field_mode = mode[-shift_solver.nf:]
    field_matrix = np.asarray(shift_solver.info.field)
    field_term = field_matrix @ field_mode
    moment_term = field_residual - field_term
    field_constraint_residual = np.linalg.norm(field_residual) / max(
        np.linalg.norm(field_term) + np.linalg.norm(moment_term),
        np.finfo(float).eps,
    )
    return Struct(
        omega=omega,
        modeReduced=mode,
        eigenResidual=float(eigen_residual),
        fieldConstraintResidual=float(field_constraint_residual),
    )


def _warm_ritz(matvec, initial_vector, dimension):
    """Build a short, twice-orthogonalized Arnoldi/Ritz correction."""
    vector = np.asarray(initial_vector, dtype=complex).reshape(-1)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("warm Ritz predictor has zero or non-finite norm")
    vector = vector / norm
    # Arnoldi consumes columns one at a time; Fortran order keeps every
    # operator input contiguous instead of passing a strided C-order view.
    basis = np.empty(
        (vector.size, dimension + 1), dtype=complex, order="F"
    )
    hessenberg = np.zeros((dimension + 1, dimension), dtype=complex)
    basis[:, 0] = vector
    effective_dimension = dimension
    for column in range(dimension):
        work = matvec(basis[:, column])
        for _ in range(2):
            projection = basis[:, :column + 1].conj().T @ work
            hessenberg[:column + 1, column] += projection
            work -= basis[:, :column + 1] @ projection
        next_norm = np.linalg.norm(work)
        hessenberg[column + 1, column] = next_norm
        if not np.isfinite(next_norm):
            raise FloatingPointError("warm Ritz Arnoldi vector is non-finite")
        if next_norm <= np.finfo(float).eps:
            effective_dimension = column + 1
            break
        basis[:, column + 1] = work / next_norm
    projected = hessenberg[:effective_dimension, :effective_dimension]
    inverse_values, vectors = np.linalg.eig(projected)
    selected = int(np.argmax(np.abs(inverse_values)))
    inverse = inverse_values[selected]
    mode = basis[:, :effective_dimension] @ vectors[:, selected]
    mode /= np.linalg.norm(mode)
    if effective_dimension < hessenberg.shape[0]:
        inverse_residual = abs(
            hessenberg[effective_dimension, effective_dimension - 1]
            * vectors[-1, selected]
        ) / max(abs(inverse), np.finfo(float).eps)
    else:
        inverse_residual = np.nan
    return inverse, mode, effective_dimension, float(inverse_residual)


def solve_eigenproblem(shift_solver, cfg, sigma, initial_vector=None,
                       warm_ritz_vector=None,
                       warm_ritz_dimension=None) -> Struct:
    if cfg.eigenBackend == "gpu_arnoldi":
        raise NotImplementedError("gpu_arnoldi requires the optional CuPy backend")
    n = shift_solver.num_unknowns
    if initial_vector is not None and np.asarray(initial_vector).size != n:
        raise ValueError("cfg.eigenInitialVector has an incompatible length")
    if warm_ritz_vector is not None and np.asarray(warm_ritz_vector).size != n:
        raise ValueError("warm Ritz predictor has an incompatible length")
    if warm_ritz_dimension is None:
        warm_ritz_dimension = 3
    if (not isinstance(warm_ritz_dimension, (int, np.integer))
            or warm_ritz_dimension < 2):
        raise ValueError("warm Ritz dimension must be an integer >= 2")
    requested_warm_ritz_dimension = int(warm_ritz_dimension)
    warm_start_used = initial_vector is not None
    hot_subspace_used = (
        warm_start_used
        and bool(shift_solver.info.get("kineticCacheHit", False))
    )
    subspace_dimension = cfg.eigsSubspaceDimension
    if hot_subspace_used:
        subspace_dimension = min(
            subspace_dimension, cfg.eigsHotSubspaceDimension
        )
    subspace_dimension = min(subspace_dimension, n - 1)
    operator_calls = 0
    start = perf_counter()

    def matvec(vector):
        nonlocal operator_calls
        if perf_counter() - start > cfg.singleShiftTimeLimit:
            raise TimeoutError(f"Single-shift eigensolve exceeded {cfg.singleShiftTimeLimit:.3g} seconds")
        operator_calls += 1
        mass_vector = (shift_solver.apply_mass(vector)
                       if hasattr(shift_solver, "apply_mass")
                       else apply_mass(vector, shift_solver.nf))
        return shift_solver.solve(mass_vector)

    warm_ritz_attempted = False
    warm_ritz_accepted = False
    warm_ritz_calls = 0
    warm_ritz_dimension = 0
    warm_ritz_residual = np.inf
    warm_ritz_inverse_residual = np.inf
    warm_iteration_time = 0.0
    # Keep margin below the public 1e-9 residual gate; on the non-normal
    # Rewoldt operator this also kept the p8 frequency difference below 1e-11.
    acceptance_tolerance = min(5e-10, 10 * cfg.eigsTolerance)
    field_acceptance_tolerance = 1e-10
    if (warm_ritz_vector is not None
            and cfg.enableWarmRitz and cfg.blockPrecision == "double"):
        warm_ritz_attempted = True
        warm_start = perf_counter()
        calls_before = operator_calls
        try:
            inverse, mode, warm_ritz_dimension, warm_ritz_inverse_residual = (
                _warm_ritz(
                    matvec, warm_ritz_vector,
                    min(requested_warm_ritz_dimension, n - 1),
                )
            )
            warm_iteration_time = perf_counter() - warm_start
            warm_ritz_calls = operator_calls - calls_before
            if not np.isfinite(inverse) or abs(inverse) <= np.finfo(float).eps:
                raise FloatingPointError("warm Ritz inverse eigenvalue is invalid")
            omega = sigma + 1 / inverse
            candidate = _candidate_diagnostics(
                shift_solver, sigma, omega, mode
            )
            warm_ritz_residual = candidate.eigenResidual
            warm_ritz_accepted = (
                np.isfinite(omega)
                and warm_ritz_residual <= acceptance_tolerance
                and candidate.fieldConstraintResidual <= field_acceptance_tolerance
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            warm_iteration_time = perf_counter() - warm_start
            warm_ritz_calls = operator_calls - calls_before
            candidate = None
        if warm_ritz_accepted:
            candidate.update(
                eigensolveTime=warm_iteration_time,
                eigsConverged=True,
                eigsInfo=Struct(
                    operatorCalls=operator_calls,
                    arpackOperatorCalls=0,
                    subspaceDimension=subspace_dimension,
                    configuredSubspaceDimension=cfg.eigsSubspaceDimension,
                    hotSubspaceDimension=cfg.eigsHotSubspaceDimension,
                    warmStartUsed=warm_start_used,
                    hotSubspaceUsed=hot_subspace_used,
                    warmRitzAttempted=True,
                    warmRitzAccepted=True,
                    warmRitzOperatorCalls=warm_ritz_calls,
                    warmRitzDimension=warm_ritz_dimension,
                    warmRitzResidual=warm_ritz_residual,
                    warmRitzResidualTolerance=acceptance_tolerance,
                    warmRitzFieldResidualTolerance=field_acceptance_tolerance,
                    warmRitzInverseResidualEstimate=warm_ritz_inverse_residual,
                ),
            )
            return candidate

    operator = LinearOperator((n, n), matvec=matvec, dtype=np.complex128)
    kwargs = dict(k=1, which="LM", tol=cfg.eigsTolerance,
                  ncv=subspace_dimension)
    if cfg.eigsMaxIterations is not None:
        kwargs["maxiter"] = cfg.eigsMaxIterations
    if initial_vector is not None:
        kwargs["v0"] = np.asarray(initial_vector).reshape(-1)
    arpack_calls_before = operator_calls
    arpack_start = perf_counter()
    converged = True
    try:
        inverse_values, vectors = eigs(operator, **kwargs)
    except ArpackNoConvergence as exc:
        if exc.eigenvalues is None or len(exc.eigenvalues) == 0:
            raise
        inverse_values, vectors, converged = exc.eigenvalues, exc.eigenvectors, False
    elapsed = warm_iteration_time + perf_counter() - arpack_start
    inverse = inverse_values[0]
    mode = vectors[:, 0]
    omega = sigma + 1 / inverse
    candidate = _candidate_diagnostics(shift_solver, sigma, omega, mode)
    candidate.update(
        eigensolveTime=elapsed, eigsConverged=converged,
        eigsInfo=Struct(
            operatorCalls=operator_calls,
            arpackOperatorCalls=operator_calls - arpack_calls_before,
            subspaceDimension=subspace_dimension,
            configuredSubspaceDimension=cfg.eigsSubspaceDimension,
            hotSubspaceDimension=cfg.eigsHotSubspaceDimension,
            warmStartUsed=warm_start_used,
            hotSubspaceUsed=hot_subspace_used,
            warmRitzAttempted=warm_ritz_attempted,
            warmRitzAccepted=False,
            warmRitzOperatorCalls=warm_ritz_calls,
            warmRitzDimension=warm_ritz_dimension,
            warmRitzResidual=warm_ritz_residual,
            warmRitzResidualTolerance=acceptance_tolerance,
            warmRitzFieldResidualTolerance=field_acceptance_tolerance,
            warmRitzInverseResidualEstimate=warm_ritz_inverse_residual,
        ),
    )
    return candidate
