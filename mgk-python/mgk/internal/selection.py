"""Nearest-mode and multi-shift fastest-growth selection."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from scipy.sparse.linalg import ArpackNoConvergence

from mgk._struct import Struct
from .eigensolver import solve_eigenproblem


def _solve_one(solver, cfg, shift, initial, warm_ritz_vector=None,
               warm_ritz_dimension=None):
    if cfg.eigenBackend == "gpu_arnoldi":
        from .gpu import solve_eigenproblem_gpu
        return solve_eigenproblem_gpu(solver, cfg, shift, initial)
    return solve_eigenproblem(
        solver, cfg, shift, initial, warm_ritz_vector=warm_ritz_vector,
        warm_ritz_dimension=warm_ritz_dimension,
    )


def solve_selected(cfg, solver_factory, warm_ritz_vector=None,
                   warm_ritz_dimension=None):
    """Return solution, selected factorization, aggregate timing, and scan metadata."""
    if cfg.modeSelection == "nearest":
        factor_start = perf_counter()
        solver = solver_factory(cfg.eigsGuess)
        factorization = perf_counter() - factor_start
        solution = _solve_one(
            solver, cfg, cfg.eigsGuess, cfg.eigsInitialVector,
            warm_ritz_vector=warm_ritz_vector,
            warm_ritz_dimension=warm_ritz_dimension,
        )
        return solution, solver, Struct(
            factorization=factorization, probe=0.0, eigensolve=solution.eigensolveTime
        ), None

    shifts = np.arange(-1.5, 1.5001, 0.25).astype(complex) + 0.5j
    count = len(shifts)
    eigenvalues = np.full(count, np.nan + 1j * np.nan)
    residuals = np.full(count, np.inf)
    converged = np.zeros(count, dtype=bool)
    timed_out = np.zeros(count, dtype=bool)
    factor_times = np.zeros(count)
    solve_times = np.zeros(count)
    solutions = [None] * count
    solvers = [None] * count
    initial = cfg.eigsInitialVector
    for index, shift in enumerate(shifts):
        factor_start = perf_counter()
        solver = solver_factory(shift)
        factor_times[index] = perf_counter() - factor_start
        if initial is None:
            vector_index = np.arange(1, solver.num_unknowns + 1)
            initial = np.sin(vector_index * np.sqrt(2)) + 1j * np.cos(vector_index * np.sqrt(3))
        solve_start = perf_counter()
        try:
            candidate = _solve_one(solver, cfg, shift, initial)
        except TimeoutError:
            timed_out[index] = True
            solve_times[index] = perf_counter() - solve_start
            continue
        except ArpackNoConvergence:
            solve_times[index] = perf_counter() - solve_start
            continue
        eigenvalues[index] = candidate.omega
        residuals[index] = candidate.eigenResidual
        converged[index] = candidate.get("eigsConverged", candidate.get("gpuArnoldiInfo", {}).get("converged", True))
        solve_times[index] = candidate.eigensolveTime
        if converged[index]:
            solutions[index], solvers[index] = candidate, solver
    tolerance = max(1e-4, 10 * cfg.eigsTolerance)
    valid = converged & np.isfinite(eigenvalues) & (residuals <= tolerance)
    representatives = []
    for index in np.flatnonzero(valid):
        match = next((position for position, current in enumerate(representatives)
                      if abs(eigenvalues[current] - eigenvalues[index])
                      <= 1e-4 * max(1, abs(eigenvalues[index]))), None)
        if match is None:
            representatives.append(index)
        elif residuals[index] < residuals[representatives[match]]:
            representatives[match] = index
    if not representatives:
        raise RuntimeError("No converged eigenmode was found by the multi-mode scan")
    selected_index = representatives[int(np.argmax(eigenvalues[representatives].imag))]
    selected, selected_solver = solutions[selected_index], solvers[selected_index]
    confirmation_shift = selected.omega + 0.05j
    factor_start = perf_counter()
    confirmation_solver = solver_factory(confirmation_shift)
    confirmation_factor = perf_counter() - factor_start
    confirmation_start = perf_counter()
    confirmation_timed_out = False
    try:
        confirmation = _solve_one(confirmation_solver, cfg, confirmation_shift, selected.modeReduced)
        confirmation_time = confirmation.eigensolveTime
        confirmation_converged = confirmation.get(
            "eigsConverged", confirmation.get("gpuArnoldiInfo", {}).get("converged", True)
        )
        accepted = (confirmation_converged and confirmation.eigenResidual <= tolerance
                    and abs(confirmation.omega - selected.omega)
                    <= 1e-3 * max(1, abs(selected.omega)))
    except TimeoutError:
        confirmation_timed_out, accepted = True, False
        confirmation_time = perf_counter() - confirmation_start
    if accepted:
        selected, selected_solver = confirmation, confirmation_solver
    timing = Struct(
        factorization=float(np.sum(factor_times) + confirmation_factor), probe=0.0,
        eigensolve=float(np.sum(solve_times) + confirmation_time),
    )
    scan = Struct(
        shifts=shifts, eigenvalues=eigenvalues, residuals=residuals,
        converged=converged, valid=valid, timedOut=timed_out,
        factorizationTime=factor_times, eigensolveTime=solve_times,
        selectedIndex=int(selected_index), confirmationShift=confirmation_shift,
        confirmationAccepted=bool(accepted), confirmationTimedOut=confirmation_timed_out,
        residualTolerance=tolerance,
    )
    return selected, selected_solver, timing, scan
