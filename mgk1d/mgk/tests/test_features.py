import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence

import mgk
from mgk._struct import Struct
from mgk.internal import selection


def test_orbit_only_solver_rejects_explicit_matrix_request():
    with np.testing.assert_raises_regex(ValueError, "matrix-free"):
        mgk.solve({"solver": {"useGpu": False, "returnMatrices": True}})


def test_bparallel_open_configuration_uses_orbit_backend():
    result = mgk.solve({
        "physical": {"minorRadius": 0.17, "electronBeta": 0.01},
        "model": {"parallelBoundary": "open", "bparallel": True},
        "grid": {"numTheta": 17, "numEnergy": 2, "numPitch": 2},
        "solver": {
            "useGpu": False, "blockPrecision": "double",
            "eigenTolerance": 1e-9, "eigenSubspaceDimension": 20,
            "singleShiftTimeLimit": 60,
            "enableFactorizationCache": False,
            "enableGpuResultCache": False,
        },
    })
    assert result.discretization == "orbit"
    assert result.eigenResidual < 1e-8
    assert result.fieldConstraintResidual < 1e-8
    assert result.bparallel is not None


def test_massless_electron_fluid_is_rejected_without_grid_backend():
    with np.testing.assert_raises_regex(ValueError, "removed grid backend"):
        mgk.solve({
            "physical": {"electronBeta": 0.01},
            "model": {"electronClosure": "massless", "aparallel": True},
            "solver": {"useGpu": False},
        })


def test_massless_electron_fluid_rejects_kinetic_electron():
    with np.testing.assert_raises(ValueError):
        mgk.solve({
            "physical": {"electronBeta": 0.01},
            "model": {"electronClosure": "massless", "aparallel": True},
            "species": {"enabled": True, "items": {"kind": "electron", "kinetic": True}},
            "solver": {"useGpu": False},
        })


def test_massless_electron_fluid_with_bparallel_is_also_rejected():
    with np.testing.assert_raises_regex(ValueError, "removed grid backend"):
        mgk.solve({
            "physical": {"electronBeta": 0.01},
            "model": {"electronClosure": "massless", "aparallel": True,
                      "bparallel": True, "parallelBoundary": "open"},
            "solver": {"useGpu": False},
        })


def test_max_growth_scan_returns_selection_evidence():
    result = mgk.solve({
        "grid": {"numTheta": 17, "numEnergy": 3, "numPitch": 3},
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "modeSelection": "max_growth_scan",
            "eigenSubspaceDimension": 12,
            "singleShiftTimeLimit": 20,
        },
    })
    assert result.modeScan.valid[result.modeScan.selectedIndex]
    assert result.modeScan.confirmationAccepted
    assert result.eigenResidual < 1e-7


def test_max_growth_scan_skips_arpack_nonconvergence(monkeypatch):
    selected_omega = 0.2 + 0.3j

    def fake_solve_one(solver, cfg, shift, initial, **kwargs):
        if shift not in {0.0 + 0.5j, selected_omega + 0.05j}:
            raise ArpackNoConvergence(
                "test nonconvergence", np.array([]), np.empty((2, 0))
            )
        return Struct(
            omega=selected_omega,
            eigenResidual=1e-12,
            eigensolveTime=0.01,
            eigsConverged=True,
            modeReduced=np.ones(2, dtype=complex),
        )

    monkeypatch.setattr(selection, "_solve_one", fake_solve_one)
    cfg = Struct(
        modeSelection="max_growth_scan",
        eigenBackend="eigs",
        eigsInitialVector=None,
        eigsTolerance=1e-8,
    )
    solution, _, _, scan = selection.solve_selected(
        cfg, lambda shift: Struct(num_unknowns=2)
    )
    assert solution.omega == selected_omega
    assert np.count_nonzero(scan.valid) == 1
    assert np.count_nonzero(scan.converged) == 1
    assert scan.confirmationAccepted
