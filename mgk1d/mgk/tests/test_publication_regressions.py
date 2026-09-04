import copy

import numpy as np
import pytest

import mgk
from mgk._struct import Struct
from mgk.internal.configuration import build_config
from mgk.internal.geometry import evaluate_geometry


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.6])
def test_s_alpha_complete_drift_formula(alpha):
    """Protect the pressure-gradient drift term that CBC alpha=0 cannot test."""
    cfg = Struct(s=0.8, q=1.4, alpha=alpha, tk=0.2, kt=0.3, epsilon=0.18)
    theta = np.linspace(-np.pi, np.pi, 97)
    geometry = evaluate_geometry(theta, cfg)
    shear = cfg.s * (theta - cfg.tk) - alpha * np.sin(theta)
    base_drift = np.cos(theta) + shear * np.sin(theta)
    v_parallel = 0.7 + 0.2 * np.cos(theta)
    v_perpendicular = 0.8 + 0.1 * np.sin(theta)
    implemented = (
        geometry.driftCurvature
        * (v_parallel**2 + 0.5 * v_perpendicular**2)
        + geometry.driftParallelCorrection * v_parallel**2
    )
    expected = (
        base_drift * (v_parallel**2 + 0.5 * v_perpendicular**2)
        - alpha * (1 + cfg.epsilon * np.cos(theta)) * v_perpendicular**2 / (2 * cfg.q**2)
    )
    np.testing.assert_allclose(implemented, expected, rtol=0, atol=2e-14)


def test_explicit_adiabatic_electrons_use_orbit_backend():
    result = mgk.solve({
        "model": {"electronClosure": "adiabatic", "parallelBoundary": "open"},
        "grid": {
            "numTheta": 17, "numEnergy": 2, "numPitch": 2,
            "numBouncePoints": 8,
        },
        "solver": {
            "useGpu": False, "blockPrecision": "double",
            "eigenTolerance": 1e-10,
            "enableFactorizationCache": False,
            "enableGpuResultCache": False,
        },
    })
    assert result.normalizedConfig.discretization == "orbit"
    assert result.normalizedConfig.electronClosure == "adiabatic"
    assert result.normalizedConfig.species.numKinetic == 1
    assert result.blockSolverInfo.numBlocks > 0
    assert result.eigenResidual < 2e-6


def test_explicit_nonkinetic_electron_matches_default_adiabatic_closure():
    common = {
        "model": {"electronClosure": "adiabatic", "parallelBoundary": "open"},
        "grid": {"numTheta": 25, "numEnergy": 3, "numPitch": 3},
        "solver": {
            "useGpu": False, "blockPrecision": "double",
            "eigenTolerance": 1e-10,
            "enableFactorizationCache": False,
            "enableGpuResultCache": False,
        },
    }
    default = mgk.solve(copy.deepcopy(common))
    explicit_cfg = copy.deepcopy(common)
    explicit_cfg["species"] = {
        "enabled": True,
        "items": {"kind": "electron", "kinetic": False},
    }
    explicit = mgk.solve(explicit_cfg)
    np.testing.assert_allclose(explicit.omega, default.omega, rtol=1e-9, atol=1e-11)
    assert explicit.normalizedConfig.species.numKinetic == 1


def test_circular_miller_approaches_large_aspect_s_alpha_geometry():
    """Check the common circular, large-aspect-ratio geometry limit."""
    theta = np.linspace(-np.pi, np.pi, 129)

    def geometry(model, minor_radius):
        _, cfg, _ = build_config({
            "physical": {"majorRadius": 1.0, "minorRadius": minor_radius},
            "geometry": {
                "model": model, "q": 1.4, "magneticShear": 0.8,
                "alpha": 0.0, "elongation": 1.0, "triangularity": 0.0,
                "shiftDerivative": 0.0, "elongationShear": 0.0,
                "triangularityShear": 0.0, "betaStar": 0.0,
                "tableResolution": 2001,
            },
            "solver": {"useGpu": False},
        })
        return evaluate_geometry(theta, cfg)

    s_alpha = geometry("s-alpha", 1e-3)
    miller = geometry("miller", 1e-3)
    for name, tolerance in (
        ("fieldStrength", 5e-6),
        ("kPerpendicularSquared", 2e-3),
        ("driftCurvature", 3e-3),
        ("parallelGradient", 5e-6),
    ):
        actual = np.asarray(miller[name])
        expected = np.asarray(s_alpha[name])
        relative = np.max(np.abs(actual - expected)) / max(
            np.max(np.abs(expected)), 1e-15
        )
        assert relative < tolerance, (name, relative)


def test_uncached_cpu_solve_is_repeatable():
    config = {
        "geometry": {"alpha": 0.3},
        "model": {"parallelBoundary": "open"},
        "grid": {"numTheta": 33, "numEnergy": 4, "numPitch": 4},
        "solver": {
            "useGpu": False, "blockPrecision": "double",
            "eigenTolerance": 1e-9,
            "singleShiftTimeLimit": 60,
            "enableFactorizationCache": False,
            "enableGpuResultCache": False,
        },
    }
    first = mgk.solve(copy.deepcopy(config))
    second = mgk.solve(copy.deepcopy(config))
    np.testing.assert_allclose(first.omega, second.omega, rtol=1e-9, atol=1e-11)
    assert first.eigenResidual < 2e-6
    assert second.eigenResidual < 2e-6
