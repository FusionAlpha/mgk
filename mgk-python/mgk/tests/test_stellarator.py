import numpy as np

import mgk
from mgk.internal.configuration import build_config
from mgk.internal.discretization import (
    build_orbit_kinematics,
    build_orbit_layout,
    build_theta_operators,
    build_trapped_path,
)
from mgk.internal.geometry import (
    evaluate_geometry,
    load_stellarator_profile,
    stellarator_bounce_interval,
    stellarator_well_centers,
)


def _profile(path, n=128):
    """Create a deliberately asymmetric, periodic local field-line profile."""
    period = 2 * np.pi
    z = np.linspace(0.0, period, n + 1)
    # One smooth magnetic well with a nonzero left/right asymmetry.  The
    # arrays are a synthetic interface smoke test, not a VMEC validation.
    field = 1.0 + 0.25 * np.cos(z) + 0.06 * np.sin(2 * z)
    metric = 1.0 + 0.15 * np.sin(z - 0.2)
    np.savez(
        path,
        z=z,
        period=period,
        B=field,
        kperpMetric=metric,
        driftCurvature=0.2 * np.cos(z) + 0.03 * np.sin(3 * z),
        driftParallelCorrection=0.02 * np.sin(z),
        parallelGradient=1.0 + 0.05 * np.cos(z),
        fieldLineLabel=np.array(0.37),
    )


def _config(profile):
    return {
        "geometry": {"model": "stellarator", "profileFile": str(profile)},
        "model": {"parallelBoundary": "periodic"},
        "grid": {
            "thetaMin": -np.pi,
            "thetaMax": np.pi,
            "numTheta": 33,
            "numEnergy": 2,
            "numPitch": 6,
            "numBouncePoints": 8,
        },
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "enableFactorizationCache": False,
            "eigenSubspaceDimension": 8,
            "eigenTolerance": 1e-7,
            "singleShiftTimeLimit": 60,
        },
    }


def test_stellarator_profile_periodicity_and_geometry(tmp_path):
    filename = tmp_path / "local_stellarator.npz"
    _profile(filename)
    _, cfg, _ = build_config(_config(filename))
    profile = load_stellarator_profile(filename)
    assert profile.fieldLineLabel == 0.37
    theta = np.linspace(-3 * np.pi, 3 * np.pi, 101)
    geometry = evaluate_geometry(theta, cfg)
    geometry_shifted = evaluate_geometry(theta + profile.period, cfg)
    for name in ("fieldStrength", "kPerpendicularSquared", "driftCurvature",
                 "driftParallelCorrection", "parallelGradient"):
        np.testing.assert_allclose(getattr(geometry, name), getattr(geometry_shifted, name))
    assert np.all(np.isfinite(geometry.kPerpendicularDerivative))


def test_stellarator_wells_roots_and_asymmetric_path(tmp_path):
    filename = tmp_path / "local_stellarator.npz"
    _profile(filename)
    _, cfg, _ = build_config(_config(filename))
    profile = load_stellarator_profile(filename)
    centers = stellarator_well_centers(cfg)
    # The domain spans exactly one profile period, so periodic copies of the
    # same physical magnetic well must not become duplicate orbit blocks.
    assert centers.size == 1
    center = centers[np.argmin(np.abs(centers))]
    target = 1.0 + 0.10
    left, right = stellarator_bounce_interval(cfg, target, center)
    assert left < center < right
    assert not np.isclose(center - left, right - center)
    np.testing.assert_allclose(
        evaluate_geometry(np.array([left, right]), cfg).fieldStrength,
        target,
        atol=2e-6,
    )

    theta_ops = build_theta_operators(cfg)
    orbits = build_orbit_kinematics(cfg, theta_ops.theta)
    assert np.all(orbits.referenceIntegrationWeight > 0)
    assert np.all(orbits.pitchWeights > 0)
    layout = build_orbit_layout(cfg, orbits)
    assert layout.numTrappedBlocks > 0
    block = np.flatnonzero(layout.blockType == "trapped")[0]
    path = build_trapped_path(
        cfg, orbits, layout.orbitIndex[block], layout.wellCenter[block],
        16, theta_ops,
    )
    np.testing.assert_allclose(path.thetaUnwrapped[0], path.leftBounce, atol=1e-10)
    np.testing.assert_allclose(path.thetaUnwrapped[8], path.rightBounce, atol=1e-10)
    assert np.all(np.isfinite(path.streamCoefficient))
    assert np.all(path.streamCoefficient > 0)
    np.testing.assert_allclose(
        (path.parallelVelocity**2 + path.perpendicularSpeed**2) / 2,
        path.energy,
        atol=2e-10,
    )


def test_stellarator_small_solve_is_finite(tmp_path):
    filename = tmp_path / "local_stellarator.npz"
    _profile(filename)
    result = mgk.solve(_config(filename))
    assert np.isfinite(result.omega.real) and np.isfinite(result.omega.imag)
    assert np.isfinite(result.eigenResidual)
    assert np.isfinite(result.fieldConstraintResidual)
    assert result.eigenResidual < 2e-6
    # Generic stellarator profiles are not reflection symmetric.  Every
    # passing/trapped page must therefore be factorized independently.
    assert result.blockSolverInfo.numFactorizedPages == result.blockSolverInfo.numBlocks
