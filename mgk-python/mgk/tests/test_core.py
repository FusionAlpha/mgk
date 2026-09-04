import numpy as np
from scipy.special import j1

from mgk import Struct
from mgk.internal.configuration import build_config
from mgk.internal.discretization import build_orbit_kinematics, build_theta_operators
from mgk.internal.geometry import evaluate_geometry
from mgk.internal.orbit import _orbit_solver_cache_keys
from mgk.internal.physics import two_j1_over_argument


def test_struct_recursive_attribute_and_mapping_access():
    value = Struct(section={"answer": 42}, items=[{"x": 1}])
    assert value.section.answer == value["section"]["answer"] == 42
    assert value.items[0].x == 1


def test_default_normalization_contract():
    input_cfg, cfg, normalization = build_config({"solver": {"useGpu": False}})
    physical = input_cfg.physical
    assert cfg.epsn == physical.densityGradientLength / physical.majorRadius
    assert cfg.etai == physical.densityGradientLength / physical.ionTemperatureGradientLength
    assert cfg.kt == physical.binormalWavenumber * normalization.gyroradius
    assert cfg.discretization == "orbit"
    assert cfg.parallelBoundary == "periodic"


def test_legacy_magnetic_mirror_false_still_selects_orbit():
    _, cfg, _ = build_config({
        "model": {"magneticMirror": False},
        "solver": {"useGpu": False},
    })
    assert cfg.discretization == "orbit"


def test_geometry_matches_s_alpha_definition():
    cfg = Struct(s=0.8, q=1.4, alpha=0.6, tk=0.3, kt=0.25, epsilon=0.2)
    theta = np.linspace(-np.pi, np.pi, 101)
    geometry = evaluate_geometry(theta, cfg)
    shear = cfg.s * (theta - cfg.tk) - cfg.alpha * np.sin(theta)
    inverse_field = 1 + cfg.epsilon * np.cos(theta)
    np.testing.assert_allclose(geometry.shearCoordinate, shear, atol=1e-14)
    np.testing.assert_allclose(geometry.kPerpendicularSquared,
                               cfg.kt**2 * (1 + shear**2) * inverse_field**2, atol=1e-14)
    correction = cfg.alpha / (2 * cfg.q**2)
    base_drift = np.cos(theta) + shear * np.sin(theta)
    np.testing.assert_allclose(
        geometry.driftCurvature, base_drift - correction, atol=1e-14
    )
    np.testing.assert_allclose(
        geometry.driftParallelCorrection, correction, atol=1e-14
    )
    parallel_speed = np.linspace(-1.2, 1.2, len(theta))
    perpendicular_speed = np.linspace(0.1, 1.4, len(theta))
    drift = (
        geometry.driftCurvature
        * (parallel_speed**2 + perpendicular_speed**2 / 2)
        + geometry.driftParallelCorrection * parallel_speed**2
    )
    expected_drift = (
        base_drift * (parallel_speed**2 + perpendicular_speed**2 / 2)
        - cfg.alpha * perpendicular_speed**2 / (4 * cfg.q**2)
    )
    np.testing.assert_allclose(drift, expected_drift, atol=1e-14)


def test_s_alpha_pressure_drift_vanishes_at_alpha_zero():
    cfg = Struct(s=0.8, q=1.4, alpha=0.0, tk=0.3, kt=0.25, epsilon=0.2)
    theta = np.linspace(-np.pi, np.pi, 17)
    geometry = evaluate_geometry(theta, cfg)
    np.testing.assert_allclose(
        geometry.driftCurvature,
        np.cos(theta) + cfg.s * (theta - cfg.tk) * np.sin(theta),
        atol=1e-14,
    )
    np.testing.assert_array_equal(geometry.driftParallelCorrection, 0.0)


def test_velocity_grid_preserves_order_and_moments():
    _, cfg, _ = build_config({"solver": {"useGpu": False}})
    theta = build_theta_operators(cfg).theta
    velocity = build_orbit_kinematics(cfg, theta)
    mapped_energy = (
        velocity.parallelSpeed**2 + velocity.perpendicularSpeed**2
    ) / 2
    np.testing.assert_allclose(
        mapped_energy,
        np.broadcast_to(velocity.energy[:, None], mapped_energy.shape),
        atol=1e-13,
    )
    assert np.all(velocity.referenceIntegrationWeight > 0)


def test_theta_operator_differentiates_polynomial():
    _, cfg, _ = build_config({"grid": {"numTheta": 17}, "solver": {"useGpu": False}})
    ops = build_theta_operators(cfg)
    np.testing.assert_allclose(ops.dTheta @ ops.theta**4, 4 * ops.theta**3, atol=2e-10)


def test_legacy_factorization_cache_switch_mapping():
    _, disabled, _ = build_config({
        "solver": {"useGpu": False, "enableGpuFactorizationCache": False}
    })
    _, overridden, _ = build_config({
        "solver": {
            "useGpu": False,
            "enableGpuFactorizationCache": False,
            "enableFactorizationCache": True,
        }
    })
    assert not disabled.enableFactorizationCache
    assert overridden.enableFactorizationCache
    assert 1 <= disabled.cpuFactorizationWorkers <= 16
    assert 1 <= disabled.cpuOperatorWorkers <= 8
    assert disabled.eigsHotSubspaceDimension == 5
    assert disabled.enableWarmRitz
    _, serial, _ = build_config({
        "solver": {
            "useGpu": False,
            "cpuFactorizationWorkers": 1,
            "cpuOperatorWorkers": 1,
        }
    })
    assert serial.cpuFactorizationWorkers == 1
    assert serial.cpuOperatorWorkers == 1
    _, tuned, _ = build_config({
        "solver": {"useGpu": False, "eigenHotSubspaceDimension": 4}
    })
    assert tuned.eigsHotSubspaceDimension == 4


def test_gpu_factorization_cache_switch_is_independent_from_cpu_cache():
    _, gpu_cfg, _ = build_config({
        "solver": {
            "useGpu": True,
            "enableFactorizationCache": False,
            "enableGpuFactorizationCache": True,
        }
    })
    assert not gpu_cfg.enableFactorizationCache
    assert gpu_cfg.enableGpuFactorizationCache
    assert _orbit_solver_cache_keys(gpu_cfg, gpu_cfg.numBouncePoints)[0] is not None

    _, gpu_uncached, _ = build_config({
        "solver": {
            "useGpu": True,
            "enableFactorizationCache": False,
            "enableGpuFactorizationCache": False,
        }
    })
    assert _orbit_solver_cache_keys(gpu_uncached, gpu_uncached.numBouncePoints) == (None, None)


def test_two_j1_over_argument_small_argument_branch():
    # The analytic series is used at |x| <= 1e-4.
    argument = np.array([0.0, 1e-8, 1e-5, 1e-4, 2e-4, 0.2])
    value, derivative = two_j1_over_argument(argument, return_derivative=True)
    expected_value = np.ones_like(argument)
    small = np.abs(argument) <= 1e-4
    expected_value[small] = 1 - argument[small] ** 2 / 8 + argument[small] ** 4 / 192
    expected_derivative = np.zeros_like(argument)
    expected_derivative[small] = -argument[small] / 4 + argument[small] ** 3 / 48
    np.testing.assert_allclose(value[small], expected_value[small], rtol=0, atol=1e-16)
    np.testing.assert_allclose(derivative[small], expected_derivative[small], rtol=0, atol=1e-16)
    np.testing.assert_allclose(value[-1], 2 * j1(0.2) / 0.2,
                               rtol=1e-14, atol=1e-14)
