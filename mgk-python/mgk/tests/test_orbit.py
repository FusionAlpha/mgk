import numpy as np

import mgk
from mgk.internal.configuration import build_config
from mgk.internal.discretization import (
    build_orbit_boundary,
    build_orbit_kinematics,
    build_orbit_layout,
    build_theta_operators,
    build_trapped_path,
)
from mgk.internal.orbit import (
    GroupedOrbitShiftSolver,
    _page_matmul,
    _passing_sponge,
    _sum_page_matmul,
    assemble_orbit_grouped_system,
)
from mgk.internal.physics import resolve_kinetic_species


def mirror_config(boundary="open"):
    return {
        "physical": {"minorRadius": 0.17},
        "model": {"magneticMirror": True, "parallelBoundary": boundary},
        "grid": {"numTheta": 33, "numEnergy": 4, "numPitch": 4, "numBouncePoints": 8},
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "eigenTolerance": 1e-8,
            "eigenSubspaceDimension": 30,
            "singleShiftTimeLimit": 60,
        },
    }


def test_orbit_topology_and_trapped_path_invariants():
    _, cfg, _ = build_config(mirror_config("periodic"))
    theta_ops = build_theta_operators(cfg)
    orbits = build_orbit_kinematics(cfg, theta_ops.theta)
    layout = build_orbit_layout(cfg, orbits)
    assert layout.numPassingBlocks == 2 * np.count_nonzero(orbits.passing)
    assert layout.numTrappedBlocks == np.count_nonzero(orbits.trapped) * len(layout.wellCenters)
    block = np.flatnonzero(layout.blockType == "trapped")[0]
    path = build_trapped_path(
        cfg, orbits, layout.orbitIndex[block], layout.wellCenter[block], 16, theta_ops
    )
    assert np.all(np.isfinite(path.streamCoefficient))
    assert np.all(path.streamCoefficient > 0)
    np.testing.assert_allclose(
        (path.parallelVelocity**2 + path.perpendicularSpeed**2) / 2,
        path.energy,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        path.fieldInterpolation @ np.ones(cfg.nth - 1), np.ones(path.numPoints), atol=1e-12
    )


def test_theta_map_alpha_matches_configured_sinh_map():
    config = mirror_config("open")
    config["grid"]["thetaMapAlpha"] = 1.5
    _, mapped_cfg, _ = build_config(config)
    mapped = build_theta_operators(mapped_cfg)
    config["grid"]["thetaMapAlpha"] = 0.0
    _, linear_cfg, _ = build_config(config)
    linear = build_theta_operators(linear_cfg)
    center = 0.5 * (linear_cfg.thmin + linear_cfg.thmax)
    half = 0.5 * (linear_cfg.thmax - linear_cfg.thmin)
    expected = center + half * np.sinh(1.5 * (linear.theta - center) / half) / np.sinh(1.5)
    np.testing.assert_allclose(mapped.theta, expected, rtol=2e-14, atol=2e-14)
    assert mapped.theta[0] == linear.theta[0]
    assert mapped.theta[-1] == linear.theta[-1]


def test_open_dtn_tail_elements_and_schur_reduction():
    config = mirror_config("open-dtn")
    config["grid"].update(numTheta=17, numEnergy=2, numPitch=3)
    config["model"].update(boundaryTailPeriods=1, boundaryTailPoints=9)
    config["solver"]["enableFactorizationCache"] = False
    _, cfg, _ = build_config(config)
    theta_ops = build_theta_operators(cfg)
    assert theta_ops.nth == cfg.nth
    assert len(theta_ops.elements) == 5
    for left, right in zip(theta_ops.elements[:-1], theta_ops.elements[1:]):
        assert left.indices[-1] == right.indices[0]
        assert left.theta[-1] == right.theta[0]
    system, _ = assemble_orbit_grouped_system(
        cfg, cfg.numBouncePoints, resolve_kinetic_species(cfg)
    )
    reduced = GroupedOrbitShiftSolver(system, cfg.eigsGuess, "double", 1, 1)
    full_system = system.deepcopy()
    del full_system["fieldSchurPartition"]
    direct = GroupedOrbitShiftSolver(full_system, cfg.eigsGuess, "double", 1, 1)
    rng = np.random.default_rng(20260818)
    rhs = rng.standard_normal(reduced.num_unknowns) + 1j * rng.standard_normal(reduced.num_unknowns)
    np.testing.assert_allclose(reduced.solve(rhs), direct.solve(rhs), rtol=2e-11, atol=2e-11)
    assert reduced.info.dtn.enabled
    assert reduced.info.dtn.numCoreFields + reduced.info.dtn.numTailFields == reduced.nf
    result = mgk.solve(config)
    assert result.thetaBoundary == "real-tail-dtn/passing-inflow/trapped-bounce-periodic"
    assert result.blockSolverInfo.dtn.enabled
    assert result.eigenResidual < 1e-7
    assert result.fieldConstraintResidual < 1e-8


def test_blas_page_contractions_match_einsum_reference():
    rng = np.random.default_rng(20260729)
    left = rng.standard_normal((7, 5, 11)) + 1j * rng.standard_normal((7, 5, 11))
    right = rng.standard_normal((5, 3, 11)) + 1j * rng.standard_normal((5, 3, 11))
    page_reference = np.einsum("ikb,kjb->ijb", left, right, optimize=True)
    np.testing.assert_allclose(
        _page_matmul(left, right), page_reference, rtol=5e-14, atol=5e-14
    )
    sum_reference = np.sum(page_reference, axis=2)
    np.testing.assert_allclose(
        _sum_page_matmul(left, right), sum_reference, rtol=5e-14, atol=5e-14
    )


def test_open_orbit_matches_reference():
    result = mgk.solve(mirror_config())
    # Current orbit reference after the open-boundary trapped-well containment
    # fix. The previous Python reference retained two wrapped
    # wells and therefore represented a different open-domain problem.
    expected = -0.800932817396531 - 5.42134469300182e-6j
    assert abs(result.omega - expected) < 5e-9
    assert result.blockSolverInfo.numBlocks == 40
    assert result.eigenResidual < 1e-7
    assert result.fieldConstraintResidual < 1e-8
    assert result.phi[0] == 0 and result.phi[-1] == 0


def test_symmetric_passing_orbits_factor_only_independent_pages():
    config = mirror_config()
    config["solver"]["enableFactorizationCache"] = False
    result = mgk.solve(config)
    assert not result.blockSolverInfo.kineticCacheHit
    assert 0 < result.blockSolverInfo.numFactorizedPages < result.blockSolverInfo.numBlocks
    assert abs(result.omega - (-0.800932817396531 - 5.42134469300182e-6j)) < 5e-9


def test_grouped_orbit_batched_solve_matches_individual_columns():
    config = mirror_config()
    config["solver"]["enableFactorizationCache"] = False
    _, cfg, _ = build_config(config)
    system, _ = assemble_orbit_grouped_system(
        cfg, cfg.numBouncePoints, resolve_kinetic_species(cfg)
    )
    serial = GroupedOrbitShiftSolver(
        system, cfg.eigsGuess, cfg.blockPrecision,
        cpu_factorization_workers=1, cpu_operator_workers=1,
    )
    solver = GroupedOrbitShiftSolver(
        system, cfg.eigsGuess, cfg.blockPrecision,
        cpu_factorization_workers=2, cpu_operator_workers=2,
    )
    np.testing.assert_allclose(solver.schur, serial.schur, rtol=2e-13, atol=2e-13)
    rng = np.random.default_rng(20260729)
    rhs = (rng.standard_normal((solver.num_unknowns, 3))
           + 1j * rng.standard_normal((solver.num_unknowns, 3)))
    np.testing.assert_allclose(
        solver.apply_shifted(rhs), serial.apply_shifted(rhs),
        rtol=2e-13, atol=2e-13,
    )
    batched = solver.solve(rhs)
    individual = np.column_stack([solver.solve(rhs[:, index]) for index in range(3)])
    np.testing.assert_allclose(batched, individual, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        solver.apply_shifted(batched), rhs, rtol=2e-11, atol=2e-11
    )


def test_extrapolated_boundary_has_continuous_endpoint_slope():
    config = mirror_config("open-extrapolated")
    config["model"]["boundaryCoordinateStretch"] = 1.5
    result = mgk.solve(config)
    left = np.diff(result.phi[:3]) / np.diff(result.theta[:3])
    right = np.diff(result.phi[-3:]) / np.diff(result.theta[-3:])
    assert abs(left[0] - left[1]) < 1e-10
    assert abs(right[0] - right[1]) < 1e-10


def test_extrapolated_boundary_map_stretch_and_sponge():
    config = mirror_config("open-extrapolated")
    config["model"].update(
        boundaryCoordinateStretch=2.5,
        boundarySpongeStrength=0.3,
        boundarySpongeFraction=0.30,
    )
    _, cfg, _ = build_config(config)
    theta_ops = build_theta_operators(cfg)
    boundary = build_orbit_boundary(cfg, theta_ops)

    expected = np.zeros((cfg.nth, cfg.nth - 2))
    expected[np.arange(1, cfg.nth - 1), np.arange(cfg.nth - 2)] = 1
    left = (theta_ops.theta[0] - theta_ops.theta[1]) / (
        theta_ops.theta[2] - theta_ops.theta[1]
    )
    right = (theta_ops.theta[-1] - theta_ops.theta[-2]) / (
        theta_ops.theta[-3] - theta_ops.theta[-2]
    )
    expected[0, :2] = [1 - left, left]
    expected[-1, -2:] = [right, 1 - right]
    np.testing.assert_allclose(boundary.fieldProlongation, expected, atol=0, rtol=0)

    config["model"]["boundaryCoordinateStretch"] = 1.0
    _, base_cfg, _ = build_config(config)
    base = build_theta_operators(base_cfg)
    half = 0.5 * (cfg.thmax - cfg.thmin)
    radius = np.abs(base.theta / half)
    ramp = np.maximum((radius - 0.70) / 0.30, 0)
    expected_theta = half * np.sign(base.theta) * (radius + 1.5 * ramp**4)
    np.testing.assert_allclose(theta_ops.theta, expected_theta, rtol=2e-14, atol=2e-14)

    stream = np.ones_like(theta_ops.theta)
    layer_start = 0.70 * half
    layer_width = np.max(np.abs(theta_ops.theta)) - layer_start
    sponge_ramp = np.maximum((np.abs(theta_ops.theta) - layer_start) / layer_width, 0)
    expected_sponge = 5 * 0.3 / layer_width * sponge_ramp**4
    np.testing.assert_allclose(
        _passing_sponge(cfg, theta_ops.theta, stream),
        expected_sponge,
        rtol=2e-14,
        atol=2e-14,
    )


def test_temperature_scan_reuses_orbit_factorization_with_cpu_warm_start():
    config = mirror_config()
    mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.2
    result = mgk.solve(config)
    assert result.blockSolverInfo.kineticCacheHit
    assert result.blockSolverInfo.numFactorizedPages == 0
    assert result.blockSolverInfo.warmStartUsed


def test_cached_cpu_continuation_uses_hot_arpack_subspace():
    config = mirror_config()
    config["solver"]["eigenSubspaceDimension"] = 8
    first = mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.2
    config["solver"]["eigenInitialVector"] = first.reducedMode
    result = mgk.solve(config)
    assert result.blockSolverInfo.kineticCacheHit
    assert result.eigsInfo.warmStartUsed
    assert result.eigsInfo.hotSubspaceUsed
    assert result.eigsInfo.configuredSubspaceDimension == 8
    assert result.eigsInfo.subspaceDimension == 5
    assert result.eigsInfo.operatorCalls >= 6
    assert result.eigenResidual < 1e-7


def test_cached_cpu_continuation_accepts_secant_warm_ritz():
    config = mirror_config()
    config["geometry"] = {"ballooningAngle": 0.017}
    config["solver"].update(
        eigenTolerance=1e-10,
        eigenSubspaceDimension=30,
        eigenHotSubspaceDimension=30,
        compactResult=True,
    )
    eta0 = 0.425 / 0.17
    first = mgk.solve(config)
    step = 2e-5
    config["physical"]["ionTemperatureGradientLength"] = 0.425 / (eta0 + step)
    config["solver"]["eigenInitialVector"] = first.reducedMode
    second = mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.425 / (eta0 + 2 * step)
    config["solver"]["eigenInitialVector"] = second.reducedMode
    third = mgk.solve(config)
    assert third.blockSolverInfo.kineticCacheHit
    assert third.eigsInfo.warmRitzAttempted
    assert third.eigsInfo.warmRitzAccepted
    assert third.eigsInfo.warmRitzOperatorCalls == 3
    assert third.eigsInfo.arpackOperatorCalls == 0
    assert third.eigenResidual <= third.eigsInfo.warmRitzResidualTolerance
    config["physical"]["ionTemperatureGradientLength"] = 0.425 / (eta0 + 3 * step)
    config["solver"]["eigenInitialVector"] = third.reducedMode
    fourth = mgk.solve(config)
    assert fourth.eigsInfo.warmRitzAccepted
    assert fourth.eigsInfo.warmRitzDimension == 2
    assert fourth.eigsInfo.warmRitzOperatorCalls == 2
    assert fourth.eigsInfo.arpackOperatorCalls == 0
    assert fourth.eigenResidual <= fourth.eigsInfo.warmRitzResidualTolerance


def test_cached_cpu_continuation_rejects_inaccurate_warm_ritz():
    config = mirror_config()
    config["geometry"] = {"ballooningAngle": 0.023}
    config["solver"].update(
        eigenTolerance=1e-10,
        eigenSubspaceDimension=30,
        eigenHotSubspaceDimension=30,
        compactResult=True,
    )
    eta0 = 0.425 / 0.17
    first = mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.425 / (eta0 + 1e-3)
    config["solver"]["eigenInitialVector"] = first.reducedMode
    second = mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.425 / (eta0 + 2e-3)
    config["solver"]["eigenInitialVector"] = second.reducedMode
    third = mgk.solve(config)
    assert third.eigsInfo.warmRitzAttempted
    assert not third.eigsInfo.warmRitzAccepted
    assert third.eigsInfo.warmRitzOperatorCalls == 3
    assert third.eigsInfo.arpackOperatorCalls > 0
    assert third.eigenResidual < 1e-9


def test_disabling_factorization_cache_disables_orbit_cache_keys():
    config = mirror_config()
    config["geometry"] = {"ballooningAngle": 0.013}
    config["solver"]["enableFactorizationCache"] = False
    first = mgk.solve(config)
    config["physical"]["ionTemperatureGradientLength"] = 0.2
    second = mgk.solve(config)
    assert not first.blockSolverInfo.kineticCacheHit
    assert not second.blockSolverInfo.kineticCacheHit
    assert second.blockSolverInfo.numFactorizedPages == second.blockSolverInfo.numBlocks
    assert not second.blockSolverInfo.warmStartUsed


def test_orbit_kinetic_electron_aparallel_matches_reference():
    config = mirror_config()
    config["physical"]["electronBeta"] = 0.01
    config["model"]["aparallel"] = True
    config["species"] = {
        "enabled": True,
        "items": {"kind": "electron", "kinetic": True},
    }
    config["grid"]["numTheta"] = 17
    config["solver"]["eigenSubspaceDimension"] = 30
    result = mgk.solve(config)
    expected = -0.812491367787435 - 1.2054619978441e-5j
    assert abs(result.omega - expected) < 1e-10
    assert len(result.aparallel) == 17
    assert result.eigenResidual < 1e-8


def test_miller_open_orbit_matches_current_reference():
    config = {
        "physical": {"majorRadius": 1.0, "minorRadius": 0.5},
        "geometry": {
            "model": "miller", "q": 2.0, "magneticShear": 1.0,
            "elongation": 1.5, "triangularity": 0.2,
        },
        "model": {"magneticMirror": True, "parallelBoundary": "open"},
        "grid": {"numTheta": 33, "numEnergy": 2, "numPitch": 4,
                 "numBouncePoints": 8},
        "solver": {
            "useGpu": False, "blockPrecision": "double",
            "eigenTolerance": 1e-7, "eigenSubspaceDimension": 20,
            "singleShiftTimeLimit": 60,
            "enableFactorizationCache": False,
            "enableGpuResultCache": False,
        },
    }
    result = mgk.solve(config)
    expected = -0.480732409827677 + 0.00115320076449857j
    assert abs(result.omega - expected) < 1e-7
    assert result.blockSolverInfo.numBlocks == 20
    assert result.fieldConstraintResidual < 1e-8


def test_bparallel_open_orbit_matches_current_reference():
    config = mirror_config()
    config["physical"]["electronBeta"] = 0.01
    config["model"]["bparallel"] = True
    config["grid"]["numTheta"] = 17
    config["grid"]["numEnergy"] = 2
    config["grid"]["numPitch"] = 2
    config["solver"].update(eigenTolerance=1e-10, eigenSubspaceDimension=30)
    result = mgk.solve(config)
    expected = -0.842557973322923 - 3.1352670916418e-5j
    assert abs(result.omega - expected) < 1e-10
    assert result.bparallel is not None
    assert result.fieldConstraintResidual < 1e-10


def test_combined_aparallel_bparallel_uses_field_ordering():
    config = mirror_config()
    config["physical"]["electronBeta"] = 0.01
    config["model"].update(aparallel=True, bparallel=True)
    config["grid"].update(numTheta=17, numEnergy=2, numPitch=2)
    config["solver"].update(eigenTolerance=1e-9, eigenSubspaceDimension=20)
    result = mgk.solve(config)
    assert result.fields == ["phi", "aparallel", "bparallel"]
    assert len(result.aparallel) == len(result.phi)
    assert len(result.bparallel) == len(result.phi)
    assert result.eigenResidual < 1e-8
    assert result.fieldConstraintResidual < 1e-8
