import numpy as np

import mgk


def test_small_default_solve_uses_orbit_backend():
    result = mgk.solve({
        "grid": {"numTheta": 33, "numEnergy": 6, "numPitch": 6},
        "solver": {
            "useGpu": False,
            "blockPrecision": "double",
            "eigenTolerance": 1e-9,
            "eigenSubspaceDimension": 20,
            "eigenMaxIterations": 500,
            "singleShiftTimeLimit": 60,
        },
    })
    assert result.discretization == "orbit"
    assert result.thetaBoundary == "passing-periodic/trapped-bounce-periodic"
    assert result.eigenResidual < 1e-7
    assert result.fieldConstraintResidual < 1e-7
    assert abs(result.phi[0] - result.phi[-1]) < 1e-12
    assert result.blockSolverInfo.numBlocks > 0
