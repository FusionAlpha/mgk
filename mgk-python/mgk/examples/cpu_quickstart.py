"""Small CPU-only MGK1D installation and solver smoke test."""

import mgk


result = mgk.solve({
    "grid": {
        "numTheta": 17,
        "numEnergy": 2,
        "numPitch": 2,
    },
    "solver": {
        "useGpu": False,
        "blockPrecision": "double",
        "eigenTolerance": 1e-9,
        "eigenSubspaceDimension": 20,
        "singleShiftTimeLimit": 60,
    },
})

print(f"omega = {result.omega.real:.12g} {result.omega.imag:+.12g}i")
print(f"eigen residual = {result.eigenResidual:.3e}")
print(f"field residual = {result.fieldConstraintResidual:.3e}")
print(f"used GPU = {result.usedGpu}")
