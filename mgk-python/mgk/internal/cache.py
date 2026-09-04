"""Stable in-process cache keys for normalized solver structures."""

from __future__ import annotations

import hashlib

import numpy as np


TRANSIENT_FIELDS = frozenset({
    "eigsGuess", "eigsTolerance", "eigsSubspaceDimension",
    "eigsHotSubspaceDimension", "enableWarmRitz", "eigsMaxIterations",
    "eigsInitialVector", "singleShiftTimeLimit", "gpuArnoldiMaxRestarts",
    "blockFactorizationCheck", "enableFactorizationCache",
    "enableGpuFactorizationCache", "enableGpuResultCache",
    "compactResult", "returnMatrices", "eigenBackend", "modeSelection",
    "cpuFactorizationWorkers", "cpuOperatorWorkers",
})


def freeze(value, excluded=frozenset()):
    if isinstance(value, dict):
        return tuple((key, freeze(item, excluded)) for key, item in sorted(dict.items(value))
                     if key not in excluded)
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item, excluded) for item in value)
    if isinstance(value, np.ndarray):
        digest = hashlib.sha1(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return (value.shape, value.dtype.str, digest)
    if isinstance(value, np.generic):
        return value.item()
    return value
