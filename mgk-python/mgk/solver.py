"""Public solve orchestration and result reconstruction."""

from __future__ import annotations

from time import perf_counter

import numpy as np

from mgk._struct import Struct
from .internal.configuration import build_config
from .internal.cache import freeze

_RESULT_CACHE: tuple[Struct, Struct] | None = None


def _solve_configured(cfg) -> Struct:
    total_start = perf_counter()
    from .internal.orbit import solve_orbit
    return solve_orbit(cfg, total_start)


def solve(config=None) -> Struct:
    """Solve an MGK1D eigenproblem from a nested mapping or :class:`Struct`."""
    global _RESULT_CACHE
    call_start = perf_counter()
    input_cfg, normalized, normalization = build_config(config)
    use_cache = normalized.useGpu and normalized.enableGpuResultCache
    result_cache_key = freeze(input_cfg)
    if use_cache and _RESULT_CACHE is not None and _RESULT_CACHE[0] == result_cache_key:
        result = _RESULT_CACHE[1].deepcopy()
        result.runtime = perf_counter() - call_start
        result.cacheHit = True
        result.timing.cacheLookup = result.runtime
        return result
    result = _solve_configured(normalized)
    result.update(
        cfg=input_cfg, normalizedConfig=normalized, normalization=normalization,
        omegaPhysical=result.omega * normalization.frequency,
        frequencyHz=(result.omega * normalization.frequency).real / (2 * np.pi),
        growthRate=(result.omega * normalization.frequency).imag,
        cacheHit=False, fields=normalized.fields.names,
    )
    if normalized.species.enabled:
        result.species = normalized.species.items
    if use_cache:
        _RESULT_CACHE = (result_cache_key, result.deepcopy())
    return result
