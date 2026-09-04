"""Species-aware convenience entry point."""

from __future__ import annotations

from copy import deepcopy

from mgk._struct import Struct, as_struct
from mgk.solver import solve as _solve


def solve(config=None):
    cfg = as_struct(deepcopy({} if config is None else config))
    if "species" not in cfg or cfg.species is None:
        cfg.species = Struct()
    cfg.species.enabled = True
    return _solve(cfg)


__all__ = ["solve"]
