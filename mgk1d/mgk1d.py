"""Deprecated compatibility alias for the renamed :mod:`mgk` package.

New code should use ``import mgk``.  This module remains intentionally thin so
older local scripts can migrate without duplicating the solver implementation.
"""

import mgk as _mgk

from mgk import *  # noqa: F401,F403

# Let legacy imports such as ``mgk1d.internal.geometry`` resolve into the new
# package tree while keeping ``mgk`` as the canonical package name.
__path__ = _mgk.__path__
__all__ = _mgk.__all__
