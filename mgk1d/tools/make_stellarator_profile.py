#!/usr/bin/env python3
"""Create a small periodic MGK1D stellarator-profile example.

This utility intentionally generates a synthetic profile. Production users can
use ``tools/convert_vmec_geometry.py`` to sample a VMEC equilibrium directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_profile(output: Path, points: int = 256, period: float = 2 * np.pi) -> None:
    if points < 16:
        raise ValueError("points must be at least 16")
    z = np.linspace(0.0, period, points + 1)
    phase = 2 * np.pi * z / period
    # Smooth, asymmetric one-well demonstration profile.
    field = 1.0 + 0.25 * np.cos(phase) + 0.06 * np.sin(2 * phase)
    metric = 1.0 + 0.15 * np.sin(phase - 0.2)
    np.savez(
        output,
        z=z,
        period=period,
        B=field,
        kperpMetric=metric,
        driftCurvature=0.2 * np.cos(phase) + 0.03 * np.sin(3 * phase),
        driftParallelCorrection=0.02 * np.sin(phase),
        parallelGradient=1.0 + 0.05 * np.cos(phase),
        fieldLineLabel=np.array(0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="output .npz filename")
    parser.add_argument("--points", type=int, default=256,
                        help="number of intervals (default: 256)")
    args = parser.parse_args()
    make_profile(args.output, args.points)
    print(f"wrote synthetic MGK1D stellarator profile: {args.output}")


if __name__ == "__main__":
    main()
