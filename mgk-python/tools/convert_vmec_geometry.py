#!/usr/bin/env python3
"""Sample a VMEC ``wout.nc`` directly into an MGK field-line profile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mgk.vmec import write_vmec_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="VMEC wout.nc input")
    parser.add_argument("output", type=Path, help="MGK profile .npz output")
    parser.add_argument("--surface", type=float, default=0.64,
                        help="normalized toroidal-flux surface s (default: 0.64)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="straight-field-line label (default: 0)")
    parser.add_argument("--z-min", type=float, default=-4 * np.pi)
    parser.add_argument("--z-max", type=float, default=4 * np.pi)
    parser.add_argument("--points", type=int, default=513)
    args = parser.parse_args()
    path = write_vmec_profile(
        args.source, args.output, s=args.surface, alpha=args.alpha,
        z_min=args.z_min, z_max=args.z_max, num_points=args.points,
    )
    print(f"wrote direct VMEC MGK profile: {path}")


if __name__ == "__main__":
    main()
