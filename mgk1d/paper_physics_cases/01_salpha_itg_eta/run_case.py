"""Run the Python migration of paper case 01."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from case_runner import run_case


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run_case("01", args.backend, args.full) else 1)

