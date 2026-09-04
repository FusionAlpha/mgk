"""Run all migrated paper physics cases and check historical checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from case_runner import CASE_DIRECTORIES, run_case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument(
        "--full", action="store_true",
        help="run complete scans instead of the representative regression points",
    )
    args = parser.parse_args()
    failed = []
    status = {}
    for case_id, name in CASE_DIRECTORIES.items():
        print(f"\n=== {case_id} {name} ===", flush=True)
        passed = run_case(case_id, args.backend, args.full)
        status[case_id] = {"name": name, "passed": passed}
        if not passed:
            failed.append(case_id)
    summary_file = Path(__file__).resolve().parent / "validation_summary.json"
    summary_file.write_text(
        json.dumps(
            {
                "backend": args.backend,
                "fullScan": args.full,
                "passed": not failed,
                "cases": status,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(f"paper case regression failed: {', '.join(failed)}")
    print("\nAll paper physics cases passed.")


if __name__ == "__main__":
    main()
