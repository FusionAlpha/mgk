# Typical Case Studies

This directory is the readable entry point for representative MGK1D results.
Each case keeps the same small layout:

```text
case_name/
  README.md       physical model, grid, provenance, and commands
  figures/        report-ready PNG figures
  data/           CSV/JSON values used by the figures
  scripts/        reproduction or source-path scripts
```

The four cases below cover the main scientific workflows currently reported
for the Python solver:

| Case | What it demonstrates |
| --- | --- |
| [Electrostatic CBC adiabatic ITG](electrostatic_cbc_adiabatic_itg/) | Adiabatic-electron ITG dispersion and residual checks |
| [Electrostatic CBC kinetic ITG/TEM](electrostatic_cbc_kinetic_itg_tem/) | Separate kinetic-electron ITG and TEM branch tracking |
| [Rewoldt 2007 Figure 1](rewoldt2007_figure1_cross_code/) | Cross-code comparison against published FULL/GTC/GT3D curves |
| [Xie 2016 electromagnetic KBM](electromagnetic_xie2016_kbm/) | Electromagnetic KBM accuracy, boundary checks, and residuals |

All copied results were generated from repository revision
`cef89e4c3a0edaff06942140754706b3a60ef0b3` (2026-09-01 working release).
The CSV/JSON files are the data actually used for the displayed figures; do
not digitize the PNGs again.

## Historical archive

The original, larger research workspace remains under `analysis/` and the
legacy figure collection remains under `figures/` in the local research tree.
Those directories contain intermediate scans, logs, and machine-specific
benchmark output and are intentionally not part of the installable Python
package. The case-study copies provide a stable, visible summary.

For installation and solver usage, start at [README.md](../../README.md).
For the exact benchmark contracts, see [BENCHMARK_CASES.md](../../BENCHMARK_CASES.md).
