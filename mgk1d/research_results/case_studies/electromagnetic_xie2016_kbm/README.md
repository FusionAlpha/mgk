# Xie 2016: Electromagnetic KBM Accuracy

This case is the electromagnetic Cyclone Base Case used to check the KBM
branch, open-boundary treatment, and Python residuals against the local CGYRO
reference and published data.

## Model and grid

- `s-alpha`: `q=1.4`, `s_hat=0.78`, `alpha=0`
- kinetic ion plus real-mass kinetic electron
- two-field electromagnetic model `[phi, A_parallel]`
- `R=0.835 m`, `a=0.18 R`, `B=2 T`, `Ti=Te=2200 eV`
- `R/Ln=2.2`, `R/LTi=R/LTe=6.9`, `k_theta rho_i=0.22`
- beta scan `beta_e=0.00 ... 0.020`
- production boundary: open-extrapolated stretch `2.5`, sponge strength `0.3`
- production grid: `Ntheta=97`, `NE=12`, `Npitch=24`, `Nbounce=48`

## Files

- `figures/xie2016_kbm_accuracy_paper_style.png`: paper-style KBM branch
  comparison.
- `figures/xie2016_kbm_accuracy_comparison.png`: MGK/CGYRO accuracy.
- `figures/xie2016_kbm_boundary_comparison.png`: boundary sensitivity check.
- `figures/xie2016_em_mode_comparison.png`: open-boundary mode comparison.
- `data/kbm_accuracy.csv`: frequency, growth-rate, and relative-error table.
- `data/kbm_boundary_comparison.csv` and `data/kbm_spurious_check.csv`:
  boundary and spurious-mode diagnostics.
The saved Python residuals are approximately `1e-14` for the eigenproblem and
`1e-16` for the field constraint. The numerical error and boundary scope are
summarized in [BENCHMARK_CASES.md](../../../BENCHMARK_CASES.md).

## Reproduce or inspect

The CSV files can be inspected without any optional dependencies. To run a
fresh Python calculation, start from the configuration examples in
`mgk/examples/validation/published_cases.py` and save the returned frequency,
growth rate, residuals, and timing alongside the case data.
