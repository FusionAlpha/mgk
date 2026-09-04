# Electrostatic CBC: Kinetic-Electron ITG and TEM

This case tracks the two physical branches separately. It avoids identifying
branches by the largest growth rate alone, which can swap ITG and TEM near a
crossing.

## Model and grid

- `s-alpha` geometry: `q=1.4`, `s_hat=0.8`, `alpha=0`
- kinetic main ion plus real-mass kinetic electron
- open ballooning boundary, `theta = [-4 pi, 4 pi]`
- `R/Ln=2.2`, `R/LTi=R/LTe=6.9`, `Ti=Te=1000 eV`, `B=2 T`
- `k_y rho_s = 0.05, 0.10, ..., 0.65`
- production grid: `Ntheta=97`, `NE=16`, `Npitch=32`, `Nbounce=48`
- continuation anchors: ITG at `k_y rho_s=0.30`, TEM at `0.60`

## Files

- `figures/cbc_kinetic_itg_tem_dispersion.png`: both branches and the
  dominant branch.
- `figures/cbc_kinetic_mode_structure_comparison.png`: kinetic-electron mode
  structure at `k_y rho_s=0.30`.
- `data/python_cpu_itg_tem_dispersion.csv`: branch values and residuals.
- `data/python_*_mode_ky0p30.csv`: the mode vectors used in the structure plot.
- `scripts/run_kinetic_itg_tem.py`: Python reproduction script.

## Reproduce

```bash
PYTHONPATH=. python3 research_results/case_studies/electrostatic_cbc_kinetic_itg_tem/scripts/run_kinetic_itg_tem.py
```

The script writes its result to this case's `data/` directory and uses the
previous eigenfrequency and reduced mode for continuation at each point.
The exact branch and timing contract is documented in
[BENCHMARK_CASES.md](../../../BENCHMARK_CASES.md).
