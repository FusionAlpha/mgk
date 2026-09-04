# Rewoldt 2007 Figure 1: Cross-Code Check

This case reproduces the electrostatic, adiabatic-electron ITG scan from
Rewoldt et al., *Computer Physics Communications* 177 (2007) 775--780 and
compares MGK1D with the published FULL/GTC/GT3D curves.

## Model and normalization

- `r/R=0.18`, `r/a=0.5`, `q=1.4`, `s_hat=0.776`
- `R/Ln=2.22`, `R/LTi=R/LTe=6.92`, `Ti/Te=1`, `beta=0`
- `k_theta rho_i = 0.1, 0.2, ..., 0.6`
- collisionless electrostatic ITG, kinetic ions, adiabatic electrons
- MGK frequencies are converted from `c_s/R` to the paper's `c_s/Ln`

## Files

- `figures/rewoldt2007_fig1_mgk_vs_published_codes.png`: growth-rate and
  frequency comparison.
- `figures/rewoldt2007_fig2_mgk_overlay.png`: second published-figure overlay.
- `data/mgk_results.csv` and `data/mgk_results.json`: MGK values, residuals,
  and timing breakdown.
- `data/figure1_cross_code_comparison.csv`: error and published-spread checks.
- `data/paper_digitized_points.csv`: digitized reference points and uncertainty.
- `scripts/run_rewoldt2007_figure1.py`: MGK scan.
- `scripts/compare_with_published_codes.py`: comparison and plotting helper.

## Reproduce

```bash
PYTHONPATH=. python3 research_results/case_studies/rewoldt2007_figure1_cross_code/scripts/run_rewoldt2007_figure1.py --quick
```

Remove `--quick` for the production grid. The comparison helper additionally
needs the separate CGYRO scan referenced in its `--cgyro` argument; the saved
CSV already records the comparison used for the checked-in figure.
