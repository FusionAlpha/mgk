# MGK1D benchmark cases, 2026-08-28 to 2026-08-31

This document records the exact contracts used for the recent Python
CPU/GPU performance and electromagnetic accuracy runs. A timing result is
comparable only when the physics, grid, branch tracking, precision, cache, and
anchor-exclusion rules below are unchanged.

## Common execution environment

The complete performance grids were run on `liustation`
(`162.105.151.196`) with one NVIDIA GeForce RTX 5090 (32 GB, compute
capability 12.0).  Recorded Python GPU runs used Python 3.12.3, NumPy 2.5.2,
CuPy 14.2.0, 16 CPU factorization workers, 8 operator workers, and one
OpenBLAS/OMP/MKL thread per worker.

The public benchmark backends are Python CPU double, Python GPU double, and
Python GPU single.

Double precision uses an eigensolver tolerance of `1e-8`; GPU single uses
`1e-6`.  GPU timings synchronize the device before starting and after the
solve.  Result-cache reuse is disabled.  Dynamic scans use the preceding mode
as the next initial vector but do not reuse a factorization when the shift
changes.

## 1. Adiabatic-electron constant-B performance case

This is the left panel of the combined electrostatic performance figure.

| Quantity | Value |
| --- | --- |
| geometry | s-alpha, `q=1`, magnetic shear `s_hat=1`, `alpha=0` |
| fields | electrostatic `[phi]` |
| electron closure | adiabatic |
| magnetic mirror | enabled, `cgyro_s_alpha` convention |
| major radius | `R=1.7 m` |
| magnetic field | `B=2 T` |
| temperatures | `Ti=Te=1000 eV` |
| main-ion mass/charge | `2 amu`, `Z=1` |
| gradients | `Ln=R/4`; `LTi=Ln/eta_i` |
| scan | `eta_i = 2.3, 2.4, 2.5, 2.6, 2.7` |
| theta domain/boundary | `[-4 pi,4 pi]`, periodic |
| theta grids | `Ntheta=33,49,65,81,97,129`; `thetaMapAlpha=3` |
| velocity grid | `Emax=12.5`, `NE=16`, `Nlambda=4`, `Nbounce=8` |
| eigensolver | nearest shift-invert; subspace `4`; max iterations `1200` |
| continuation | anchor at `eta_i=2.5`; preceding frequency and mode |
| timing statistic | mean wall time of four changed points; anchor excluded |

The shift is updated at every eta point.  Consequently the historical CPU
and GPU runs report zero factorization-cache hits.  A fixed-shift A/B run is
not part of the official figure because it changes this timing contract.

The Python CPU/GPU runs use the same physical inputs and continuation policy.
At `Ntheta=129`, optimized Python GPU double/single timings were
`0.019525/0.009405 s` per changed point.

Companion analysis-workspace runner (not installed with the Python package):

```text
analysis/paper_electrostatic_performance/run_python_adiabatic_constant_b_orbit.py
```

## 2. Kinetic-electron CBC performance case

This is the right panel of the combined electrostatic performance figure.

| Quantity | Value |
| --- | --- |
| geometry | s-alpha, `q=1.4`, `s_hat=0.8`, `alpha=0` |
| fields | electrostatic `[phi]` |
| species | kinetic main ion plus kinetic electron |
| major/minor radius | `R=1.0 m`, `a=0.18 m` |
| magnetic field | `B=2 T` |
| temperatures | `Ti=Te=1000 eV` |
| main-ion mass/charge | `2 amu`, `Z=1` |
| gradients | `R/Ln=2.2`, `R/LTi=R/LTe=6.9` |
| beta | `beta_e=0` |
| scan | 29 `ky rho_s` points: `0.10:0.05:0.50`, then `0.525:0.025:1.00` |
| theta domain/boundary | `[-4 pi,4 pi]`, open |
| theta grids | `Ntheta=33,49,65,81,97,129`; `thetaMapAlpha=0` |
| velocity grid | `Emax=12.5`, `NE=16`, `Nlambda=24`, `Nbounce=24` |
| eigensolver | nearest shift-invert; subspace `8`; max iterations `1200` |
| branch anchors | ITG at `ky rho_s=0.30`; TEM at `0.60` |
| continuation | preceding mode plus dynamic two-point frequency extrapolation |
| timing statistic | mean wall time of 27 changed points; both anchors excluded |

There are 174 points per backend. At `Ntheta=129`, optimized Python GPU
double/single timings were `0.080272/0.040572 s` per changed point.

Companion analysis-workspace runner (not installed with the Python package):

```text
analysis/paper_electrostatic_performance/run_python_kinetic_cbc_orbit.py
```

## 3. Shen-2025-inspired electromagnetic performance case

The source parameter family is Y. Shen et al., *Nuclear Fusion* 65 (2025)
086026, doi: `10.1088/1741-4326/aded22`.

| Quantity | Value |
| --- | --- |
| geometry | s-alpha, `q=2`, `s_hat=1`, ballooning angle `0` |
| fields, two-field | `[phi, A_parallel]` |
| fields, three-field | `[phi, A_parallel, B_parallel]` |
| species | kinetic main ion plus kinetic electron |
| major/minor radius | `R=1.0 m`, `a=0.0018 R` |
| magnetic field | `B=2 T` |
| temperatures | `Ti=Te=1000 eV` |
| ion mass/charge | `1837 me`, `Z=1` |
| gradients | `Ln=LTi=LTe=0.2 R`; therefore `eta_i=eta_e=1` |
| wavenumber | `k_theta rho_s=0.30` |
| beta scan | `beta_e=0.018,0.019,0.020,0.021,0.022` |
| pressure gradient | `alpha=80 beta_e` (`alpha=1.6` at the anchor) |
| theta domain/boundary | `[-5 pi,5 pi]`, periodic |
| theta grids | `Ntheta=33,49,65,81,97,129`; `thetaMapAlpha=0` |
| velocity grid | `Emax=12.5`, `NE=16`, `Nlambda=24`, `Nbounce=24` |
| CPU eigensolver | subspace `24`, max iterations `6000` |
| GPU eigensolver | subspace `16`, max iterations `1200`, max restarts `100` |
| continuation | beta `0.020` anchor, then preceding frequency and mode |
| timing statistic | mean wall time of four changed points; anchor excluded |

The GPU production curves use seeded native multi-field assembly.  All 60
points in each GPU precision report `gpuAssembly=True`.  For trusted CPU
reference points, the maximum relative complex-frequency difference is
`1.36e-13` for GPU double and `3.26e-5` for GPU single.  Direct two-/three-field
assembly validation gave CPU/GPU frequency differences of about `1.9e-11`
and `4.7e-14`, respectively.

This is a Shen-inspired MGK performance case, not an exact reproduction of
the paper's HD7 model.  The paper uses kinetic ions and massless electrons
with an integral Rayleigh-Ritz method; this benchmark uses the MGK kinetic-ion,
kinetic-electron orbit discretization and MGK CPU/GPU backends.

Companion Python runner and plotter (not installed with the Python package):

```text
analysis/paper_electromagnetic_performance/run_python_shen_scan_performance.py
analysis/paper_electromagnetic_performance/plot_six_backend_performance.py
```

`run_python_shen_scan_performance.py` produces the Python summary CSV files.
The performance PDF is generated separately by
`plot_six_backend_performance.py` from the retained benchmark tables.

## 4. Xie-2016 electromagnetic CBC/KBM accuracy case

This case compares MGK with Xie et al. (2016) and the current CGYRO local
reference.

| Quantity | Value |
| --- | --- |
| geometry | s-alpha, `q=1.4`, `s_hat=0.78`, `alpha=0` |
| fields | `[phi, A_parallel]` (`phi` only at exactly zero beta) |
| species | kinetic main ion plus kinetic electron |
| major/minor radius | `R=0.835 m`, `a=0.18 R` |
| magnetic field | `B=2 T` |
| temperatures | `Ti=Te=2200 eV` |
| ion mass/charge | `1837 me`, `Z=1` |
| gradients | `R/Ln=2.2`, `R/LTi=R/LTe=6.9` |
| wavenumber | `k_theta rho_i=0.22` |
| beta scan | `beta_e=0:0.001:0.020` |
| theta domain | `[-4 pi,4 pi]` |
| production boundary | `open-extrapolated`, stretch `2.5`, sponge strength `0.3`, sponge fraction `0.30` |
| production grid | `Ntheta=97`, `Emax=8`, `NE=12`, `Nlambda=24`, `Nbounce=48` |
| Python verification backend | CPU double, tolerance `1e-8`, subspace `80`, max iterations `1200` |

MGK tracks ITG, TEM, and KBM separately. The KBM anchor is `beta_e=0.015`
with seed `-2.0+0.95i`; the reported dominant branch is the largest growth
rate among valid branches. Each beta point is continued from the preceding
frequency and mode, with `eigenResidual<1e-8` and `fieldResidual<1e-10`.

For the KBM interval `beta_e=0.012--0.020`, Python residuals are about
`4e-14` for the eigenproblem and
`2e-16` for the field constraint.

A separate hard-open mode comparison uses `Ntheta=65` and no stretch/sponge.
At `beta_e=0.015`, the saved mode has a well-resolved peak and this hard-open
result must not be compared as though it were the `Ntheta=97`
absorbing-boundary production case.

Repository scripts and derived files:

```text
analysis/xie2016_em_cbc_python_scan.py
analysis/xie2016_em_mode_compare.py
analysis/plot_xie2016_kbm_accuracy.py
analysis/xie2016_kbm_accuracy.csv
figures/electromagnetic_xie2016_kbm/xie2016_kbm_accuracy_comparison.pdf
```

## Reproduction checklist

Every future result directory should retain:

- the exact source revision or package path;
- host, CPU allocation, GPU model, driver, CUDA, Python, NumPy/SciPy,
  and CuPy versions;
- the complete physical, geometry, species, grid, and solver dictionaries;
- branch anchors, scan order, frequency extrapolation, and initial-vector
  policy;
- precision, tolerance, subspace dimension, iteration/restart limits;
- factorization/result-cache switches and cache-hit counts;
- whether `gpuAssembly` is true;
- cold-anchor and changed-point timings separately;
- frequency, growth rate, eigen residual, and field residual at every point.

Algebraic residuals establish only that the discretized eigenproblem was
solved.  They do not replace theta-domain, theta-grid, energy, pitch, bounce,
and energy-cutoff convergence studies.
