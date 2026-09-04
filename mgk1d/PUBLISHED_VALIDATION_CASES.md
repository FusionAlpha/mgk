# Published validation cases

This document maps the validation cases in Luo *et al.*, “Sub-Second
Collisionless Gyrokinetic Eigenvalue Solutions via Orbit-Invariant
Decomposition,” arXiv:2608.17418, to MGK1D 0.1.2 inputs. It separates facts
stated in the paper from the grids selected for the current Python product
reproduction. Executable configurations are in
`mgk/examples/validation/published_cases.py`; reference checkpoints are in
`mgk/examples/validation/published_reference_results.json`.

## Common conventions

- Normal modes use `exp(-i omega t)`. Negative real frequency is the ion
  diamagnetic direction; positive real frequency is the electron diamagnetic
  direction.
- The model is local, collisionless, linear, and electrostatic. All cases use
  `beta_e=0`, `aparallel=false`, and `bparallel=false`.
- Unless stated otherwise, MGK frequencies are `omega*R0/v_th,i`, with
  `v_th,i=sqrt(T_i/m_i)`. CBC/Miller values expressed as `omega*a/c_s` use
  `a/R0=0.36` for the Rewoldt CBC or `a/R0=1/3` for the Miller triangularity
  case. Here `c_s=v_th,i` because `Ti=Te` and the ions are singly charged.
- The examples choose `R0=1 m`, `B0=2 T`, deuterium ions, and `Ti=Te=1 keV`.
  These values only realize the dimensionless inputs; they are not
  experimental operating-point claims.
- Passing particles use an open inflow boundary in the Python reproduction;
  trapped particles close at bounce points.

The paper does not list every velocity-grid order for every physics figure.
Grid entries labelled “product grid” below are therefore explicit MGK1D 0.1.2
reproduction settings, not unpublished original-author inputs.

## Local stellarator profile interface

The published cases above are s-alpha/Miller validation cases.  MGK1D also
accepts a preprocessed periodic `.npz` field-line profile with the schema
documented in `README.md`. This is an interface and numerical-chain
smoke test for a fixed local stellarator field line; it is not a claim that the
paper's axisymmetric benchmarks validate a full three-dimensional VMEC/GVEC
calculation.  A future equilibrium-conversion tool should record the source
equilibrium, surface label, field-line label, normalization, and sampling
resolution alongside the profile.

## Case 1 — adiabatic-electron ITG eta scan (paper Fig. 1)

| Quantity | Value |
|---|---:|
| Geometry | circular s-alpha |
| `q`, `s_hat`, `alpha`, `theta0` | `1`, `1`, `0`, `0` |
| `r/R0` | `0.05` |
| `Ln/R0` | `0.25` |
| `Te/Ti` | `1` |
| `ky*rho_i` | `0.45/sqrt(2) = 0.318198...` |
| Electron model | adiabatic |
| Scan | `eta_i=Ln/LTi = 2.3, 2.4, 2.5, 2.6, 2.7` |
| Product grid | `theta in [-5pi,5pi]`, `Ntheta=97`, `NE=16`, `Nlambda=24`, `Nb=32`, `Emax/Ti=12.5` |

The paper reports a maximum MGK/CGYRO relative complex-frequency difference
below 2%, with continuous frequency and parallel structure along one
ion-diamagnetic ITG branch. At `eta_i=2.5`, the current Python CPU-double
checkpoint is:

```text
omega*R0/v_th,i = -0.7967676272 + 0.3416034428 i
```

The paper describes its ion calculation as restricted to passing orbits.
MGK1D 0.1.2's public orbit backend integrates the full passing/trapped pitch
population and does not expose the removed legacy no-mirror/grid switch. This
product case is therefore a maintained regression counterpart, not a claim of
bit-for-bit reconstruction of that unpublished population restriction.

## Case 2 — kinetic-electron CBC ITG/TEM competition (paper Fig. 2)

| Quantity | Value |
|---|---:|
| Geometry | circular s-alpha |
| `q`, `s_hat`, `alpha`, `theta0` | `1.4`, `0.776`, `0`, `0` |
| `r/R0` | `0.18` |
| `R0/Ln` | `2.22` |
| `R0/LTi`, `R0/LTe` | `6.92`, `6.92` |
| `Ti/Te` | `1` |
| Mass ratio | physical electron/deuterium-ion mass ratio |
| Electron model | kinetic; passing and trapped populations retained |
| Paper scan | 29 points: `0.10:0.05:0.50`, then `0.525:0.025:1.00` in `ky*rho_s` |
| Product grid | `theta in [-4pi,4pi]`, `Ntheta=97`, `NE=16`, `Nlambda=32`, `Nb=48`, `Emax/T=12.5` |

ITG and TEM must be continued separately. Selecting only the fastest-growing
root causes an apparent real-frequency sign change near `ky*rho_s=0.6`; this
is a branch switch, not reversal of one mode. The archived plotted path used
GPU single precision. Two archived `omega*R0/c_s` checkpoints are:

| Point | Paper-path checkpoint |
|---|---:|
| ITG, `ky*rho_s=0.30` | `-0.86981636 + 0.49619199 i` |
| TEM, `ky*rho_s=0.60` | `+0.94887610 + 0.19643729 i` |

The current CPU-double `ky*rho_s=0.30` checkpoints are
`-0.86977156+0.49625707i` (ITG) and `+0.47885143+0.19335488i` (TEM). Single
precision is retained only as historical paper-path evidence; customer or
publication results must be confirmed in double precision.

The companion mode-structure case in Fig. 2(c) keeps the same geometry,
density gradient, and electron-temperature gradient but sets
`R0/LTi=2.22` (`eta_i=1`) and `k_theta*rho_i=0.335`. The unstable branch is a
TEM and is compared after setting `phi(0)` real and positive.

## Case 3 — strong electron-gradient kinetic mode (paper Fig. 3)

| Quantity | Value |
|---|---:|
| Geometry | circular s-alpha |
| `q`, `s_hat`, `alpha`, `theta0` | `1.4`, `0.776`, `0`, `0` |
| `r/R0` | `0.18` |
| `Ln/R0` | `0.018` |
| `eta_i`, `eta_e` | `0`, `3.13` |
| `R0/LTe` | approximately `174` |
| `ky*rho_i` | `0.7` |
| Electron model | kinetic, physical mass ratio |
| Product grid | `theta in [-8pi,8pi]`, `Ntheta=193`, `NE=16`, `Nlambda=32`, `Nb=48`, `Emax/T=12.5` |

The paper gives `omega*R0/v_th,i ~= 23.3+13.8i` for MGK and
`23.4+13.8i` for CGYRO, a complex-frequency difference of about 0.3%. The
current Python CPU-double result is:

```text
omega*R0/v_th,i = 23.37427403 + 13.77882357 i
```

Mode structures are compared on `-8pi <= theta <= 8pi` after independent peak
normalization and phase alignment.

## Case 4 — Miller triangularity scan (paper Fig. 4)

| Quantity | Value |
|---|---:|
| Geometry | local Miller, GACODE input convention |
| `R0/a`, `r/a` | `3`, `0.5` |
| `q`, `s_hat`, `alpha` | `2`, `1`, `0` |
| `a/Ln`, `a/LTi` | `1`, `3` |
| `ky*rho_s` | `0.3` |
| `Ti/Te` | `1` |
| `kappa`, `s_kappa`, `s_delta`, `Delta'` | `1`, `0`, `0`, `0` |
| Scan | `delta = -0.4, -0.2, 0, 0.2, 0.4` |
| Electron model | adiabatic |
| Product grid | `theta in [-4pi,4pi]`, `Ntheta=97`, `NE=16`, `Nlambda=32`, `Nb=48`, `Emax/T=12.5` |

The ITG branch is continued separately from the circular point toward positive
and negative triangularity. The paper reports a maximum MGK/CGYRO relative
complex-frequency difference below 1.5%. At `delta=0.2`, the current Python
CPU-double checkpoint is:

```text
omega*a/c_s = -0.2222502655 + 0.1763659947 i
```

This case also regresses the complete Miller magnetic-drift term; using only
the alpha-zero circular drift expression is not equivalent.

## Case 5 — extended kinetic-electron TEM in circular Miller geometry (paper Fig. 5)

This case uses the Fig. 2(c) Rewoldt companion inputs (`q=1.4`,
`s_hat=0.776`, `r/R0=0.18`, `R0/Ln=R0/LTi=2.22`, `R0/LTe=6.92`,
`Ti=Te`, physical mass ratio, `ky*rho_i=0.335`) but evaluates the geometry
through the circular Miller representation. The product grid is
`theta in [-5pi,5pi]`, `Ntheta=121`, `NE=16`, `Nlambda=32`, `Nb=48`, and
`Emax/T=12.5`, with an open passing-particle boundary.

| Result | `omega*a/c_s` |
|---|---:|
| MGK, paper | `0.1870 + 0.2100 i` |
| MGK1D 0.1.2 CPU double | `0.18696111 + 0.20965143 i` |
| CGYRO, `Nr=9` | `0.1915 + 0.2047 i` |
| CGYRO, `Nr=5` | `0.2000 + 0.2535 i` |

Relative to `Nr=9`, the paper reports a 2.5% complex-frequency difference and
a phase-aligned potential overlap of `0.9991`. The normalized outer-turn
maximum falls from `0.119` at `Nr=5` to `6.4e-3` at `Nr=9`, demonstrating that
the shorter CGYRO domain is boundary sensitive.

## Performance workloads (paper Fig. 6)

The performance figure uses the five-point Fig. 1 eta scan and the 29-point
CBC scan above, with rounded CBC inputs `s_hat=0.8`, `R0/Ln=2.2`, and
`R0/LTi=R0/LTe=6.9`. `Ntheta` is varied over `33, 49, 65, 81, 97, 129` at a
fixed velocity grid. The CPU-double `Ntheta=129` result is the internal
resolution reference; both workloads stay below 1% mean frequency variation
for `Ntheta>=65`.

The timing machine in the paper had an Intel Core i7-14700HX and an NVIDIA
GeForce RTX 4060 Laptop GPU. At `Ntheta=97`, mean changed-point times were:

| Workload | CPU double | GPU double | GPU single |
|---|---:|---:|---:|
| adiabatic eta scan | about 0.43 s | about 0.12 s | about 0.03 s |
| kinetic CBC scan | about 1.5 s | about 0.22 s | about 0.06 s |

These measurements exclude the initial anchor and include reassembly,
factorization, and eigensolution at every changed point. They are continuation
throughput on that hardware, not cold-start latency or a portable customer
performance guarantee. The paper does not state the fixed velocity-grid
orders in its text, so they must not be reconstructed from assumption.

## Reproduction and acceptance

From the source tree:

```bash
python mgk/examples/validation/published_cases.py salpha-adiabatic-itg
python mgk/examples/validation/published_cases.py cbc-itg
python mgk/examples/validation/published_cases.py cbc-tem
python mgk/examples/validation/published_cases.py strong-gradient-tem
python mgk/examples/validation/published_cases.py miller-triangularity-itg
python mgk/examples/validation/published_cases.py miller-extended-tem
```

Add `--backend gpu` for a CUDA double-precision run. Add `--json` to inspect a
configuration without solving.

An eigenpair is accepted only if both the eigen residual and field constraint
residual are at most `2e-6`. A production result additionally requires checks
in `Ntheta`, `NE`, `Nlambda`, bounce points, parallel extent, and energy cutoff.
Branch scans must track frequency and mode overlap; a nearest eigenvalue at an
isolated shift is not enough to prove branch identity.

The internal release evidence archive additionally records CPU physics/numerics
assessments, CPU/GPU parity, and repeatability. It is distinct from these five
historical paper cases and can be supplied as a separate validation dossier
rather than bundled in the runtime source distribution.
