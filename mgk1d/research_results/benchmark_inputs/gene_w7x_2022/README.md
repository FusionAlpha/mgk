# GENE W7-X 2022 benchmark inputs

These inputs target the linear electrostatic cases in Gonzalez-Jerez et al.,
*Electrostatic gyrokinetic simulations in Wendelstein 7-X geometry: benchmark
between the codes stella and GENE* (2022), DOI
`10.1017/S0022377822000393`, arXiv `2107.06060`.

The equilibrium currently used for local development is a VMEC++ 0.7.3
reconstruction of the paper source archive's `input.kjm30`. It is not an
official VMEC2000 output. The reconstructed file is intentionally not bundled
here because its provenance and convergence qualification must remain clear.

`parameters_test1_bean_itg` is the first corrected GENE initial-value input:

- `x0=sqrt(s0)=0.8`, `alpha=0`, `n_pol=1`;
- `a/Ln=1`, `a/LTi=3`, adiabatic electrons;
- `kx=0`, `ky*rhoi=2.1`, `Nz=256`, `Nvpar=36`, `Nmu=24`;
- `dt*vthi/a=0.14`, as reported for GENE in the paper;
- `q0=-1111`, so GENE reads `q=1/iota` from VMEC instead of mistaking the
  reported rotational transform `iota=0.910` for the safety factor.

Run from a temporary directory containing an `out` subdirectory, because GENE
writes diagnostics relative to the current working directory. The input uses
an absolute development path for the reconstructed equilibrium; update
`geomdir` when using an official VMEC2000 `wout.nc`.

These files are reproducibility inputs, not stored reference answers. A result
must satisfy GENE's `omega_prec` convergence exit before it is added to the
GENE/MGK comparison CSV.

On 2026-08-26, the fixed-step input and its automatic-step diagnostic variant
both showed repeated GENE amplitude rescaling and were rejected. Their exact
outcomes are recorded in
`analysis/gene_w7x_corrected_audit_20260826.csv`; this is an open numerical
setup issue, not a GENE/MGK agreement claim.
