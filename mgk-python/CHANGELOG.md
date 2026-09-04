# Changelog

All notable externally supported changes are recorded here.

## 0.1.2 — 2026-08-20

- Public source distribution is now Python-only: legacy commercial-software
  implementation files, scripts, and comparison exports are kept locally for
  internal history but excluded from GitHub and release artifacts.
- Defined the external supported scope as local electrostatic ITG/TEM and
  orbit-only electromagnetic ITG/TEM/KBM with two- or three-field layouts.
- Added s-alpha and Miller geometry coverage with adiabatic/kinetic electrons.
- Corrected the nonzero-alpha s-alpha pressure-gradient drift contribution.
- Unified the Python implementation on the matrix-free orbit backend.
- Added double-precision CPU/GPU reference validation and branch tracking.
- Added package version export, curated external metadata, compatibility
  documentation, and release-audit tooling.
- Added documented and executable reproduction inputs for the validation cases
  published in arXiv:2608.17418.
- Added audited wheel/sdist construction, artifact path/content checks,
  SHA-256 manifests, and a CycloneDX engineering SBOM.
- Removed the retired grid backend from the Python public solve path.
- Added the first local-stellarator profile interface: periodic preprocessed
  field-line `.npz` input, per-well asymmetric trapped-bounce roots, and
  positive direct pitch-cosine quadrature.
- Added a direct VMEC `wout.nc` Fourier reader and local field-line sampler;
  GENE is no longer required to prepare a stellarator profile. Non-closing
  field-line samples are explicitly marked non-periodic and require an open
  parallel boundary.
- Corrected periodic stellarator topology so one-period domains do not count
  outside periodic copies as additional trapped wells, and disabled the
  s-alpha/Miller reflection-factorization shortcut for generic asymmetric
  stellarator profiles.
- Fixed the GPU factorization-cache switch and added a detailed NVIDIA GPU
  usage, precision, benchmarking, and troubleshooting guide. Added the exact
  contracts for the adiabatic, kinetic-CBC, Shen, and Xie-2016 validation runs.
- Moved electrostatic orbit assembly and field-response contractions to CuPy,
  added precision-aware GPU template/assembly caches, and reduced repeated
  response kernels in the GPU shift solver. Extended the native assembly path
  to electromagnetic `[phi, A_parallel]` and
  `[phi, A_parallel, B_parallel]` systems, including their gyro-average
  factors and derivatives.
- Added the visible `research_results/case_studies/` index with report-ready figures, the CSV/
  JSON values used to draw them, parameter summaries, and reproduction/source
  snapshots for four representative electrostatic and electromagnetic cases.

## Compatibility policy

Version 0.x may add configuration fields and diagnostics.  The public call
signature and documented configuration fields will not be removed within a
0.1.x patch release.  A removal requires a deprecation notice in at least one
minor release unless needed to correct a safety or scientific-validity issue.
