# Third-party dependency notice

MGK is distributed by FusionAlpha under the BSD 3-Clause License. The complete
project license text is included in [LICENSE](LICENSE).

MGK1D 0.1.2 declares the following direct runtime dependencies.  They are not
vendored into the pure-Python MGK1D wheel.

| Dependency | Purpose | Upstream license (informational) |
|---|---|---|
| NumPy | arrays and dense linear algebra | BSD-3-Clause |
| SciPy | special functions, sparse operators, ARPACK interface | BSD-3-Clause |
| CuPy (optional) | CUDA arrays and GPU linear algebra | MIT |

CuPy CUDA wheels and the NVIDIA driver/runtime are subject to their own package
contents and vendor terms.  The company release process must archive the exact
dependency lock/constraints used for distribution and have legal/compliance
review the corresponding license texts.  This notice is an engineering
inventory, not legal advice and not a substitute for the final SBOM.

Test-only dependency: pytest (MIT).  Python itself is distributed under the
Python Software Foundation License when redistributed separately.
