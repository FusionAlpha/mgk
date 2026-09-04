# External release process

MGK1D release artifacts are built from the repository root with:

```bash
python tools/build_external_candidate.py
```

The normal command fails when `LICENSE` or matching PEP 639 metadata is
missing. During engineering review only, a clearly labelled technical release
candidate may be produced with:

```bash
python tools/build_external_candidate.py --allow-license-pending
```

The builder refuses to overwrite a non-empty output directory. Each candidate
contains a wheel, source distribution, standalone documentation/examples
bundle, SHA-256 manifest, CycloneDX engineering SBOM, build information, and
JSON/Markdown release audit. The audit checks
archive path safety, the distribution allowlist, local absolute-path/private-IP
leaks, metadata consistency, documentation, and license state.

Before external delivery, the release owner must:

1. install the company-approved `LICENSE` or EULA and PEP 639 metadata;
2. supply company identity, support contact, and contractual SLA;
3. review all direct and transitive dependency licenses and approve the SBOM;
4. run the compatibility matrix and archive the result;
5. verify the wheel in a clean environment and run the CPU/GPU smoke tests;
6. confirm the hashes after copying artifacts to the delivery system.

The generated SBOM records declared dependency ranges. A customer-specific
environment or container must additionally carry a resolved dependency lock
and an environment SBOM, including the actual CuPy/CUDA/driver components.
