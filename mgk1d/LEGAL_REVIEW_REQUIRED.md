# License decision required before external distribution

No company-approved MGK1D license or commercial EULA was present in the source
tree during the 2026-08-20 release audit. Engineering must not select a
license on behalf of the copyright owner.

Before any wheel, source archive, container, or customer bundle is delivered:

1. identify the legal copyright owner and approved product name;
2. choose and approve the commercial EULA or open-source license;
3. install the exact approved text as `LICENSE` (and `NOTICE` if required);
4. add the matching PEP 639 `license`/`license-files` metadata to
   `pyproject.toml`;
5. review direct and transitive dependency licenses and export an SBOM;
6. add the contractual support contact and SLA outside the code package.

Until these steps are complete, generated artifacts are technical release
candidates only and must not be externally distributed.
