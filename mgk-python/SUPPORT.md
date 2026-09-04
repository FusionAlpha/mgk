# External support contract

Version 0.1.2 supports the documented electrostatic and electromagnetic
orbit-only Python API and the platform matrix in `COMPATIBILITY.md`.
Supported field layouts are `[phi]`, `[phi,A_parallel]`, and
`[phi,A_parallel,B_parallel]`; the latter two require positive `beta_e`.
Customers should report issues through their MGK1D distributor or account
representative and include:

- `mgk.__version__` and Python version;
- operating system, CPU, and (if applicable) GPU/driver/CuPy versions;
- the smallest configuration that reproduces the issue;
- eigen and field residuals;
- whether CPU double reproduces a GPU result.

The following are outside the 0.1.2 external support scope:

- collisions, rotation, global radial physics, and nonlinear evolution;
- massless-fluid electrons and the retired grid backend;
- electromagnetic models or field closures other than the documented
  orbit-only two- and three-field layouts;
- undocumented names under `mgk.internal`;
- interpretation of customer equilibria or scientific conclusions.

No service-level response time is stated in this repository.  Contractual SLA,
company identity, and support contact details must be supplied by the external
distribution agreement.
