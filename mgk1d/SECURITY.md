# Security

MGK1D is an offline numerical library.  It contains no telemetry, network
client, shell-command execution, dynamic code evaluation, or model download.
It accepts in-process Python mappings and returns in-memory numerical results.

The optional CuPy backend writes compiled CUDA kernel cache files to the
directory selected by `CUPY_CACHE_DIR`; MGK1D defaults this to a directory
under the operating system temporary directory.  Deployments with restricted
filesystem policy should set this variable to an approved per-user path.

Do not load untrusted Python configuration code.  MGK1D validates values passed
to `solve`, but Python code that constructs those values executes with the
caller's privileges.

Report suspected vulnerabilities through the support channel supplied with
the commercial distribution.  Do not include proprietary simulation inputs
or customer data in an initial report.
