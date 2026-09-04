# Compatibility matrix

## Supported target

| Component | Supported range |
|---|---|
| Python | 3.10–3.13 |
| NumPy | 2.x (`>=2.0,<3`) |
| SciPy | 1.x (`>=1.13,<2`) |
| CuPy CUDA 12 | 13.x–14.x |
| CuPy CUDA 13 | 14.x |
| CPU OS | 64-bit Linux and macOS |
| GPU OS | 64-bit Linux with a compatible NVIDIA driver |

Windows is not certified in version 0.1.2.  Other POSIX platforms may work but
are outside the support matrix until tested.

## Verified environments

| Platform | Python | NumPy | SciPy | GPU runtime | Result |
|---|---:|---:|---:|---|---|
| macOS 15.7.9 arm64 | 3.10.10 | 2.2.6 | 1.15.3 | current source, CPU | 53 passed, 3 CUDA-only skipped |
| macOS 15.7.3 arm64 | 3.10.10 | 2.2.6 | 1.15.3 | CPU wheel | 41 passed, 2 CUDA-only skipped |
| macOS 15.7.3 arm64 | 3.11.15 | 2.4.6 | 1.17.1 | CPU wheel | 41 passed, 2 CUDA-only skipped |
| macOS 15.7.3 arm64 | 3.13.12 | 2.5.2 | 1.18.0 | CPU wheel | 41 passed, 2 CUDA-only skipped |
| macOS 15.7.3 arm64 | 3.10.10 | 2.0.0 | 1.13.0 | CPU wheel, minimum declared dependencies | 41 passed, 2 CUDA-only skipped |
| Linux x86_64 | 3.12.3 | 2.5.2 | 1.18.0 | CuPy 14.2 / CUDA 12.9 / RTX 5070 Ti | 41 passed |

The first row is the complete current-source test suite run on 2026-08-31.
The remaining macOS rows are archived clean-environment installations from
the earlier audited 0.1.2 wheel; their smaller pass count reflects the test
suite at that release checkpoint, not skipped current tests. The Linux row
includes the two CUDA-only runtime tests from that checkpoint. The two later
release-example configuration checks are backend-neutral and passed in every
macOS wheel environment; they were not rerun on the GPU host. Python 3.12 CPU
behavior is also exercised by the Linux GPU run; a separate macOS 3.12
interpreter was not locally installed.

The minimum-dependency row verifies the declared lower bounds on Python 3.10.
On newer Python versions, the package manager selects dependency releases that
also support that interpreter. A platform is not advertised as certified
merely because dependency resolution succeeds.

## Backend rules

- Always set `solver.useGpu` explicitly in production configuration.
- CPU and GPU publication/customer checkpoints use `blockPrecision="double"`.
- Single-precision GPU output requires a double-precision confirmation.
- Kinetic-electron calculations should use `parallelBoundary="open"`.
- `adiabatic + TEM` is not a physical feature combination.
