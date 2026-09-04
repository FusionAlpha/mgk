# MGK1D Python

Repository: [github.com/FusionAlpha/mgk](https://github.com/FusionAlpha/mgk)
License: BSD 3-Clause. Copyright (c) 2026 FusionAlpha.

MGK1D is a local, collisionless, linear gyrokinetic eigenvalue solver for
ion-temperature-gradient (ITG), trapped-electron-mode (TEM), and
electromagnetic kinetic-ballooning-mode (KBM) studies. The Python package uses
a matrix-free passing/trapped-orbit discretization and supports:

- s-alpha and Miller local magnetic geometry;
- adiabatic or kinetic electrons;
- electrostatic `[phi]`, electromagnetic `[phi,A_parallel]`, and
  `[phi,A_parallel,B_parallel]` field layouts;
- CPU and NVIDIA CUDA backends;
- double-precision production calculations and single-precision GPU previews.

Python is the only public installation and execution path. The repository and
release artifacts do not require commercial software.

## Install

MGK1D requires Python 3.10--3.13, NumPy 2.x, and SciPy 1.13 or newer.

Install from a source checkout for CPU use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a built wheel:

```bash
python -m pip install ./mgk-0.1.2-py3-none-any.whl
```

On 64-bit Linux with a compatible NVIDIA driver, install the matching CuPy
variant from a source checkout:

```bash
python -m pip install -e '.[gpu12]'
# or
python -m pip install -e '.[gpu13]'
```

Do not install `cupy-cuda12x` and `cupy-cuda13x` in the same environment. A
CUDA toolkit compiler is not required, but a compatible NVIDIA driver is.

## Verify the installation

```bash
python -c "import mgk; print(mgk.__version__)"
python mgk/examples/cpu_quickstart.py
python -m pip install -e '.[test]'
python -m pytest -q
```

On a GPU host:

```bash
python - <<'PY'
import cupy as cp
print("CuPy", cp.__version__)
print("GPU count", cp.cuda.runtime.getDeviceCount())
if cp.cuda.runtime.getDeviceCount():
    print("GPU", cp.cuda.Device().name)
PY
```

## Typical case studies

The repository's most visible scientific results are collected in
[research_results/case_studies/](research_results/case_studies/README.md). Each case includes a report-ready
figure, the CSV/JSON values used to draw it, physical and numerical
parameters, and the relevant Python source snapshot. The original large
analysis workspace remains a local historical archive; the case-study data
are the compact reproducible record intended for reports and code review.

## CPU quick start

```python
import mgk

config = {
    "geometry": {
        "model": "s-alpha",
        "q": 1.0,
        "magneticShear": 1.0,
        "alpha": 0.0,
    },
    "model": {
        "electronClosure": "adiabatic",
        "parallelBoundary": "periodic",
    },
    "grid": {
        "numTheta": 65,
        "numEnergy": 16,
        "numPitch": 24,
        "numBouncePoints": 32,
    },
    "solver": {
        "useGpu": False,
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
        "singleShiftTimeLimit": 60,
    },
}

result = mgk.solve(config)
print("MGK1D", mgk.__version__)
print("omega R/v_ti =", result.omega)
print("omega [rad/s] =", result.omegaPhysical)
print("frequency [Hz] =", result.frequencyHz)
print("growth rate [1/s] =", result.growthRate)
print("eigen residual =", result.eigenResidual)
print("field residual =", result.fieldConstraintResidual)
```

Modes use the `exp(-i*omega*t)` convention, so `Im(omega)>0` denotes growth.
`result.omega` is normalized to `v_ti/R`; `result.omegaPhysical` is in rad/s.

## Kinetic electrons

Provide an electron species explicitly and use an open parallel boundary:

```python
config = {
    "geometry": {"model": "s-alpha", "q": 1.4, "magneticShear": 0.8},
    "model": {
        "electronClosure": "kinetic",
        "parallelBoundary": "open",
    },
    "species": {
        "enabled": True,
        "items": {"kind": "electron", "kinetic": True},
    },
    "grid": {
        "numTheta": 97,
        "numEnergy": 16,
        "numPitch": 32,
        "numBouncePoints": 48,
    },
    "solver": {
        "useGpu": False,
        "blockPrecision": "double",
        "eigenTolerance": 1e-8,
    },
}

result = mgk.solve(config)
```

ITG and TEM branches can coexist. Track them from separate anchor points using
the previous result's `omegaPhysical` and `reducedMode`; choosing only the
largest growth rate at each scan point can silently switch branches.

## Electromagnetic fields

Electromagnetic calculations require `physical.electronBeta > 0`. Starting
from the kinetic-electron configuration above, enable two fields with:

```python
config["physical"] = {"electronBeta": 0.02}
config["model"].update({"aparallel": True, "bparallel": False})
result = mgk.solve(config)
print(result.fields)  # ['phi', 'aparallel']
```

Enable three fields with:

```python
config["model"].update({"aparallel": True, "bparallel": True})
result = mgk.solve(config)
print(result.fields)  # ['phi', 'aparallel', 'bparallel']
```

The corresponding field arrays are returned as `result.phi`,
`result.aparallel`, and `result.bparallel`.

## Select CPU or GPU explicitly

Production scripts should not depend on device auto-detection:

```python
# CPU double
config["solver"].update({
    "useGpu": False,
    "eigenBackend": "eigs",
    "blockPrecision": "double",
})

# GPU double
config["solver"].update({
    "useGpu": True,
    "eigenBackend": "gpu_arnoldi",
    "blockPrecision": "double",
    "enableGpuFactorizationCache": True,
    "enableGpuResultCache": False,
})
```

Use double precision for publication and externally reported frequencies.
Single precision is intended for exploratory GPU scans and every retained
point must be confirmed with a double-precision checkpoint.

## Accepting a result

Always save the frequency, both algebraic residuals, field layout, and timing:

```python
print(result.omega)
print(result.eigenResidual)
print(result.fieldConstraintResidual)
print(result.fields)
print(result.timing)

assert result.eigenResidual < 2e-6
assert result.fieldConstraintResidual < 2e-6
```

Finite residuals do not establish grid convergence. Production studies must
also scan the theta domain, `numTheta`, `numEnergy`, `numPitch`,
`numBouncePoints`, and `energyMax`.

## Documentation

- [CONFIGURATION_REFERENCE.md](CONFIGURATION_REFERENCE.md): every Python
  configuration field, effective default, unit, and constraint.
- [GPU_GUIDE.md](GPU_GUIDE.md): CUDA/CuPy installation, precision, caching,
  timing rules, performance guidance, and troubleshooting.
- [BENCHMARK_CASES.md](BENCHMARK_CASES.md): exact inputs and provenance for
  the adiabatic, kinetic-electron, Shen electromagnetic, and Xie-2016 KBM
  benchmark figures.
- [PUBLISHED_VALIDATION_CASES.md](PUBLISHED_VALIDATION_CASES.md): published
  validation inputs, normalization, branch selection, and checkpoints.
- [COMPATIBILITY.md](COMPATIBILITY.md): tested Python, dependency, and platform
  combinations.
- [SUPPORT.md](SUPPORT.md): supported physics and external support boundary.

The stable Python entry points documented here are `mgk.solve`,
`mgk.Struct`, and `mgk.__version__`. Names under `mgk.internal` are
implementation details and may change between releases.
The deprecated `mgk1d` import shim is retained only for older scripts; new
code should use `mgk`.

## Current limitations

The public Python solver does not include collisions, rotation, global radial
physics, nonlinear evolution, massless-fluid electrons, or the retired grid
backend. MGK1D performs no telemetry and makes no network requests. Set
`solver.useGpu` explicitly for reproducible backend selection.

This software is distributed under the BSD 3-Clause license. See
[LICENSE](LICENSE) for the complete license text.
