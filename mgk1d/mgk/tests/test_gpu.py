import os
import tempfile

import numpy as np
import pytest

os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mgk-cupy-cache"))

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    GPU_AVAILABLE = False

import mgk


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy CUDA device unavailable")
def test_gpu_arnoldi_matches_cpu_orbit_solver():
    common = {
        "grid": {"numTheta": 33, "numEnergy": 6, "numPitch": 6},
        "solver": {
            "blockPrecision": "double",
            "eigenTolerance": 1e-9,
            "eigenSubspaceDimension": 20,
            "singleShiftTimeLimit": 60,
            "enableGpuResultCache": False,
        },
    }
    common["solver"]["useGpu"] = False
    cpu = mgk.solve(common)
    common["solver"].update(useGpu=True, eigenBackend="gpu_arnoldi", gpuArnoldiMaxRestarts=5)
    gpu = mgk.solve(common)
    repeated = mgk.solve(common)
    assert abs(gpu.omega - cpu.omega) < 1e-10
    assert gpu.blockSolverInfo.gpuAssembly
    assert gpu.gpuArnoldiInfo.converged
    assert gpu.eigenResidual < 1e-9
    assert repeated.blockSolverInfo.kineticCacheHit
    assert repeated.blockSolverInfo.numFactorizedPages == 0
    assert repeated.blockSolverInfo.warmStartUsed
    assert repeated.gpuArnoldiInfo.warmStartUsed


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy CUDA device unavailable")
def test_gpu_multifield_assembly_matches_cpu_solver():
    """The native A_parallel/B_parallel assembly must preserve the CPU mode."""
    electron_mass = 9.1093837139e-31
    ion_mass = 1837.0 * electron_mass
    temperature = 1000.0
    elementary_charge = 1.602176634e-19
    ion_thermal_speed = np.sqrt(temperature * elementary_charge / ion_mass)
    ion_gyroradius = ion_thermal_speed / (elementary_charge * 2.0 / ion_mass)
    config = {
        "physical": {
            "magneticField": 2.0, "majorRadius": 1.0,
            "minorRadius": 0.0018, "ionMass": ion_mass,
            "ionChargeNumber": 1, "ionTemperature": temperature,
            "electronTemperature": temperature, "electronBeta": 0.02,
            "densityGradientLength": 0.2,
            "ionTemperatureGradientLength": 0.2,
            "electronTemperatureGradientLength": 0.2,
            "binormalWavenumber": 0.30 / ion_gyroradius,
        },
        "geometry": {
            "q": 2.0, "magneticShear": 1.0, "alpha": 1.6,
            "ballooningAngle": 0.0, "mirrorConvention": "cgyro_s_alpha",
        },
        "model": {
            "magneticMirror": True, "aparallel": True, "bparallel": True,
            "parallelBoundary": "periodic", "electronClosure": "kinetic",
        },
        "species": {"enabled": True, "items": {
            "kind": "electron", "kinetic": True, "mass": electron_mass,
            "temperature": temperature, "densityGradientLength": 0.2,
            "temperatureGradientLength": 0.2,
        }},
        "grid": {
            "thetaMin": -5.0 * np.pi, "thetaMax": 5.0 * np.pi,
            "numTheta": 17, "thetaMapAlpha": 0.0,
            "energyMax": 12.5, "numEnergy": 3,
            "numPitch": 3, "numBouncePoints": 8,
        },
        "solver": {
            "blockPrecision": "double", "eigenTolerance": 1e-9,
            "eigenSubspaceDimension": 20, "singleShiftTimeLimit": 60,
            "enableFactorizationCache": False,
            "enableGpuFactorizationCache": False,
            "enableGpuResultCache": False, "compactResult": True,
        },
    }
    cpu_config = {**config, "solver": {**config["solver"], "useGpu": False}}
    cpu = mgk.solve(cpu_config)
    gpu_config = {
        **config,
        "solver": {
            **config["solver"], "useGpu": True,
            "eigenBackend": "gpu_arnoldi",
            "eigenInitialVector": cpu.reducedMode,
        },
    }
    gpu = mgk.solve(gpu_config)
    assert gpu.blockSolverInfo.gpuAssembly
    assert abs(gpu.omega - cpu.omega) < 1e-8
    assert gpu.eigenResidual < 1e-8
    assert gpu.fieldConstraintResidual < 1e-8


@pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy CUDA device unavailable")
def test_gpu_result_and_orbit_factorization_caches_obey_switches():
    exact = {
        "grid": {"numTheta": 17, "numEnergy": 3, "numPitch": 3},
        "solver": {
            "useGpu": True,
            "blockPrecision": "double",
            "eigenSubspaceDimension": 12,
            "singleShiftTimeLimit": 60,
            "enableGpuResultCache": True,
        },
    }
    assert not mgk.solve(exact).cacheHit
    assert mgk.solve(exact).cacheHit

    orbit = {
        "physical": {"minorRadius": 0.17},
        "geometry": {"ballooningAngle": 0.017},
        "model": {"magneticMirror": True, "parallelBoundary": "open"},
        "grid": {"numTheta": 17, "numEnergy": 3, "numPitch": 3, "numBouncePoints": 8},
        "solver": {
            "useGpu": True,
            "blockPrecision": "double",
            "eigenSubspaceDimension": 30,
            "gpuArnoldiMaxRestarts": 5,
            "singleShiftTimeLimit": 60,
            "enableFactorizationCache": True,
            "enableGpuResultCache": False,
        },
    }
    mgk.solve(orbit)
    orbit["physical"]["ionTemperatureGradientLength"] = 0.2
    cached = mgk.solve(orbit)
    assert cached.blockSolverInfo.kineticCacheHit
    assert cached.blockSolverInfo.numFactorizedPages == 0
    assert cached.blockSolverInfo.warmStartUsed
    assert cached.gpuArnoldiInfo.warmStartUsed

    uncached = {
        **orbit,
        "geometry": {"ballooningAngle": 0.019},
        "physical": {"minorRadius": 0.17},
        "solver": {**orbit["solver"], "enableFactorizationCache": False},
    }
    first_uncached = mgk.solve(uncached)
    uncached["physical"]["ionTemperatureGradientLength"] = 0.2
    second_uncached = mgk.solve(uncached)
    assert not first_uncached.blockSolverInfo.kineticCacheHit
    assert not second_uncached.blockSolverInfo.kineticCacheHit
    assert not second_uncached.blockSolverInfo.warmStartUsed
