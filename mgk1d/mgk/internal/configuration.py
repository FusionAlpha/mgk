"""Configuration validation and dimensional normalization."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from math import ceil, floor, pi, sqrt
import os
from pathlib import Path
from typing import Any

import numpy as np

from mgk._struct import Struct, as_struct
from .physics import build_field_layout, build_kinetic_model

ELEMENTARY_CHARGE = 1.602176634e-19
ATOMIC_MASS_UNIT = 1.66053906660e-27


def _defaults() -> Struct:
    return Struct(
        physical=Struct(
            magneticField=2.0,
            majorRadius=1.7,
            minorRadius=0.0,
            ionMass=2 * ATOMIC_MASS_UNIT,
            ionChargeNumber=1,
            ionTemperature=1000.0,
            electronTemperature=1000.0,
            electronTemperatureGradientLength=None,
            electronBeta=0.0,
            densityGradientLength=1.7 / 4,
            ionTemperatureGradientLength=1.7 / 10,
            binormalWavenumber=None,
        ),
        geometry=Struct(
            model="s-alpha", q=1.0, magneticShear=1.0, alpha=0.0,
            ballooningAngle=0.0, shiftDerivative=0.0,
            elongation=1.0, elongationShear=0.0,
            triangularity=0.0, triangularityShear=0.0,
            betaStar=0.0, tableResolution=1001,
            profileFile=None, fieldLineLabel=0.0,
            # Keep the geometry-level spelling for compatibility; None
            # preserves the legacy model.mirrorConvention input.
            mirrorConvention=None,
        ),
        model=Struct(
            # Deprecated compatibility input.  The public solver is now
            # orbit-only; false is accepted but no longer selects a separate
            # grid backend.
            magneticMirror=True,
            aparallel=False,
            bparallel=False,
            parallelBoundary="periodic",
            boundaryCoordinateStretch=1.0,
            boundarySpongeStrength=0.0,
            boundarySpongeFraction=0.30,
            boundaryTailPeriods=1,
            boundaryTailPoints=17,
            mirrorConvention="cgyro_s_alpha",
            electronClosure="auto",
        ),
        species=Struct(enabled=False, items=[]),
        grid=Struct(
            thetaMin=-4 * pi,
            thetaMax=4 * pi,
            numTheta=129,
            thetaMapAlpha=0.0,
            energyMax=12.5,
            numEnergy=16,
            numPitch=16,
            numBouncePoints=32,
        ),
        solver=Struct(
            useGpu=None,
            blockPrecision="single",
            eigenBackend="",
            modeSelection="nearest",
            singleShiftTimeLimit=5.0,
            eigenTolerance=1e-7,
            eigenSubspaceDimension=7,
            eigenHotSubspaceDimension=5,
            enableWarmRitz=True,
            eigenMaxIterations=None,
            eigenInitialVector=None,
            gpuArnoldiMaxRestarts=3,
            cpuFactorizationWorkers=0,
            cpuOperatorWorkers=0,
            frequencyGuess=None,
            checkBlockFactorization=False,
            enableFactorizationCache=None,
            enableGpuFactorizationCache=True,
            enableGpuResultCache=True,
            compactResult=False,
            returnMatrices=False,
        ),
    )


def _merge(input_cfg: Struct, defaults: Struct) -> Struct:
    unsupported = set(input_cfg) - set(defaults)
    if unsupported:
        raise ValueError(f"Unsupported configuration field(s): {', '.join(sorted(unsupported))}")
    merged = deepcopy(defaults)
    for name, value in dict.items(input_cfg):
        if value is not None:
            merged[name] = value
    return merged


def _positive(value: Any, name: str) -> None:
    if not np.isscalar(value) or not np.isreal(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"cfg.{name} must be a positive finite scalar")


def _validate(cfg: Struct) -> None:
    p, g, m, grid, solver = cfg.physical, cfg.geometry, cfg.model, cfg.grid, cfg.solver
    for name in (
        "magneticField", "majorRadius", "ionMass", "ionTemperature",
        "electronTemperature", "densityGradientLength",
        "ionTemperatureGradientLength", "electronTemperatureGradientLength",
    ):
        _positive(p[name], f"physical.{name}")
    if not 0 <= p.minorRadius < p.majorRadius:
        raise ValueError("cfg.physical.minorRadius must satisfy 0 <= a < R")
    if not np.isfinite(p.ionChargeNumber) or p.ionChargeNumber == 0:
        raise ValueError("cfg.physical.ionChargeNumber must be nonzero")
    if p.electronBeta < 0 or not np.isfinite(p.electronBeta):
        raise ValueError("cfg.physical.electronBeta must be nonnegative")
    if p.binormalWavenumber is not None:
        _positive(p.binormalWavenumber, "physical.binormalWavenumber")
    if not np.isfinite(g.q) or g.q == 0:
        raise ValueError("cfg.geometry.q must be nonzero")
    if str(g.model).lower() not in {"s-alpha", "miller", "stellarator"}:
        raise ValueError("cfg.geometry.model must be s-alpha, miller, or stellarator")
    for name in (
        "magneticShear", "alpha", "ballooningAngle", "shiftDerivative",
        "elongation", "elongationShear", "triangularity",
        "triangularityShear", "betaStar",
    ):
        if not np.isscalar(g[name]) or not np.isreal(g[name]) or not np.isfinite(g[name]):
            raise ValueError(f"cfg.geometry.{name} must be a finite real scalar")
    if g.elongation <= 0 or not -1 < g.triangularity < 1:
        raise ValueError("Miller elongation must be positive and triangularity in (-1, 1)")
    if (not isinstance(g.tableResolution, (int, np.integer))
            or g.tableResolution < 5 or g.tableResolution % 2 == 0):
        raise ValueError("cfg.geometry.tableResolution must be an odd integer >= 5")
    if str(g.model).lower() == "miller" and p.minorRadius <= 0:
        raise ValueError("Miller geometry requires physical.minorRadius > 0")
    if str(g.model).lower() == "stellarator":
        if not isinstance(g.profileFile, (str, os.PathLike)) or not str(g.profileFile).strip():
            raise ValueError(
                "stellarator geometry requires geometry.profileFile pointing to a preprocessed .npz profile"
            )
        if not Path(g.profileFile).expanduser().is_file():
            raise ValueError(f"stellarator geometry profile does not exist: {g.profileFile}")
        from .geometry import load_stellarator_profile
        profile = load_stellarator_profile(g.profileFile)
        if not profile.periodic and m.parallelBoundary == "periodic":
            raise ValueError(
                "non-periodic stellarator profiles require an open parallel boundary"
            )
        if (not profile.periodic
                and (grid.thetaMin < profile.z[0] or grid.thetaMax > profile.z[-1])):
            raise ValueError(
                "non-periodic stellarator profile must cover the complete theta grid"
            )
    if (not np.isscalar(g.fieldLineLabel) or not np.isreal(g.fieldLineLabel)
            or not np.isfinite(g.fieldLineLabel)):
        raise ValueError("cfg.geometry.fieldLineLabel must be a finite real scalar")
    for name in ("magneticMirror", "aparallel", "bparallel"):
        if not isinstance(m[name], (bool, int, np.bool_, np.integer)):
            raise ValueError(f"cfg.model.{name} must be scalar logical")
    if (not np.isscalar(m.boundaryCoordinateStretch)
            or not np.isfinite(m.boundaryCoordinateStretch)
            or m.boundaryCoordinateStretch < 1):
        raise ValueError("boundaryCoordinateStretch must be finite and >= 1")
    if (not np.isscalar(m.boundarySpongeStrength)
            or not np.isfinite(m.boundarySpongeStrength)
            or m.boundarySpongeStrength < 0):
        raise ValueError("boundarySpongeStrength must be finite and nonnegative")
    if (not np.isscalar(m.boundarySpongeFraction)
            or not np.isfinite(m.boundarySpongeFraction)
            or not 0 < m.boundarySpongeFraction <= 0.5):
        raise ValueError("boundarySpongeFraction must satisfy 0 < value <= 0.5")
    if (not isinstance(m.boundaryTailPeriods, (int, np.integer))
            or m.boundaryTailPeriods < 0):
        raise ValueError("boundaryTailPeriods must be a nonnegative integer")
    if (not isinstance(m.boundaryTailPoints, (int, np.integer))
            or m.boundaryTailPoints < 4):
        raise ValueError("boundaryTailPoints must be an integer >= 4")
    if m.parallelBoundary not in {"periodic", "open", "open-extrapolated", "open-dtn"}:
        raise ValueError("parallelBoundary must be periodic, open, open-extrapolated, or open-dtn")
    if str(m.electronClosure).lower() not in {"auto", "adiabatic", "kinetic", "massless"}:
        raise ValueError("electronClosure must be auto, adiabatic, kinetic, or massless")
    if m.mirrorConvention not in {"physical", "cgyro_s_alpha"}:
        raise ValueError("mirrorConvention must be physical or cgyro_s_alpha")
    if m.parallelBoundary in {"open-extrapolated", "open-dtn"} and not m.magneticMirror:
        raise ValueError("open-extrapolated and open-dtn currently require orbit discretization")
    if m.boundaryCoordinateStretch != 1 and m.parallelBoundary != "open-extrapolated":
        raise ValueError("boundaryCoordinateStretch requires open-extrapolated")
    if (m.parallelBoundary == "open-dtn"
            and (m.boundaryCoordinateStretch != 1 or m.boundarySpongeStrength != 0)):
        raise ValueError("open-dtn requires coordinate stretch and sponge damping to be disabled")
    if (not np.isscalar(grid.thetaMapAlpha) or not np.isreal(grid.thetaMapAlpha)
            or not np.isfinite(grid.thetaMapAlpha) or grid.thetaMapAlpha < 0):
        raise ValueError("thetaMapAlpha must be a nonnegative finite scalar")
    if m.parallelBoundary == "open-dtn" and grid.thetaMapAlpha != 0:
        raise ValueError("thetaMapAlpha is not supported with open-dtn")
    if (m.aparallel or m.bparallel) and p.electronBeta <= 0:
        raise ValueError("electronBeta must be positive when an electromagnetic field is enabled")
    for name, lower in (("numTheta", 3), ("numEnergy", 2), ("numPitch", 2), ("numBouncePoints", 8)):
        value = grid[name]
        if not isinstance(value, (int, np.integer)) or value < lower:
            raise ValueError(f"cfg.grid.{name} must be an integer >= {lower}")
    if grid.thetaMin >= grid.thetaMax:
        raise ValueError("thetaMin must be smaller than thetaMax")
    if not np.isfinite(grid.thetaMin) or not np.isfinite(grid.thetaMax):
        raise ValueError("theta limits must be finite")
    _positive(grid.energyMax, "grid.energyMax")
    if solver.blockPrecision not in {"single", "double"}:
        raise ValueError("blockPrecision must be single or double")
    if solver.eigenBackend not in {"", "eigs", "gpu_arnoldi"}:
        raise ValueError("eigenBackend must be eigs or gpu_arnoldi")
    if solver.modeSelection not in {"nearest", "max_growth_scan"}:
        raise ValueError("modeSelection must be nearest or max_growth_scan")
    _positive(solver.eigenTolerance, "solver.eigenTolerance")
    if (not isinstance(solver.eigenSubspaceDimension, (int, np.integer))
            or solver.eigenSubspaceDimension < 2):
        raise ValueError("eigenSubspaceDimension must be an integer >= 2")
    if (not isinstance(solver.eigenHotSubspaceDimension, (int, np.integer))
            or solver.eigenHotSubspaceDimension < 3):
        raise ValueError("eigenHotSubspaceDimension must be an integer >= 3")
    if not isinstance(solver.enableWarmRitz, (bool, int, np.bool_, np.integer)):
        raise ValueError("enableWarmRitz must be scalar logical")
    if (not isinstance(solver.gpuArnoldiMaxRestarts, (int, np.integer))
            or solver.gpuArnoldiMaxRestarts < 1):
        raise ValueError("gpuArnoldiMaxRestarts must be an integer >= 1")
    if (not isinstance(solver.cpuFactorizationWorkers, (int, np.integer))
            or solver.cpuFactorizationWorkers < 0):
        raise ValueError("cpuFactorizationWorkers must be an integer >= 0")
    if (not isinstance(solver.cpuOperatorWorkers, (int, np.integer))
            or solver.cpuOperatorWorkers < 0):
        raise ValueError("cpuOperatorWorkers must be an integer >= 0")
    if (not np.isscalar(solver.singleShiftTimeLimit)
            or np.isnan(solver.singleShiftTimeLimit)
            or solver.singleShiftTimeLimit <= 0):
        raise ValueError("singleShiftTimeLimit must be positive or infinity")
    if solver.frequencyGuess is not None and (
            not np.isscalar(solver.frequencyGuess) or not np.isfinite(solver.frequencyGuess)):
        raise ValueError("frequencyGuess must be a finite scalar")
    if solver.eigenBackend == "gpu_arnoldi" and not solver.useGpu:
        raise ValueError("GPU Arnoldi requires useGpu=true")


def _cupy_available() -> bool:
    try:
        import os
        import tempfile
        os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mgk-cupy-cache"))
        import cupy  # type: ignore
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def build_config(input_config: Mapping[str, Any] | None = None) -> tuple[Struct, Struct, Struct]:
    supplied = as_struct({} if input_config is None else input_config)
    defaults = _defaults()
    unsupported = set(supplied) - set(defaults)
    if unsupported:
        raise ValueError(f"Unsupported top-level configuration field(s): {', '.join(sorted(unsupported))}")
    provided_grid = set(supplied.get("grid", {}))
    provided_solver = set(supplied.get("solver", {}))
    provided_geometry = set(supplied.get("geometry", {}))
    provided_model = set(supplied.get("model", {}))
    cfg = Struct()
    for section in defaults:
        value = supplied.get(section, {})
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise TypeError(f"cfg.{section} must be a mapping")
        try:
            cfg[section] = _merge(as_struct(value), defaults[section])
        except ValueError as exc:
            raise ValueError(str(exc).replace("configuration", f"cfg.{section}")) from exc
    if cfg.physical.electronTemperatureGradientLength is None:
        cfg.physical.electronTemperatureGradientLength = cfg.physical.ionTemperatureGradientLength
    if cfg.solver.enableFactorizationCache is None:
        cfg.solver.enableFactorizationCache = cfg.solver.enableGpuFactorizationCache
    cfg.model.parallelBoundary = str(cfg.model.parallelBoundary).lower()
    cfg.model.electronClosure = str(cfg.model.electronClosure).lower()
    cfg.geometry.model = str(cfg.geometry.model).lower()
    if cfg.geometry.profileFile is not None:
        cfg.geometry.profileFile = str(Path(cfg.geometry.profileFile).expanduser().resolve())
    if (cfg.geometry.model == "stellarator" and "q" not in provided_geometry):
        from .geometry import load_stellarator_profile
        profile = load_stellarator_profile(cfg.geometry.profileFile)
        if profile.vmecQ is not None and np.isfinite(profile.vmecQ) and profile.vmecQ != 0:
            cfg.geometry.q = float(profile.vmecQ)
    if "mirrorConvention" in provided_geometry:
        cfg.model.mirrorConvention = str(cfg.geometry.mirrorConvention).lower()
    elif "mirrorConvention" in provided_model:
        cfg.model.mirrorConvention = str(cfg.model.mirrorConvention).lower()
    elif (cfg.model.aparallel or cfg.model.bparallel
          or cfg.geometry.model in {"miller", "stellarator"}):
        # Electromagnetic orbit calculations use the physical mirror
        # convention unless the user explicitly overrides it. Keep the old
        # cgyro convention for the electrostatic compatibility path.
        cfg.model.mirrorConvention = "physical"
    else:
        cfg.model.mirrorConvention = str(cfg.model.mirrorConvention).lower()
    cfg.solver.blockPrecision = str(cfg.solver.blockPrecision).lower()
    cfg.solver.eigenBackend = str(cfg.solver.eigenBackend).lower()
    cfg.solver.modeSelection = str(cfg.solver.modeSelection).lower()
    if "numEnergy" not in provided_grid:
        cfg.grid.numEnergy = 28
    if "numPitch" not in provided_grid:
        cfg.grid.numPitch = 32
    if "eigenSubspaceDimension" not in provided_solver:
        cfg.solver.eigenSubspaceDimension = 8
    if cfg.solver.useGpu is None:
        cfg.solver.useGpu = _cupy_available()
    _validate(cfg)
    p, g, m, grid, solver = cfg.physical, cfg.geometry, cfg.model, cfg.grid, cfg.solver
    vti = sqrt(p.ionTemperature * ELEMENTARY_CHARGE / p.ionMass)
    omega_ci = abs(p.ionChargeNumber) * ELEMENTARY_CHARGE * p.magneticField / p.ionMass
    rho_i = vti / omega_ci
    omega_ref = vti / p.majorRadius
    if p.binormalWavenumber is None:
        p.binormalWavenumber = (0.45 / sqrt(2)) / rho_i
    normalized = Struct(
        s=g.magneticShear, q=g.q, alpha=g.alpha, tk=g.ballooningAngle,
        epsilon=p.minorRadius / p.majorRadius,
        geometryModel=g.model,
        geometryProfileFile=g.profileFile,
        geometryFieldLineLabel=float(g.fieldLineLabel),
        minorRadius=p.minorRadius, majorRadius=p.majorRadius,
        shiftDerivative=g.shiftDerivative,
        elongation=g.elongation, elongationShear=g.elongationShear,
        triangularity=g.triangularity,
        triangularityShear=g.triangularityShear,
        betaStar=g.betaStar, tableResolution=g.tableResolution,
        tau=p.electronTemperature / p.ionTemperature,
        epsn=p.densityGradientLength / p.majorRadius,
        etai=p.densityGradientLength / p.ionTemperatureGradientLength,
        kt=p.binormalWavenumber * rho_i,
        thmin=grid.thetaMin, thmax=grid.thetaMax, nth=grid.numTheta,
        thetaMapAlpha=grid.thetaMapAlpha,
        boundaryCoreMin=grid.thetaMin, boundaryCoreMax=grid.thetaMax,
        boundaryCoreNumTheta=grid.numTheta,
        energyMax=grid.energyMax, numEnergy=grid.numEnergy,
        numPitch=grid.numPitch, numBouncePoints=grid.numBouncePoints,
        numVelocity=2 * grid.numEnergy * grid.numPitch,
        discretization="orbit",
        mirrorConvention=m.mirrorConvention, aparallel=bool(m.aparallel),
        bparallel=bool(m.bparallel),
        parallelBoundary=m.parallelBoundary,
        electronClosure=m.electronClosure,
        boundaryCoordinateStretch=m.boundaryCoordinateStretch,
        boundarySpongeStrength=m.boundarySpongeStrength,
        boundarySpongeFraction=m.boundarySpongeFraction,
        boundaryTailPeriods=m.boundaryTailPeriods,
        boundaryTailPoints=m.boundaryTailPoints,
        betaElectron=p.electronBeta,
        useGpu=bool(solver.useGpu), blockPrecision=solver.blockPrecision,
        eigenBackend=solver.eigenBackend, modeSelection=solver.modeSelection,
        singleShiftTimeLimit=solver.singleShiftTimeLimit,
        eigsTolerance=solver.eigenTolerance,
        eigsSubspaceDimension=solver.eigenSubspaceDimension,
        eigsHotSubspaceDimension=solver.eigenHotSubspaceDimension,
        enableWarmRitz=bool(solver.enableWarmRitz),
        eigsMaxIterations=solver.eigenMaxIterations,
        eigsInitialVector=solver.eigenInitialVector,
        gpuArnoldiMaxRestarts=solver.gpuArnoldiMaxRestarts,
        cpuFactorizationWorkers=(
            # Batched small-page LAPACK benefits from modest over-subscription
            # because the outer group coordinators spend most of their time
            # waiting for inner page workers.
            min(16, 2 * (os.cpu_count() or 1))
            if solver.cpuFactorizationWorkers == 0
            else int(solver.cpuFactorizationWorkers)
        ),
        cpuOperatorWorkers=(
            min(8, os.cpu_count() or 1)
            if solver.cpuOperatorWorkers == 0
            else int(solver.cpuOperatorWorkers)
        ),
        blockFactorizationCheck=bool(solver.checkBlockFactorization),
        enableFactorizationCache=bool(solver.enableFactorizationCache),
        enableGpuFactorizationCache=bool(solver.enableGpuFactorizationCache),
        enableGpuResultCache=bool(solver.enableGpuResultCache),
        compactResult=bool(solver.compactResult), returnMatrices=bool(solver.returnMatrices),
    )
    if normalized.geometryModel == "stellarator":
        profile_path = Path(normalized.geometryProfileFile)
        stat = profile_path.stat()
        normalized.geometryProfileKey = (
            f"{profile_path}:{stat.st_mtime_ns}:{stat.st_size}"
        )
    else:
        normalized.geometryProfileKey = None
    if normalized.parallelBoundary == "open-dtn":
        left_interface = pi + 2 * pi * floor((normalized.boundaryCoreMin - pi) / (2 * pi))
        right_interface = pi + 2 * pi * ceil((normalized.boundaryCoreMax - pi) / (2 * pi))
        tolerance = 100 * np.finfo(float).eps * max(
            1.0, abs(normalized.boundaryCoreMin), abs(normalized.boundaryCoreMax)
        )
        if abs(left_interface - normalized.boundaryCoreMin) < tolerance:
            left_interface = normalized.boundaryCoreMin
        if abs(right_interface - normalized.boundaryCoreMax) < tolerance:
            right_interface = normalized.boundaryCoreMax

        def transition_points(width):
            if width <= tolerance:
                return 0
            return max(4, int(round(
                1 + (normalized.boundaryTailPoints - 1) * width / (2 * pi)
            )))

        normalized.boundaryLeftTailInterface = left_interface
        normalized.boundaryRightTailInterface = right_interface
        normalized.boundaryLeftTransitionPoints = transition_points(
            normalized.boundaryCoreMin - left_interface
        )
        normalized.boundaryRightTransitionPoints = transition_points(
            right_interface - normalized.boundaryCoreMax
        )
        normalized.thmin = left_interface - 2 * pi * normalized.boundaryTailPeriods
        normalized.thmax = right_interface + 2 * pi * normalized.boundaryTailPeriods
        normalized.nth = (
            normalized.boundaryCoreNumTheta
            + 2 * normalized.boundaryTailPeriods * (normalized.boundaryTailPoints - 1)
            + max(normalized.boundaryLeftTransitionPoints - 1, 0)
            + max(normalized.boundaryRightTransitionPoints - 1, 0)
        )
    normalized.fields = build_field_layout(normalized)
    if not normalized.eigenBackend:
        normalized.eigenBackend = "gpu_arnoldi" if normalized.useGpu else "eigs"
    normalized.eigsGuess = ((-0.615095 + 0.263450j) * normalized.kt / normalized.epsn
                            if solver.frequencyGuess is None else solver.frequencyGuess / omega_ref)
    normalized.species = build_kinetic_model(cfg.species, p, normalized)
    if normalized.electronClosure == "auto":
        electron = normalized.species.items[1]
        normalized.electronClosure = "kinetic" if electron.kinetic else "adiabatic"
    electron = normalized.species.items[1]
    if normalized.electronClosure == "kinetic" and not electron.kinetic:
        raise ValueError("The kinetic electron closure requires cfg.species.items.kinetic=true")
    if normalized.electronClosure in {"adiabatic", "massless"} and electron.kinetic:
        raise ValueError(f"The {normalized.electronClosure} electron closure requires cfg.species.items.kinetic=false")
    if normalized.electronClosure == "massless":
        raise ValueError(
            "The massless electron closure was part of the removed grid "
            "backend and is not supported by the orbit-only solver"
        )
    if normalized.returnMatrices:
        raise ValueError(
            "returnMatrices was part of the removed grid backend; the "
            "orbit-only solver is matrix-free"
        )
    normalized.numVelocity *= normalized.species.numKinetic
    normalization = Struct(
        velocity=vti, length=p.majorRadius, frequency=omega_ref,
        time=1 / omega_ref, cyclotronFrequency=omega_ci, gyroradius=rho_i,
    )
    return cfg, normalized, normalization
