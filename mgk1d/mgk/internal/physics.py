"""Species normalization and field closure."""

from __future__ import annotations

from copy import deepcopy
from math import sqrt

import numpy as np
from scipy.special import j0, j1

from mgk._struct import Struct, as_struct

ELECTRON_MASS = 9.1093837139e-31


def build_field_layout(cfg) -> Struct:
    names = ["phi"]
    if cfg.aparallel:
        names.append("aparallel")
    if cfg.bparallel:
        names.append("bparallel")
    return Struct(names=names, count=len(names))


def two_j1_over_argument(argument, return_derivative=False):
    """Return ``2*J1(x)/x`` and optionally its derivative.

    The series branch avoids loss of
    precision when the gyro-average argument is small.  The old scalar return
    behavior is preserved unless ``return_derivative`` is requested.
    """
    argument = np.asarray(argument)
    result = np.ones_like(argument, dtype=np.result_type(argument, float))
    derivative = np.zeros_like(result)
    large = np.abs(argument) > 1e-4
    if np.any(large):
        large_argument = argument[large]
        result[large] = 2 * j1(large_argument) / large_argument
        # d/dx [2 J1(x)/x] = 2 J0(x)/x - 4 J1(x)/x^2.
        derivative[large] = (2 * j0(large_argument) / large_argument
                             - 4 * j1(large_argument) / large_argument**2)
    small = argument[~large]
    if np.any(~large):
        result[~large] = 1 - small**2 / 8 + small**4 / 192
        derivative[~large] = -small / 4 + small**3 / 48
    return (result, derivative) if return_derivative else result


def _validate_species(species: Struct) -> None:
    for name in ("mass", "temperature", "densityFraction", "densityGradientLength", "temperatureGradientLength"):
        value = species[name]
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"species.{name} must be a positive finite scalar")
    if not np.isfinite(species.chargeNumber) or species.chargeNumber == 0:
        raise ValueError("species.chargeNumber must be nonzero")


def _normalize_species(source: Struct, physical, cfg) -> Struct:
    species = deepcopy(source)
    mass_ratio = source.mass / physical.ionMass
    temperature_ratio = source.temperature / physical.ionTemperature
    charge_ratio = source.chargeNumber / physical.ionChargeNumber
    thermal_speed_ratio = sqrt(temperature_ratio / mass_ratio)
    frequency_scale = -cfg.kt * temperature_ratio / charge_ratio
    density_gradient_ratio = source.densityGradientLength / physical.majorRadius
    species.update(
        massRatio=mass_ratio, temperatureRatio=temperature_ratio,
        thermalSpeedRatio=thermal_speed_ratio,
        gyroradiusRatio=mass_ratio * thermal_speed_ratio / charge_ratio,
        streamingScale=thermal_speed_ratio, driftFrequencyScale=frequency_scale,
        diamagneticFrequency=frequency_scale,
        gradientFrequency=frequency_scale / density_gradient_ratio,
        temperatureGradientRatio=source.densityGradientLength / source.temperatureGradientLength,
        fieldDriveScale=charge_ratio / temperature_ratio,
        momentScale=charge_ratio * source.densityFraction,
        polarizationScale=source.densityFraction * charge_ratio**2 / temperature_ratio,
        maxwellianScale=1.0,
    )
    return species


def build_kinetic_model(input_cfg, physical, normalized) -> Struct:
    main_ion = Struct(
        name="mainIon", kind="ion", kinetic=True, mass=physical.ionMass,
        chargeNumber=physical.ionChargeNumber, temperature=physical.ionTemperature,
        densityFraction=1.0, densityGradientLength=physical.densityGradientLength,
        temperatureGradientLength=physical.ionTemperatureGradientLength,
    )
    electron = Struct(
        name="electron", kind="electron", kinetic=False, mass=ELECTRON_MASS,
        chargeNumber=-1, temperature=physical.electronTemperature,
        densityFraction=abs(physical.ionChargeNumber),
        densityGradientLength=physical.densityGradientLength,
        temperatureGradientLength=physical.electronTemperatureGradientLength,
    )
    items = input_cfg.items
    if items:
        if not input_cfg.enabled:
            raise ValueError("cfg.species.items requires cfg.species.enabled=true")
        if isinstance(items, dict):
            items = [items]
        if len(items) != 1:
            raise ValueError("The current physics model accepts one electron species entry")
        supplied = as_struct(items[0])
        unsupported = set(supplied) - set(electron)
        if unsupported:
            raise ValueError(f"Unsupported electron species field(s): {', '.join(sorted(unsupported))}")
        electron.update(supplied)
        electron.name = str(electron.name)
        electron.kind = str(electron.kind).lower()
        electron.kinetic = bool(electron.kinetic)
    _validate_species(main_ion)
    _validate_species(electron)
    if electron.kind != "electron" or electron.chargeNumber >= 0:
        raise ValueError("cfg.species.items must describe a negatively charged electron")
    charge_balance = main_ion.chargeNumber + electron.chargeNumber * electron.densityFraction
    if abs(charge_balance) > 1e-12 * max(1, abs(main_ion.chargeNumber)):
        raise ValueError("Ion and electron density fractions must satisfy charge neutrality")
    normalized_items = [
        _normalize_species(main_ion, physical, normalized),
        _normalize_species(electron, physical, normalized),
    ]
    return Struct(enabled=bool(input_cfg.enabled),
                  numKinetic=sum(item.kinetic for item in normalized_items),
                  items=normalized_items)


def resolve_kinetic_species(cfg):
    result = [species for species in cfg.species.items if species.kinetic]
    if not result or len(result) != cfg.species.numKinetic:
        raise ValueError("Inconsistent kinetic-species metadata")
    return result
