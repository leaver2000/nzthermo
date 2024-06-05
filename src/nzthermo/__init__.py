__all__ = [
    # ._version
    "__version__",
    # ._c
    "OPENMP_ENABLED",
    "lcl",
    "moist_lapse",
    "wet_bulb_temperature",
    "delta_t",
    # ._ufunc
    "potential_temperature",
    "wind_direction",
    "wind_components",
    "wind_magnitude",
    "equivalent_potential_temperature",
    "wet_bulb_potential_temperature",
    "wet_bulb_temperature",
    # .core
    "cape_cin",
    "ccl",
    "dewpoint",
    "dewpoint_from_specific_humidity",
    "downdraft_cape",
    "dry_lapse",
    "el",
    "exner_function",
    "lfc",
    "parcel_profile",
    "mixing_ratio",
    "mixing_ratio_from_specific_humidity",
    "saturation_mixing_ratio",
    "saturation_vapor_pressure",
    "vapor_pressure",
    "virtual_temperature",
    "most_unstable_parcel",
    # .utils
    "timeseries",
    "parcel_profile",
    # aliases
    "wdir",
    "wspd",
    "theta",
    "theta_e",
    "theta_w",
]
from typing import Final
from ._core import OPENMP_ENABLED, lcl, moist_lapse, parcel_profile
from ._ufunc import (
    delta_t,
    potential_temperature,
    equivalent_potential_temperature,
    wind_direction,
    wind_components,
    wind_magnitude,
    wet_bulb_potential_temperature,
    wet_bulb_temperature,
    dry_lapse,
    dewpoint,
)
from ._version import __version__
from .core import (
    cape_cin,
    ccl,
    dewpoint_from_specific_humidity,
    downdraft_cape,
    el,
    exner_function,
    lfc,
    mixing_ratio,
    mixing_ratio_from_specific_humidity,
    most_unstable_parcel,
    # parcel_profile,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    vapor_pressure,
    virtual_temperature,
)
from .utils import timeseries

wdir: Final = wind_direction
wspd: Final = wind_magnitude


theta: Final = potential_temperature
theta_e: Final = equivalent_potential_temperature
theta_w: Final = wet_bulb_potential_temperature
malr: Final = moist_lapse
dalr: Final = dry_lapse
esat: Final = saturation_vapor_pressure
e: Final = vapor_pressure
mr: Final = mixing_ratio
