__all__ = [
    # .functional
    "functional",
    # ._version
    "__version__",
    # ._core
    "OPENMP_ENABLED",
    "E0",
    "P0",
    "T0",
    "Cp",
    "Lv",
    "Md",
    "Mw",
    "Rd",
    "Rv",
    "epsilon",
    "kappa",
    "moist_lapse",
    "parcel_profile",
    "parcel_profile_with_lcl",
    # ._ufunc
    "delta_t",
    "dewpoint",
    "dry_lapse",
    "equivalent_potential_temperature",
    "lcl",
    "lcl_pressure",
    "potential_temperature",
    "saturation_mixing_ratio",
    "saturation_vapor_pressure",
    "wet_bulb_temperature",
    "wet_bulb_potential_temperature",
    "dewpoint_from_specific_humidity",
    "lfc",
    "parcel_profile",
    "mixing_ratio",
    "vapor_pressure",
    "virtual_temperature",
    "wind_components",
    "wind_direction",
    "wind_magnitude",
    "wind_vector",
    # .core
    "el",
    "cape_cin",
    "ccl",
    "downdraft_cape",
    "most_unstable_parcel",
    "most_unstable_parcel_index",
    "most_unstable_cape_cin",
    # .utils
    "timeseries",
]

try:
    from ._version import __version__
except ImportError:
    __version__ = "undefined"

from . import functional
from ._core import (
    E0,
    OPENMP_ENABLED,
    P0,
    T0,
    Cp,
    Lv,
    Md,
    Mw,
    Rd,
    Rv,
    epsilon,
    kappa,
    moist_lapse,
    parcel_profile,
    parcel_profile_with_lcl,
)
from ._ufunc import (
    delta_t,
    dewpoint,
    dewpoint_from_specific_humidity,
    dry_lapse,
    equivalent_potential_temperature,
    lcl,
    lcl_pressure,
    potential_temperature,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    vapor_pressure,
    virtual_temperature,
    wet_bulb_potential_temperature,
    wet_bulb_temperature,
    wind_components,
    wind_direction,
    wind_magnitude,
    wind_vector,
)
from .core import (
    cape_cin,
    ccl,
    downdraft_cape,
    el,
    lfc,
    mixing_ratio,
    most_unstable_parcel,
    most_unstable_parcel_index,
    most_unstable_cape_cin,
)
from .utils import timeseries
