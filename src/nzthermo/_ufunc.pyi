import enum
from typing import Annotated, TypeVar

from ._typing import _ufunc1x1, _ufunc2x1, _ufunc2x2, _ufunc3x1, _ufunc3x2
from .typing import Dimensionless, Kelvin, Meter, Pascal

_T = TypeVar("_T")

_Jkg = enum.Enum("J/kg", float)

@_ufunc2x1
def less_or_close(x: float, y: float) -> bool: ...
@_ufunc2x1
def greater_or_close(x: float, y: float) -> bool: ...
@_ufunc3x1
def between_or_close(x: float, y0: float, y1: float) -> bool: ...
@_ufunc2x1
def delta_t(year: int, month: int) -> float: ...

Theta = Annotated[_T, "Potential temperature"]
ThetaE = Annotated[_T, "Equivalent potential temperature"]
ThetaW = Annotated[_T, "Wet bulb potential temperature"]

# wind
@_ufunc2x1
def wind_direction(u: float, v: float) -> float: ...
@_ufunc2x1
def wind_magnitude(u: float, v: float) -> float: ...
@_ufunc2x2
def wind_vector(u: float, v: float) -> tuple[float, float]: ...
@_ufunc2x2
def wind_components(direction: float, speed: float) -> tuple[float, float]: ...

# 1x1
@_ufunc1x1
def standard_height(pressure: Pascal[float]) -> Meter[float]: ...
@_ufunc1x1
def standard_pressure(height: Meter[float]) -> Pascal[float]: ...
@_ufunc1x1
def exner_function(pressure: Pascal[float]) -> Pascal[float]: ...
@_ufunc1x1
def dewpoint(vapor_pressure: Pascal[float]) -> Kelvin[float]: ...
@_ufunc1x1
def saturation_vapor_pressure(temperature: Kelvin[float]) -> Pascal[float]: ...
@_ufunc1x1
def wobus(temperature: Kelvin[float]) -> float: ...

# 2x1
@_ufunc2x1
def mixing_ratio(partial_pressure: Pascal[float], total_pressure: Pascal[float]) -> float: ...
@_ufunc2x1
def potential_temperature(
    pressure: Pascal[float], temperature: Kelvin[float]
) -> Theta[Kelvin[float]]: ...
@_ufunc2x1
def saturation_mixing_ratio(
    pressure: Pascal[float], temperature: Kelvin[float]
) -> Dimensionless[float]: ...
@_ufunc2x1
def virtual_temperature(
    temperature: Kelvin[float], vapor_pressure: Pascal[float]
) -> Kelvin[float]: ...
@_ufunc2x1
def vapor_pressure(
    pressure: Pascal[float], mixing_ratio: Dimensionless[float]
) -> Pascal[float]: ...
@_ufunc2x1
def dewpoint_from_specific_humidity(
    pressure: Pascal[float], specific_humidity: Dimensionless[float]
) -> Kelvin[float]: ...
@_ufunc2x1
def dry_static_energy(height: Meter[float], temperature: Kelvin[float]) -> _Jkg: ...

# 3x1
@_ufunc3x1
def lcl_pressure(
    pressure: Pascal[float], temperature: Kelvin[float], dewpoint: Kelvin[float]
) -> ThetaE[Pascal[float]]: ...
@_ufunc3x1
def equivalent_potential_temperature(
    pressure: Pascal[float], temperature: Kelvin[float], dewpoint: Kelvin[float]
) -> ThetaE[Kelvin[float]]: ...
@_ufunc3x1
def wet_bulb_potential_temperature(
    pressure: Pascal[float], temperature: Kelvin[float], dewpoint: Kelvin[float]
) -> ThetaW[Kelvin[float]]: ...
@_ufunc3x1
def wet_bulb_temperature(
    pressure: Pascal[float], temperature: Kelvin[float], dewpoint: Kelvin[float]
) -> Kelvin[float]: ...
@_ufunc3x1
def dry_lapse(
    pressure: Pascal[float], temperature: Kelvin[float], reference_pressure: Pascal[float]
) -> Kelvin[float]: ...
@_ufunc3x1
def moist_static_energy(
    height: Meter[float], temperature: Kelvin[float], specific_humidity: Dimensionless[float]
) -> _Jkg: ...

# 3x2
@_ufunc3x2
def lcl(
    pressure: Pascal[float], temperature: Kelvin[float], dewpoint: Kelvin[float]
) -> tuple[Pascal[float], Kelvin[float]]: ...
