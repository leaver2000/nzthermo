from typing import Annotated, TypeVar

from ._typing import _ufunc1x1, _ufunc2x1, _ufunc2x2, _ufunc3x1
from .typing import Kelvin, Pascal

_T = TypeVar("_T")

@_ufunc2x1
def delta_t(year: int, month: int) -> float: ...

Theta = Annotated[_T, "Potential temperature"]
ThetaE = Annotated[_T, "Equivalent potential temperature"]
ThetaW = Annotated[_T, "Wet bulb potential temperature"]

# ............................................................................................... #
# ............................................................................................... #
@_ufunc1x1
def dewpoint(vapor_pressure: Pascal[float]) -> Kelvin[float]: ...
@_ufunc1x1
def saturation_vapor_pressure(temperature: Kelvin[float]) -> Pascal[float]: ...
@_ufunc2x1
def saturation_mixing_ratio(pressure: Pascal[float], temperature: Kelvin[float]) -> float: ...
@_ufunc2x1
def virtual_temperature(temperature: Kelvin[float], vapor_pressure: Pascal[float]) -> Kelvin[float]: ...
def potential_temperature(pressure: Pascal[float], temperature: Kelvin[float]) -> Theta[Kelvin[float]]: ...
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
@_ufunc2x1
def wind_direction(u: float, v: float) -> float: ...
@_ufunc2x1
def wind_magnitude(u: float, v: float) -> float: ...
@_ufunc2x2
def wind_components(direction: float, speed: float) -> tuple[float, float]: ...
@_ufunc1x1
def wobus(temperature: Kelvin[float]) -> float: ...
@_ufunc3x1
def dry_lapse(
    pressure: Pascal[float], temperature: Kelvin[float], reference_pressure: Pascal[float]
) -> Kelvin[float]: ...
