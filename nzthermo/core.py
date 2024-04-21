# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
This is an implementation of the metpy thermodynamics module but without the pint requirement and
additional support for higher dimensions in what would have normally been 1D arrays
"""

from __future__ import annotations

from typing import (
    Annotated,
    Any,
    Final,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from . import functional as F
from ._c import lcl, moist_lapse
from ._typing import Kelvin, Kilogram, N, Pascal, Ratio, Z, shape
from .const import *

# .....{ types }.....
T = TypeVar("T")
Pair = tuple[T, T]
dtype_t = TypeVar("dtype_t", bound=np.floating[Any], contravariant=True)
float_ = TypeVar("float_", bound=np.float_)


Array2d = np.ndarray[Pair[int], np.dtype[float_]]
Indices = NDArray[np.intp]
newaxis: Final[None] = np.newaxis


# =================================================================================================
# .....{ basic thermodynamics }.....
# None of the following require any type of array broadcasting or fancy indexing
# =================================================================================================
# arg count: 1
def dewpoint(vapor_pressure: Pascal[NDArray[float_]]) -> Kelvin[NDArray[float_]]:
    """
    ther are two ways to calculate the dewpoint temperature from the vapor pressure
    ```
    ln = np.log(e / E0)
    Td = T0 + 243.5 * ln / (17.67 - ln)

    ln = np.log(e / E0)
    Td = ((17.67 - ln) * T0 + 243.5 * ln) / (17.67 - ln)
    ```
    """
    vapor_pressure[vapor_pressure <= 0.0] = np.nan
    ln = np.log(vapor_pressure / E0)
    return T0 + 243.5 * ln / (17.67 - ln)


_dewpoint: Final = dewpoint  # alias for the dewpoint function to mitigate namespace conflicts


def saturation_vapor_pressure(temperature: Kelvin[NDArray[float_]]) -> Pascal[NDArray[float_]]:
    return E0 * np.exp(17.67 * (temperature - T0) / (temperature - 29.65))


def mixing_ratio_from_specific_humidity(specific_humidity: Kilogram[NDArray[float_]]) -> Kilogram[NDArray[float_]]:
    return specific_humidity / (1 - specific_humidity)


# arg count: 2
def vapor_pressure(
    pressure: Pascal[NDArray[float_]], mixing_ratio: Ratio[NDArray[float_] | float]
) -> Pascal[NDArray[float_]]:
    return pressure * mixing_ratio / ((Rd / Rv) + mixing_ratio)


def exner_function(
    pressure: Pascal[NDArray[float_]], refrence_pressure: Pascal[NDArray[float_] | float] = P0
) -> Pascal[NDArray[float_]]:
    r"""\Pi = \left( \frac{p}{p_0} \right)^{R_d/c_p} = \frac{T}{\theta}"""
    return (pressure / refrence_pressure) ** (Rd / Cpd)


# arg count: 3
def mixing_ratio(
    partial_press: Pascal[NDArray[float_]],
    total_press: Pascal[NDArray[float_] | float],
    molecular_weight_ratio: Ratio[NDArray[float_] | float] = Rd / Rv,
) -> Ratio[NDArray[float_]]:
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


def dry_lapse(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    reference_pressure: Pascal[NDArray[float_] | float] | None = None,
    *,
    axis: int = 0,
) -> Kelvin[NDArray[float_]]:
    """``T * (p / p_0)^{R_d / C_p}``"""
    if reference_pressure is None:
        reference_pressure = pressure[axis]
    return temperature * (pressure / reference_pressure) ** (Rd / Cpd)  # pyright: ignore


def saturation_mixing_ratio(
    pressure: Pascal[NDArray[float_]], temperature: Kelvin[NDArray[float_]]
) -> Ratio[NDArray[float_]]:
    return mixing_ratio(saturation_vapor_pressure(temperature), pressure)


# .....{ theta }.....
Theta = Annotated[T, "Potential temperature"]
ThetaE = Annotated[T, "Equivalent potential temperature"]
ThetaW = Annotated[T, "Wet bulb potential temperature"]


def potential_temperature(
    pressure: Pascal[NDArray[float_]], temperature: Kelvin[NDArray[float_]]
) -> Theta[Kelvin[NDArray[float_]]]:
    return temperature / exner_function(pressure)


def equivalent_potential_temperature(
    pressure: Pascal[NDArray[float_]], temperature: Kelvin[NDArray[float_]], dewpoint: Kelvin[NDArray[float_]]
) -> ThetaE[Kelvin[NDArray[float_]]]:
    r = saturation_mixing_ratio(pressure, dewpoint)
    e = saturation_vapor_pressure(dewpoint)
    t_l = 56 + 1.0 / (1.0 / (dewpoint - 56) + np.log(temperature / dewpoint) / 800.0)
    th_l = potential_temperature(pressure - e, temperature) * (temperature / t_l) ** (0.28 * r)
    return th_l * np.exp(r * (1 + 0.448 * r) * (3036.0 / t_l - 1.78))


# .....{ virtual temperature }.....
def virtual_temperature(
    temperature: Kelvin[NDArray[float_]],
    mixing_ratio: Ratio[NDArray[float_]],
    *,
    molecular_weight_ratio: float = Rd / Rv,
) -> Kelvin[NDArray[float_]]:
    return temperature * ((mixing_ratio + molecular_weight_ratio) / (molecular_weight_ratio * (1 + mixing_ratio)))


def dewpoint_from_specific_humidity(
    pressure: Pascal[NDArray[float_]], specific_humidity: Kilogram[NDArray[float_]], *, eps: float = Rd / Rv
) -> Kelvin[NDArray[float_]]:
    w = mixing_ratio_from_specific_humidity(specific_humidity)
    return dewpoint(pressure * w / (eps + w))


# -------------------------------------------------------------------------------------------------
# wet bulb temperature
# -------------------------------------------------------------------------------------------------
def wet_bulb_temperature(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[np.float32]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float32]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[np.float32]]],
) -> Kelvin[np.ndarray[shape[N], np.dtype[np.float32]]]:
    # TODO:...
    if not all(x.ndim == 1 for x in (pressure, temperature, dewpoint)):
        raise NotImplementedError("Currently only support for 1D arrays")
    lcl_p, lcl_t = lcl(pressure, temperature, dewpoint)

    return moist_lapse(pressure, lcl_t, lcl_p)


# -------------------------------------------------------------------------------------------------
# convective condensation level
# -------------------------------------------------------------------------------------------------
class ConvectiveCondensationLevel(NamedTuple, Generic[float_]):
    pressure: Pascal[NDArray[float_]]
    temperature: Kelvin[NDArray[float_]]
    convective_temperature: Kelvin[NDArray[float_]]

    @classmethod
    def from_intersect(
        cls,
        p0: Pascal[NDArray[float_]],
        intersect: F.Intersection[float_],
        which: Literal["lower", "upper"] = "lower",
    ) -> ConvectiveCondensationLevel[float_]:
        p, t, _ = intersect.lower() if which == "lower" else intersect.upper()
        ct = dry_lapse(p0, t, p)

        return ConvectiveCondensationLevel(p, t, ct)


@overload
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["all"] = "all",
) -> Pair[ConvectiveCondensationLevel[float_]]: ...
@overload
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["lower", "upper"] = "lower",
) -> ConvectiveCondensationLevel[float_]: ...
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["lower", "upper", "all"] = "lower",
) -> ConvectiveCondensationLevel[float_] | Pair[ConvectiveCondensationLevel[float_]]:
    """
    # Convective Condensation Level (CCL)

    The Convective Condensation Level (CCL) is the level at which condensation will occur if
    sufficient afternoon heating causes rising parcels of air to reach saturation. The CCL is
    greater than or equal in height (lower or equal pressure level) than the LCL. The CCL and the
    LCL are equal when the atmosphere is saturated. The CCL is found at the intersection of the
    saturation mixing ratio line (through the surface dewpoint) and the environmental temperature.
    """
    if temperature.ndim == 1:
        temperature = temperature[newaxis, :]
        dewpoint = dewpoint[newaxis, :]
        pressure = pressure[newaxis, :]
    elif temperature.ndim != 2:
        raise ValueError("temperature and dewpoint must be 1D or 2D arrays")

    p0 = pressure[:, 0]  # (N,)
    td0 = dewpoint[:, 0]  # (N,)
    td = _dewpoint(  # (N, Z)
        vapor_pressure(pressure, mixing_ratio(saturation_vapor_pressure(td0[:, newaxis]), p0[:, newaxis]))
    )

    intersect = F.intersect_nz(pressure, td, temperature, log_x=True)  # (N, Z)

    if which != "all":
        return ConvectiveCondensationLevel.from_intersect(p0, intersect, which)

    return (
        ConvectiveCondensationLevel.from_intersect(p0, intersect, "lower"),
        ConvectiveCondensationLevel.from_intersect(p0, intersect, "upper"),
    )


# -------------------------------------------------------------------------------------------------
# Parcel Profile
# -------------------------------------------------------------------------------------------------
def _parcel_profile(
    pressure: np.ndarray,
    temperature: np.ndarray,
    dewpoint: np.ndarray,
    refrence_pressure: np.ndarray | None = None,
    *,
    axis=-1,
):
    Z = pressure.shape[axis]
    P = np.reshape(np.moveaxis(pressure, axis, -1), (-1, Z))  # (N, Z)
    T = temperature.reshape(-1, 1)  # (N, 1)
    Td = dewpoint.reshape(-1, 1)  # (N, 1)
    N = T.shape[0]

    if refrence_pressure is None:
        P0 = P[:, :1]
    else:
        P0 = refrence_pressure.reshape(-1, 1)  # (N, 1)

    if not T.shape == Td.shape == P0.shape == (N, 1):
        raise ValueError("temperature, dewpoint, and refrence pressure must have the same shape")
    # ---------------------------------------------------------------------------------------------
    # - Find the LCL
    # ---------------------------------------------------------------------------------------------
    lcl_p, lcl_t = lcl(P0, T, Td)  # ((N, 1), (N, 1))

    # - lower
    # [[101300. 100000. 97500. 95000. 94171.484 nan ...], ...] (N, Z)
    p_mask = P >= lcl_p
    p_lower = F.mask_insert(p_mask, P, lcl_p, kind="above", copy=False)  # (N, Z)
    t_lower = dry_lapse(p_lower, T, P0)

    # - upper
    # [[... nan 97500. 95000. 94171.484 90000. ...], ...] (N, Z)
    p_upper = F.mask_insert(~p_mask, P, lcl_p, kind="below", copy=False)  # (N, Z)
    t_upper = np.full_like(t_lower, np.nan)

    nans = np.isnan(t_lower) | np.isnan(p_upper)
    nx, zx = np.nonzero(~nans)
    _, i = np.unique(nx, return_index=True)
    nx, zx = nx[i], zx[i]

    t_upper[nx, :] = moist_lapse(
        p_upper[nx, :],
        t_lower[nx, zx],
        p_upper[nx, zx],
    )

    # OKAY now concat every thing
    index = np.nonzero(p_lower.mask)
    p_lower[index] = p_upper[index]
    t_lower[index] = t_upper[index]

    return (p_lower.squeeze(), lcl_p.squeeze()), (t_lower.squeeze(), lcl_t.squeeze())


class ParcelProfile(NamedTuple, Generic[float_]):
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    lcl_pressure: Annotated[
        Pascal[np.ndarray[shape[N], np.dtype[float_]]],
        "",
    ]

    temperature: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    parcel_temperature: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    lcl_temperature: Annotated[
        Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
        "",
    ]

    dewpoint: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]


def parcel_profile(
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    *,
    pressure_2m: Pascal[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
    temperature_2m: Kelvin[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
    dewpoint_2m: Kelvin[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
) -> ParcelProfile[float_]:
    assert pressure.ndim == 1
    P = pressure.reshape(1, -1)  # (1, Z)
    T, Td = temperature, dewpoint
    Z = T.shape[1]
    Z += 1  # add one for the 2m values

    if pressure_2m is not None:
        P2 = pressure_2m.reshape(-1, 1)
    else:
        P2 = P[:, :1]

    if temperature_2m is None:
        T2 = T[:, :1]
    else:
        T2 = temperature_2m.reshape(-1, 1)

    if dewpoint_2m is None:
        Td2 = Td[:, :1]
    else:
        Td2 = dewpoint_2m.reshape(-1, 1)

    (pp_p, lcl_p), (pp_t, lcl_t) = _parcel_profile(P, T2, Td2, P2, axis=1)
    if pp_t.shape == (N, Z - 1):  # concat the 2m values
        pp_t = np.concatenate([T2, pp_t], axis=1)

    interp_t, interp_td = F.interpolate_nz(lcl_p, pressure, T, Td, interp_nan=True)

    t_new = F.insert_along_z(T, interp_t, pressure, lcl_p)
    td_new = F.insert_along_z(Td, interp_td, pressure, lcl_p)

    return ParcelProfile(
        pressure=pp_p,
        lcl_pressure=lcl_p,
        temperature=pp_t,
        parcel_temperature=t_new,
        lcl_temperature=lcl_t,
        dewpoint=td_new,
    )


# -------------------------------------------------------------------------------------------------
# downdraft_cape
# -------------------------------------------------------------------------------------------------
def downdraft_cape(
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
) -> np.ndarray[shape[N], np.dtype[float_]]:
    dtype = temperature.dtype
    pressure, temperature, dewpoint = (x.astype(dtype) for x in (pressure, temperature, dewpoint))

    assert temperature.shape == dewpoint.shape
    N, Z = temperature.shape
    if pressure.shape == (Z,):
        pressure = pressure.reshape(1, Z)
    elif pressure.shape != (1, Z):
        raise ValueError("pressure shape must be either (Z,) or (1, Z)")

    mid_layer_idx = ((pressure <= 7e4) & (pressure >= 5e4)).squeeze()
    p_layer, t_layer, td_layer = (x[:, mid_layer_idx] for x in (pressure, temperature, dewpoint))

    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    nx, zx = np.arange(N), np.argmin(theta_e, axis=1)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    p_top = p_layer[0, zx]  # (N,)
    t_top = t_layer[nx, zx]  # (N,)
    td_top = td_layer[nx, zx]  # (N,)
    wb_top = wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

    # reshape our pressure into a 2d pressure grid and put the hard cap on everything above the hard cap
    cap = -(np.searchsorted(np.squeeze(pressure)[::-1], np.min(p_top)) - 1)
    pressure = pressure[0, :cap].repeat(N).reshape(-1, N).transpose()  # (N, Z -cap)
    if not np.any(pressure):
        return np.repeat(np.nan, N).astype(dtype)

    # our moist_lapse rate function has nan ignoring capabilities
    pressure[pressure < p_top[:, newaxis]] = np.nan
    temperature = temperature[:, :cap]
    dewpoint = dewpoint[:, :cap]

    trace = moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    e_vt = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))  # (N, Z)
    p_vt = virtual_temperature(trace, saturation_mixing_ratio(pressure, trace))  # (N, Z)

    delta = e_vt - p_vt
    logp = np.log(pressure)

    # the np.trapz function does not support np.nan values, but we only really have a couple of unique
    # levels in the gridded dataset. So this is really just like a loop of 4, 5 iterations
    dcape = np.zeros(N, dtype=dtype)  # (N,)
    batch, levels = np.nonzero(np.diff(pressure > p_top[:, newaxis], prepend=True, append=False))
    for lvl in np.unique(levels):
        above = levels == lvl
        nx = batch[above]  # the sample indices
        zx = np.s_[: np.argmin(~np.isnan(pressure[above]).any(axis=0))]
        dcape[nx] = -(Rd * np.trapz(delta[nx, zx], logp[nx, zx], axis=1))

    return dcape
