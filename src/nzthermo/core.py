# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
The calculations in this module are based on the MetPy library. These calculations supplement
the original implementation in areas where previously only support for `1D profiles
(not higher-dimension vertical cross sections or grids)`.
"""

from __future__ import annotations

from typing import Any, Literal as L, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import functional as F
from ._core import Rd, moist_lapse, parcel_profile_with_lcl
from ._ufunc import (
    dewpoint as _dewpoint,
    dry_lapse,
    equivalent_potential_temperature,
    exner_function,
    greater_or_close,
    lcl,
    lcl_pressure,
    mixing_ratio,
    potential_temperature,
    pressure_vector,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    vapor_pressure,
    virtual_temperature,
    wet_bulb_temperature,
)
from .typing import Kelvin, N, Pascal, Z, shape
from .utils import Parcel, broadcast_nz

_S = TypeVar("_S", bound=shape)
_T = TypeVar("_T", bound=np.floating[Any], covariant=True)


FASTPATH: dict[str, Any] = {"__fastpath": True}


@broadcast_nz
def parcel_mixing_ratio(
    pressure: Pascal[pressure_vector[_S, np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[_S, np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[_S, np.dtype[_T]]],
    /,
    *,
    where: np.ndarray[_S, np.dtype[np.bool_]] | None = None,
) -> np.ndarray[_S, np.dtype[_T]]:
    r"""Calculate the parcel mixing ratio.

    Calculate the mixing ratio of a parcel given the pressure, temperature, and dewpoint. The
    mixing ratio is the ratio of the mass of water vapor to the mass of dry air in a given
    volume of air. This function calculates the mixing ratio of a parcel of air given the
    pressure, temperature, and dewpoint of the parcel.


    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    where : array_like[(N, Z), bool], optional
        Where ``True`` the saturation mixing ratio is calculated, using the dewpoint and where
        ``False`` the mixing ratio is calculated using the temperature. If not provided the
        mixing ratio is calculated using the dewpoint upto the LCL (lifting condensation level)
        and the temperature above the LCL.

    Returns
    -------
    parcel_mixing_ratio : array_like[(N, Z), floating]
        Parcel mixing ratio (kg/kg)

    """
    if where is None:
        where = pressure.is_below(
            lcl_pressure(pressure[:, :1], temperature[:, :1], dewpoint[:, :1])
        )

    r = saturation_mixing_ratio(
        pressure,
        temperature,
        out=saturation_mixing_ratio(pressure, dewpoint, where=where),
        where=~where,
    )

    return r


# -------------------------------------------------------------------------------------------------
# convective condensation level
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def ccl(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    height: ArrayLike | None = None,
    mixed_layer_depth: ArrayLike | None = None,
    which: L["bottom", "top"] = "bottom",
) -> tuple[Pascal[NDArray[_T]], Kelvin[NDArray[_T]], Kelvin[NDArray[_T]]]:
    r"""Calculate convective condensation level (CCL) and convective temperature.

    The Convective Condensation Level (CCL) is the level at which condensation will occur if
    sufficient afternoon heating causes rising parcels of air to reach saturation. The CCL is
    greater than or equal in height (lower or equal pressure level) than the LCL. The CCL and the
    LCL are equal when the atmosphere is saturated. The CCL is found at the intersection of the
    saturation mixing ratio line (through the surface dewpoint) and the environmental temperature.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    height : array_like[(N,), floating], optional

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...

    """
    if mixed_layer_depth is None:
        r = mixing_ratio(saturation_vapor_pressure(dewpoint[:, :1]), pressure[:, :1])
    else:
        r = mixed_layer(
            pressure,
            mixing_ratio(saturation_vapor_pressure(dewpoint), pressure),
            height=height,
            depth=mixed_layer_depth,
            # TODO:
            # the metpy implementation forces interpolation here which is not implemented
            # currently with our mixed layer function
            interpolate=True,
        )[0][:, np.newaxis]

    if height is not None:
        raise NotImplementedError

    rt_profile = _dewpoint(vapor_pressure(pressure, r))

    p, t = F.intersect_nz(pressure, rt_profile, temperature, "increasing", log_x=True).pick(which)

    return p, t, dry_lapse(pressure[:, 0], t, p)


# -------------------------------------------------------------------------------------------------
# downdraft_cape
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def downdraft_cape(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    where: np.ndarray[shape[N, Z], np.dtype[np.bool_]] | None = None,
) -> np.ndarray[shape[N], np.dtype[_T]]:
    r"""Calculate downward CAPE (DCAPE).

    Calculate the downward convective available potential energy (DCAPE) of a given upper air
    profile. Downward CAPE is the maximum negative buoyancy energy available to a descending
    parcel. Parcel descent is assumed to begin from the lowest equivalent potential temperature
    between 700 and 500 hPa. This parcel is lowered moist adiabatically from the environmental
    wet bulb temperature to the surface.  This assumes the parcel remains saturated
    throughout the descent.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    where : array_like[(N, Z), bool], optional

    Returns
    -------
    DCAPE : array_like[(N,), floating]
        Downward Convective Available Potential Energy (J/kg)

    Examples
    --------
    TODO: add examples

    """
    N, _ = temperature.shape
    nx = np.arange(N)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    if where is None:
        where = (pressure <= 70000.0) & (pressure >= 50000.0)

    theta_e = equivalent_potential_temperature(
        pressure,
        temperature,
        dewpoint,
        where=where,
        out=np.full_like(temperature, np.inf),
    )
    zx = theta_e.argmin(axis=1)

    p_top = pressure[0 if pressure.shape[0] == 1 else nx, zx]
    t_top = temperature[nx, zx]  # (N,)
    td_top = dewpoint[nx, zx]  # (N,)
    wb_top = wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

    pressure = pressure.where(pressure.is_below(p_top[:, np.newaxis], close=True), np.nan)
    e_vt = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))  # (N, Z)
    trace = moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    p_vt = virtual_temperature(trace, saturation_mixing_ratio(pressure, trace))  # (N, Z)

    DCAPE = Rd * F.nantrapz(p_vt - e_vt, np.log(pressure), axis=1)

    return DCAPE


# -------------------------------------------------------------------------------------------------
# el & lfc
# -------------------------------------------------------------------------------------------------
def _el_lfc(
    pick: L["EL", "LFC", "BOTH"],
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    parcel_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]] | None = None,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[_T]] | None = None,
) -> tuple[Parcel[_T], Parcel[_T]] | Parcel[_T]:
    r"""short cut for el, lfc, and el_lfc functions."""
    if parcel_profile is None:
        pressure, temperature, dewpoint, parcel_profile = parcel_profile_with_lcl(
            pressure, temperature, dewpoint
        )

    N = temperature.shape[0]
    p0, t0 = pressure[:, 0], temperature[:, 0]
    if dewpoint_start is None:
        td0 = dewpoint[:, 0]  # (N,)
    else:
        td0 = dewpoint_start

    LCL = Parcel.from_func(lcl, p0, t0, td0).unsqueeze()

    pressure, parcel_profile, temperature = (
        pressure[:, 1:],
        parcel_profile[:, 1:],
        temperature[:, 1:],
    )

    if pick != "LFC":  # find the Equilibrium Level (EL)
        top_idx = np.arange(N), np.argmin(~np.isnan(pressure), axis=1) - 1
        left_of_env = (parcel_profile[top_idx] <= temperature[top_idx])[:, np.newaxis]
        EL = F.intersect_nz(
            pressure,
            parcel_profile,
            temperature,
            "decreasing",
            log_x=True,
        ).where(
            # If the top of the sounding parcel is warmer than the environment, there is no EL
            lambda el: el.is_above(LCL) & left_of_env
        )

        if pick == "EL":
            return EL.pick(which_el)

    LFC = F.intersect_nz(
        pressure,
        parcel_profile,
        temperature,
        "increasing",
        log_x=True,
    ).where_above(LCL)

    no_lfc = LFC.is_nan().all(axis=1, out=np.empty((N, 1), dtype=np.bool_), keepdims=True)

    is_lcl = no_lfc & greater_or_close(
        # the mask only needs to be applied to either the temperature or parcel_temperature_profile
        np.where(LCL.is_below(pressure, close=True), parcel_profile, np.nan),
        temperature,
    ).any(axis=1, out=np.empty((N, 1), dtype=np.bool_), keepdims=True)

    LFC = LFC.select(
        [~no_lfc, is_lcl],
        [LFC.pressure, LCL.pressure],
        [LFC.temperature, LCL.temperature],
    )

    if pick == "LFC":
        return LFC.pick(which_lfc)

    return EL.pick(which_el), LFC.pick(which_lfc)


@broadcast_nz
def el(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    parcel_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]] | None = None,
    *,
    which: L["top", "bottom"] = "top",
) -> Parcel[_T]:
    r"""Calculate the equilibrium level (EL).

    This works by finding the last intersection of the ideal parcel path and the measured
    environmental temperature. If there is one or fewer intersections, there is no equilibrium
    level.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    parcel_profile : array_like[[N, Z], floating], optional
        The parcel's temperature profile from which to calculate the EL. Defaults to the
        surface parcel profile.
    which : `str`, optional
        Pick which EL to return. Options are `top` or `bottom`. Default is `top`.
        'top' returns the lowest-pressure EL, default.
        'bottom' returns the highest-pressure EL.
        NOT IMPLEMENTED YET:
        'wide' returns the EL whose corresponding LFC is farthest away.
        'most_cape' returns the EL that results in the most CAPE in the profile.

    Returns
    -------
    EL : `(ndarray[[N], floating], ndarray[[N], floating])`

    Examples
    --------
    TODO : add examples

    See Also
    --------
    lfc
    parcel_profile
    el_lfc
    """

    return _el_lfc("EL", pressure, temperature, dewpoint, parcel_profile, which_el=which)


@broadcast_nz
def lfc(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    parcel_profile: np.ndarray | None = None,
    *,
    which: L["top", "bottom"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[_T]] | None = None,
) -> Parcel[_T]:
    return _el_lfc(
        "LFC",
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc=which,
        dewpoint_start=dewpoint_start,
    )


@broadcast_nz
def el_lfc(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    parcel_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]] | None = None,
    *,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[_T]] | None = None,
) -> tuple[Parcel[_T], Parcel[_T]]:
    r"""TODO ...

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    return _el_lfc(
        "BOTH",
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc=which_lfc,
        which_el=which_el,
        dewpoint_start=dewpoint_start,
    )


# -------------------------------------------------------------------------------------------------
# cape_cin
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def cape_cin(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    parcel_profile: np.ndarray,
    *,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate CAPE and CIN.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    parcel_profile : array_like[(N, Z), floating]

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    # The mixing ratio of the parcel comes from the dewpoint below the LCL, is saturated
    # based on the temperature above the LCL

    parcel_profile = virtual_temperature(
        parcel_profile, parcel_mixing_ratio(pressure, temperature, dewpoint, **FASTPATH)
    )
    temperature = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))
    # Calculate the EL limit of integration
    (EL, _), (LFC, _) = _el_lfc(
        "BOTH",
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc,
        which_el,
    )
    EL, LFC = np.reshape((EL, LFC), (2, -1, 1))  # reshape for broadcasting

    tzx = F.zero_crossings(pressure, parcel_profile - temperature)  # temperature zero crossings

    p, t = tzx.where_between(LFC, EL, close=True)
    CAPE = Rd * F.nantrapz(t, np.log(p), axis=1)
    CAPE[CAPE < 0.0] = 0.0

    p, t = tzx.where_below(LFC, close=True)
    CIN = Rd * F.nantrapz(t, np.log(p), axis=1)
    CIN[CIN > 0.0] = 0.0

    return CAPE, CIN


# -------------------------------------------------------------------------------------------------
# most_unstable
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def most_unstable_parcel_index(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    depth: float = 30000.0,
    height: float | None = None,
    bottom: float | None = None,
) -> np.ndarray[shape[N], np.dtype[np.intp]]:
    """TODO ...

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    if height is not None:
        raise NotImplementedError("height argument is not implemented")

    pbot = (pressure[:, :1] if bottom is None else np.asarray(bottom)).reshape(-1, 1)
    ptop = pbot - depth

    theta_e = equivalent_potential_temperature(
        pressure,
        temperature,
        dewpoint,
        where=pressure.is_between(pbot, ptop),
        out=np.full(temperature.shape, -np.inf, dtype=temperature.dtype),
    )

    return np.argmax(theta_e, axis=1)


@broadcast_nz
def most_unstable_parcel(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    depth: Pascal[float] = 30000.0,
    bottom: Pascal[float] | None = None,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[_T]]],
    Kelvin[np.ndarray[shape[N], np.dtype[_T]]],
    Kelvin[np.ndarray[shape[N], np.dtype[_T]]],
    np.ndarray[shape[N, Z], np.dtype[np.intp]],
]:
    r"""Calculate the most unstable parcel profile.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    idx = most_unstable_parcel_index(
        pressure, temperature, dewpoint, depth=depth, bottom=bottom, **FASTPATH
    )

    return (
        pressure[np.arange(pressure.shape[0]), idx],
        temperature[np.arange(temperature.shape[0]), idx],
        dewpoint[np.arange(dewpoint.shape[0]), idx],
        idx,
    )


@broadcast_nz
def most_unstable_cape_cin(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    depth: Pascal[float] = 30000.0,
    bottom: Pascal[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Calculate most unstable CAPE and CIN.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    idx = most_unstable_parcel_index(
        pressure,
        temperature,
        dewpoint,
        depth,
        bottom,
        **FASTPATH,
    )
    mask = np.arange(pressure.shape[1]) >= idx[:, np.newaxis]

    p, t, td, mu_profile = parcel_profile_with_lcl(
        pressure,
        temperature,
        dewpoint,
        where=mask,
    )

    return cape_cin(p.view(pressure_vector), t, td, parcel_profile=mu_profile, **FASTPATH)


# -------------------------------------------------------------------------------------------------
# mixed_layer
# -------------------------------------------------------------------------------------------------
def mixed_layer(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *args: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    depth: ArrayLike = 10000.0,
    height: ArrayLike | None = None,
    bottom: ArrayLike | None = None,
    where: ArrayLike | None = None,
    interpolate: bool = False,
) -> tuple[np.ndarray[shape[N], np.dtype[_T]], ...]:
    r"""Calculate the mean temperature and dewpoint of a mixed layer.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    pressure, *args = F.exactly_2d(pressure, *args)

    if height is not None:
        raise NotImplementedError("height argument is not implemented")
    if interpolate:
        raise NotImplementedError("interpolate argument is not implemented")

    if where is None:
        bottom = (pressure[:, :1] if bottom is None else np.asarray(bottom)).reshape(-1, 1)
        top = bottom - np.asarray(depth)
        where = pressure.view(pressure_vector).is_between(bottom, top)
    else:
        where = np.asarray(where, dtype=np.bool_)

    depth = np.asarray(
        # use asarray otherwise the depth is cast to pressure_vector which doesn't
        # make sense for the temperature and dewpoint outputs
        np.max(pressure, initial=-np.inf, axis=1, where=where)
        - np.min(pressure, initial=np.inf, axis=1, where=where)
    )

    return tuple(F.nantrapz(args, pressure, axis=-1, where=where) / -depth)


@broadcast_nz
def mixed_parcel(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    parcel_start_pressure: ArrayLike | None = None,
    height: ArrayLike | None = None,
    bottom: ArrayLike | None = None,
    depth: float | NDArray[np.floating[Any]] = 10000.0,
    interpolate=False,
) -> tuple[
    np.ndarray[shape[N], np.dtype[_T]],
    np.ndarray[shape[N], np.dtype[_T]],
    Kelvin[np.ndarray[shape[N], np.dtype[_T]]],
]:
    r"""Calculate the mean temperature and dewpoint of a mixed parcel.

    Parameters
    ----------
    pressure : array_like[(N, Z), floating]
        Atmospheric pressure profile (Pa). This array must be from high to low pressure.
    temperature : array_like[(N, Z), floating]
        Temperature (K) at the levels given by `pressure`
    dewpoint : array_like[(N, Z), floating]
        Dewpoint (K) at the levels given by `pressure`
    TODO : ...

    Returns
    -------
    TODO : ...

    Examples
    --------
    TODO : ...
    """
    if height is not None:
        raise NotImplementedError("height argument is not implemented")
    if interpolate:
        raise NotImplementedError("interpolate argument is not implemented")
    if parcel_start_pressure is None:
        parcel_start_pressure = pressure[:, 0]

    theta = potential_temperature(pressure, temperature)
    mixing_ratio = saturation_mixing_ratio(pressure, dewpoint)
    mean_theta, mean_mixing_ratio = mixed_layer(
        pressure,
        theta,
        mixing_ratio,
        bottom=bottom,
        height=height,
        depth=depth,
        interpolate=interpolate,
    )
    mean_temperature = mean_theta * exner_function(parcel_start_pressure)
    mean_dewpoint = _dewpoint(vapor_pressure(parcel_start_pressure, mean_mixing_ratio))

    return (
        np.broadcast_to(parcel_start_pressure, mean_temperature.shape),
        mean_temperature,
        mean_dewpoint,
    )


@broadcast_nz
def mixed_layer_cape_cin(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    *,
    parcel_start_pressure: ArrayLike | None = None,
    height: ArrayLike | None = None,
    bottom: ArrayLike | None = None,
    depth: float | NDArray[np.floating[Any]] = 10000.0,
    interpolate=False,
):
    r"""Calculate mixed-layer CAPE and CIN.

    Calculate the convective available potential energy (CAPE) and convective inhibition (CIN)
    of a given upper air profile and mixed-layer parcel path. CIN is integrated between the
    surface and LFC, CAPE is integrated between the LFC and EL (or top of sounding).
    Intersection points of the measured temperature profile and parcel profile are
    logarithmically interpolated. Kwargs for `mixed_parcel` can be provided, such as `depth`.
    Default mixed-layer depth is 100 hPa.

    Parameters
    ----------
    pressure : `pint.Quantity`
        Pressure profile

    temperature : `pint.Quantity`
        Temperature profile

    dewpoint : `pint.Quantity`
        Dewpoint profile

    kwargs
        Additional keyword arguments to pass to `mixed_parcel`

    Returns
    -------
    `pint.Quantity`
        Mixed-layer Convective Available Potential Energy (CAPE)
    `pint.Quantity`
        Mixed-layer Convective INhibition (CIN)

    Examples
    --------
    >>> from metpy.calc import dewpoint_from_relative_humidity, mixed_layer_cape_cin
    >>> from metpy.units import units
    >>> # pressure
    >>> p = [1008., 1000., 950., 900., 850., 800., 750., 700., 650., 600.,
    ...      550., 500., 450., 400., 350., 300., 250., 200.,
    ...      175., 150., 125., 100., 80., 70., 60., 50.,
    ...      40., 30., 25., 20.] * units.hPa
    >>> # temperature
    >>> T = [29.3, 28.1, 25.5, 20.9, 18.4, 15.9, 13.1, 10.1, 6.7, 3.1,
    ...      -0.5, -4.5, -9.0, -14.8, -21.5, -29.7, -40.0, -52.4,
    ...      -59.2, -66.5, -74.1, -78.5, -76.0, -71.6, -66.7, -61.3,
    ...      -56.3, -51.7, -50.7, -47.5] * units.degC
    >>> # relative humidity
    >>> rh = [.85, .75, .56, .39, .82, .72, .75, .86, .65, .22, .52,
    ...       .66, .64, .20, .05, .75, .76, .45, .25, .48, .76, .88,
    ...       .56, .88, .39, .67, .15, .04, .94, .35] * units.dimensionless
    >>> # calculate dewpoint
    >>> Td = dewpoint_from_relative_humidity(T, rh)
    >>> mixed_layer_cape_cin(p, T, Td, depth=50 * units.hPa)
    (<Quantity(711.239032, 'joule / kilogram')>, <Quantity(-5.48053989, 'joule / kilogram')>)

    See Also
    --------
    cape_cin, mixed_parcel, parcel_profile

    Notes
    -----
    Only functions on 1D profiles (not higher-dimension vertical cross sections or grids).
    Since this function returns scalar values when given a profile, this will return Pint
    Quantities even when given xarray DataArray profiles.

    """
    if height is not None:
        raise NotImplementedError("height argument is not implemented")
    if interpolate:
        raise NotImplementedError("interpolate argument is not implemented")

    start_p = np.atleast_1d(
        pressure[:, 0] if parcel_start_pressure is None else parcel_start_pressure
    ).reshape(-1, 1)

    parcel_pressure, parcel_temp, parcel_dewpoint = np.reshape(
        mixed_parcel(
            pressure,
            temperature,
            dewpoint,
            parcel_start_pressure=parcel_start_pressure,
            **FASTPATH,
        ),
        (3, -1, 1),
    )

    pressure, temperature, dewpoint = np.where(
        pressure <= (start_p - depth),
        [np.broadcast_to(pressure, temperature.shape), temperature, dewpoint],
        -np.inf,
    )
    pressure, temperature, dewpoint = F.map_partial(
        np.concatenate,
        [
            (parcel_pressure, pressure),
            (parcel_temp, temperature),
            (parcel_dewpoint, dewpoint),
        ],
        axis=1,
    )
    pressure, temperature, dewpoint = F.sort_nz(np.isneginf, pressure, temperature, dewpoint)
    pressure, temperature, dewpoint, profile = parcel_profile_with_lcl(
        pressure, temperature, dewpoint
    )

    CAPE, CIN = cape_cin(pressure, temperature, dewpoint, profile, **FASTPATH)

    return CAPE, CIN
