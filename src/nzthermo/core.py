# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
The calculations in this module are based on the MetPy library. These calculations supplement
the original implementation in areas where previously only support for `1D profiles
(not higher-dimension vertical cross sections or grids)`.
"""

from __future__ import annotations

from typing import Any, Final, Literal as L, TypeVar

import numpy as np
from numpy.typing import NDArray

from . import _core as core, functional as F
from ._ufunc import (
    between_or_close as _between_or_close,
    dewpoint as _dewpoint,
    dry_lapse,
    equivalent_potential_temperature,
    greater_or_close as _greater_or_close,
    lcl,
    lcl_pressure,
    less_or_close as _less_or_close,
    mixing_ratio,
    saturation_mixing_ratio,
    saturation_vapor_pressure,
    vapor_pressure,
    virtual_temperature,
    wet_bulb_temperature,
)
from .const import Rd
from .typing import Kelvin, N, Pascal, Z, shape
from .utils import Vector1d, broadcast_nz

float_ = TypeVar("float_", np.float_, np.floating[Any], covariant=True)
newaxis: Final[None] = np.newaxis

_FASTPATH: dict[str, Any] = {"__fastpath": True}
between_or_close = lambda x, a, b: _between_or_close(x, a, b).astype(np.bool_)
greater_or_close = lambda x, y: _greater_or_close(x, y).astype(np.bool_)
less_or_close = lambda x, y: _less_or_close(x, y).astype(np.bool_)


# -------------------------------------------------------------------------------------------------
# downdraft_cape
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def downdraft_cape(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
) -> np.ndarray[shape[N], np.dtype[float_]]:
    """Calculate downward CAPE (DCAPE).

    Calculate the downward convective available potential energy (DCAPE) of a given upper air
    profile. Downward CAPE is the maximum negative buoyancy energy available to a descending
    parcel. Parcel descent is assumed to begin from the lowest equivalent potential temperature
    between 700 and 500 hPa. This parcel is lowered moist adiabatically from the environmental
    wet bulb temperature to the surface.  This assumes the parcel remains saturated
    throughout the descent.

    Parameters
    ----------
    TODO: add parameters

    Examples
    --------
    TODO: add examples

    """
    N, _ = temperature.shape

    mid_layer_idx = ((pressure <= 7e4) & (pressure >= 5e4)).squeeze()
    p_layer, t_layer, td_layer = (x[:, mid_layer_idx] for x in (pressure, temperature, dewpoint))

    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    nx, zx = np.arange(N), np.argmin(theta_e, axis=1)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    p_top = p_layer[0, zx]  # (N,)
    t_top = t_layer[nx, zx]  # (N,)
    td_top = td_layer[nx, zx]  # (N,)
    wb_top = wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

    # reshape our pressure into a 2d pressure grid and put the hard cap on everything above the
    # hard cap
    cap = -(np.searchsorted(np.squeeze(pressure)[::-1], np.min(p_top)) - 1)
    pressure = pressure[0, :cap].repeat(N).reshape(-1, N).transpose()  # (N, Z -cap)
    if not np.any(pressure):
        return np.repeat(np.nan, N).astype(pressure.dtype)

    # our moist_lapse rate function has nan ignoring capabilities
    pressure[pressure < p_top[:, newaxis]] = np.nan
    temperature = temperature[:, :cap]
    dewpoint = dewpoint[:, :cap]

    e_vt = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))  # (N, Z)
    trace = core.moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    p_vt = virtual_temperature(trace, saturation_mixing_ratio(pressure, trace))  # (N, Z)

    delta = e_vt - p_vt

    dcape = -(Rd * F.nantrapz(delta, np.log(pressure), axis=1))

    return dcape


# -------------------------------------------------------------------------------------------------
# convective condensation level
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    /,
    height=None,
    mixed_layer_depth=None,
    which: L["bottom", "top"] = "bottom",
):
    """
    # Convective Condensation Level (CCL)

    The Convective Condensation Level (CCL) is the level at which condensation will occur if
    sufficient afternoon heating causes rising parcels of air to reach saturation. The CCL is
    greater than or equal in height (lower or equal pressure level) than the LCL. The CCL and the
    LCL are equal when the atmosphere is saturated. The CCL is found at the intersection of the
    saturation mixing ratio line (through the surface dewpoint) and the environmental temperature.
    """

    p0 = pressure[:, 0]  # (N,)
    td0 = dewpoint[:, 0]  # (N,)

    if mixed_layer_depth is None:
        r = mixing_ratio(saturation_vapor_pressure(td0[:, newaxis]), p0[:, newaxis])
    else:
        raise NotImplementedError
    if height is not None:
        raise NotImplementedError

    rt_profile = _dewpoint(vapor_pressure(pressure, r))

    p, t = F.intersect_nz(pressure, rt_profile, temperature, "increasing", log_x=True).pick(which)

    return p, t, dry_lapse(p0, t, p)


# -------------------------------------------------------------------------------------------------
# el & lfc
# -------------------------------------------------------------------------------------------------
def _multiple_el_lfc_options(
    x: np.ndarray[shape[N, Z], np.dtype[float_]],
    y: np.ndarray[shape[N, Z], np.dtype[float_]],
    which: L["bottom", "top"] = "top",
) -> Vector1d[float_]:
    """
    it is assumed that the x and y arrays are sorted in ascending order
    >>> [[76852.646 nan nan ... ] [45336.262 88486.399 nan ... ]]
    """
    # idx: tuple[slice, slice]
    if which == "bottom":
        idx = np.s_[
            np.arange(x.shape[0]),
            np.argmin(~np.isnan(x), axis=1) - 1,  # the last non-nan value
        ]

    elif which == "top":
        idx = np.s_[
            np.arange(x.shape[0]),
            np.argmax(~np.isnan(x), axis=1),  # the first non-nan value
        ]

    return Vector1d(x[idx], y[idx])


def anyz(
    x: np.ndarray[shape[N, Z], np.dtype[float_]],
) -> np.ndarray[shape[N, L[1]], np.dtype[np.bool_]]:
    return np.any(x, axis=1, keepdims=True)


def _el_lfc(
    pick: L["EL", "LFC", "BOTH"],
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]] | None = None,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[float_]] | None = None,
) -> tuple[Vector1d[float_], Vector1d[float_]] | Vector1d[float_]:
    p0, t0 = pressure[:, 0], temperature[:, 0]
    if dewpoint_start is None:
        td0 = dewpoint[:, 0]  # (N,)
    else:
        td0 = dewpoint_start

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    LCL = Vector1d.from_func(lcl, p0, t0, td0).unsqueeze()
    pressure, parcel_temperature_profile, temperature = (
        pressure[:, 1:],
        parcel_temperature_profile[:, 1:],
        temperature[:, 1:],
    )

    # find the Equilibrium Level (EL)
    EL = F.intersect_nz(
        pressure,
        parcel_temperature_profile,
        temperature,
        "decreasing",
        log_x=True,
    )
    if pick == "EL":
        return EL.where(EL.is_above(LCL)).pick(which_el)

    # find the Level of Free Convection (LFC)
    positive_area_above_the_LCL = LCL.is_above(pressure, close=True)
    parcel_temperature_profile, temperature = (
        np.where(positive_area_above_the_LCL, parcel_temperature_profile, np.nan),
        np.where(positive_area_above_the_LCL, temperature, np.nan),
    )
    LFC = F.intersect_nz(
        pressure,
        parcel_temperature_profile,
        temperature,
        "increasing",
        log_x=True,
    )

    no_lfc = LFC.is_nan().all(axis=1, keepdims=True)

    lfc_is_the_lfc = ~no_lfc & LFC.is_above(LCL)
    lfc_is_the_lcl = no_lfc & _greater_or_close(parcel_temperature_profile, temperature).astype(
        np.bool_
    ).any(axis=1, keepdims=True)

    lfc_is_the_lcl |= ~no_lfc & LCL.is_above(LFC.bottom()).any(axis=1, keepdims=True)

    LFC = LFC.select(
        [lfc_is_the_lfc, lfc_is_the_lcl],
        [LFC.pressure, LCL.pressure],
        [LFC.temperature, LCL.temperature],
    )

    if pick == "LFC":
        return LFC.pick(which_lfc)

    return EL.pick(which_el), LFC.pick(which_lfc)


@broadcast_nz
def el(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]] | None = None,
    which: L["top", "bottom"] = "top",
) -> Vector1d[float_]:
    return _el_lfc(
        "EL", pressure, temperature, dewpoint, parcel_temperature_profile, which_el=which
    )


@broadcast_nz
def lfc(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: np.ndarray | None = None,
    which: L["top", "bottom"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[float_]] | None = None,
) -> Vector1d[float_]:
    return _el_lfc(
        "LFC",
        pressure,
        temperature,
        dewpoint,
        parcel_temperature_profile,
        which_lfc=which,
        dewpoint_start=dewpoint_start,
    )


@broadcast_nz
def el_lfc(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]] | None = None,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[float_]] | None = None,
) -> tuple[Vector1d[float_], Vector1d[float_]]:
    return _el_lfc(
        "BOTH",
        pressure,
        temperature,
        dewpoint,
        parcel_temperature_profile,
        which_lfc=which_lfc,
        which_el=which_el,
        dewpoint_start=dewpoint_start,
    )


# -------------------------------------------------------------------------------------------------
# cape_cin
# -------------------------------------------------------------------------------------------------
def most_unstable_parcel_index(
    pressure,
    temperature,
    dewpoint,
    /,
    depth: float = 30000.0,
    height: float | None = None,
    bottom: float | None = None,
):
    if height is not None:
        raise NotImplementedError("height argument is not implemented")

    pressure = np.atleast_2d(pressure)
    p0 = pressure[:, 0] if bottom is None else np.asarray(bottom)  # .reshape(-1, 1)
    top = p0 - depth
    (mask,) = np.nonzero(between_or_close(pressure, top, p0).astype(np.bool_).squeeze())
    t_layer = temperature[:, mask]
    td_layer = dewpoint[:, mask]
    p_layer = pressure[:, mask]
    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)
    return np.argmax(theta_e, axis=1)


@broadcast_nz
def cape_cin(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_profile: np.ndarray,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
) -> tuple[np.ndarray, np.ndarray]:
    lcl_p = lcl_pressure(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])  # ✔️

    # The mixing ratio of the parcel comes from the dewpoint below the LCL, is saturated
    # based on the temperature above the LCL
    parcel_mixing_ratio = np.where(
        pressure > lcl_p[:, newaxis],  # below_lcl
        saturation_mixing_ratio(pressure, dewpoint),
        saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    temperature = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))
    parcel_profile = virtual_temperature(parcel_profile, parcel_mixing_ratio)
    # Calculate the EL limit of integration
    (el_p, _), (lfc_p, _) = _el_lfc(
        "BOTH",
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc,
        which_el,
    )

    el_p[np.isnan(el_p)] = np.nanmin(pressure, axis=1)

    lfc_p, el_p = np.reshape((lfc_p, el_p), (2, -1, 1))  # reshape for broadcasting

    X, Y = F.zero_crossings(pressure, parcel_profile - temperature)  # ((N, Z), ...)

    mask = _between_or_close(X, el_p, lfc_p).astype(np.bool_)
    x, y = np.where(mask[newaxis, ...], [X, Y], np.nan)
    CAPE = Rd * F.nantrapz(y, np.log(x), axis=1)
    CAPE[(CAPE < 0.0)] = 0.0

    mask = _greater_or_close(X, lfc_p).astype(np.bool_)
    x, y = np.where(mask[newaxis, ...], [X, Y], np.nan)
    CIN = Rd * F.nantrapz(y, np.log(x), axis=1)
    CIN[(CIN > 0.0)] = 0.0

    return CAPE, CIN


@broadcast_nz
def most_unstable_parcel(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    depth: Pascal[float] = 30_000.0,
    bottom: Pascal[float] | None = None,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    np.ndarray[shape[N, Z], np.dtype[np.intp]],
]:
    depth = 100.0 if depth is None else depth
    p0 = (pressure[:, 0] if bottom is None else np.asarray(bottom)).reshape(-1, 1)
    top = p0 - depth

    n = np.arange(temperature.shape[0])
    mask = between_or_close(pressure, p0, top).astype(np.bool_)
    p = pressure[n, mask]
    t = temperature[n, mask]
    td = dewpoint[n, mask]

    # theta_e = equivalent_potential_temperature(p, t, td)
    # idx = np.argmax(theta_e, axis=1)

    # return p[n, idx], t[n, idx], td[n, idx], np.array([n, idx])
