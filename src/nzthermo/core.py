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
    between_or_close,
    dewpoint as _dewpoint,
    dry_lapse,
    equivalent_potential_temperature,
    greater_or_close,
    lcl,
    lcl_pressure,
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

float_ = TypeVar("float_", bound=np.floating[Any], covariant=True)
newaxis: Final[None] = np.newaxis
NaN = np.nan
z_axis: Final[tuple[slice, None]] = np.s_[:, newaxis]
N_AXIS: Final[tuple[None, slice]] = np.s_[newaxis, :]


FASTPATH: dict[str, Any] = {"__fastpath": True}


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
    nx = np.arange(N)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    mask = (pressure <= 70000.0) & (pressure >= 50000.0)

    if broadcasted := pressure.shape == temperature.shape:
        p_layer, t_layer, td_layer = np.where(
            mask[newaxis, :, :], [pressure, temperature, dewpoint], NaN
        )
    else:
        p_layer = np.where(mask, pressure, NaN)
        t_layer, td_layer = np.where(mask[newaxis, :, :], [temperature, dewpoint], NaN)

    theta_e = equivalent_potential_temperature(p_layer, t_layer, td_layer)

    zx = np.nanargmin(theta_e, axis=1)

    p_top = p_layer[nx, zx] if broadcasted else p_layer[0, zx]
    t_top = t_layer[nx, zx]  # (N,)
    td_top = td_layer[nx, zx]  # (N,)
    wb_top = wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

    # our moist_lapse rate function has nan ignoring capabilities
    pressure = np.where(pressure >= p_top[:, newaxis], pressure, NaN)
    e_vt = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))  # (N, Z)
    trace = core.moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    p_vt = virtual_temperature(trace, saturation_mixing_ratio(pressure, trace))  # (N, Z)

    DCAPE = Rd * F.nantrapz(p_vt - e_vt, np.log(pressure), axis=1)

    return DCAPE


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
def _el_lfc(
    pick: L["EL", "LFC", "BOTH"],
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]]
    | None = None,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[float_]] | None = None,
) -> tuple[Vector1d[float_], Vector1d[float_]] | Vector1d[float_]:
    if parcel_temperature_profile is None:
        pressure, temperature, dewpoint, parcel_temperature_profile = core.parcel_profile_with_lcl(
            pressure, temperature, dewpoint
        )

    N = temperature.shape[0]
    p0, t0 = pressure[:, 0], temperature[:, 0]
    if dewpoint_start is None:
        td0 = dewpoint[:, 0]  # (N,)
    else:
        td0 = dewpoint_start

    LCL = Vector1d.from_func(lcl, p0, t0, td0).unsqueeze()

    pressure, parcel_temperature_profile, temperature = (
        pressure[:, 1:],
        parcel_temperature_profile[:, 1:],
        temperature[:, 1:],
    )

    top_idx = np.arange(N), np.argmin(~np.isnan(pressure), axis=1) - 1
    # find the Equilibrium Level (EL)
    left_of_env = (parcel_temperature_profile[top_idx] <= temperature[top_idx])[:, newaxis]
    EL = F.intersect_nz(
        pressure,
        parcel_temperature_profile,
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
        parcel_temperature_profile,
        temperature,
        "increasing",
        log_x=True,
    ).where_above(LCL)

    is_lcl = (no_lfc := LFC.is_nan().all(axis=1, keepdims=True)) & greater_or_close(
        # the mask only needs to be applied to either the temperature or parcel_temperature_profile
        np.where(LCL.is_below(pressure, close=True), parcel_temperature_profile, NaN),
        temperature,
    ).any(axis=1, keepdims=True)

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
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]]
    | None = None,
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
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]]
    | None = None,
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

    mask = between_or_close(X, el_p, lfc_p).astype(np.bool_)
    x, y = np.where(mask[newaxis, ...], [X, Y], NaN)
    CAPE = Rd * F.nantrapz(y, np.log(x), axis=1)
    CAPE[CAPE < 0.0] = 0.0

    mask = greater_or_close(X, lfc_p).astype(np.bool_)
    x, y = np.where(mask[newaxis, ...], [X, Y], NaN)
    CIN = Rd * F.nantrapz(y, np.log(x), axis=1)
    CIN[CIN > 0.0] = 0.0

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
