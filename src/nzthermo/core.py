# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
This is an implementation of the metpy thermodynamics module but without the pint requirement and
additional support for higher dimensions in what would have normally been 1D arrays
"""

from __future__ import annotations

import operator
import warnings
from typing import (
    Final,
    Generic,
    Literal as L,
    NamedTuple,
    TypeVar,
)
from typing import Callable, Concatenate, ParamSpec
import functools
import numpy as np
from numpy.typing import NDArray

from . import _core as core, _ufunc as uf, functional as F
from .const import Rd, Rv
from .typing import Kelvin, N, Pascal, Z, shape, Ratio, NZArray

# .....{ types }.....
_T = TypeVar("_T")
_P = ParamSpec("_P")
Pair = tuple[_T, _T]
float_ = TypeVar("float_", bound=np.float_)
newaxis: Final[None] = np.newaxis


class ElementNd(NamedTuple, Generic[_T, float_]):
    x: Pascal[np.ndarray[_T, np.dtype[float_]]]
    y: Kelvin[np.ndarray[_T, np.dtype[float_]]]


class Element1d(ElementNd[shape[N], float_]): ...


class Element2d(ElementNd[shape[N, Z], float_]):
    def pick(self, which: L["bottom", "top"] = "top") -> ElementNd[shape[N], float_]:
        x, y = self.x, self.y
        nx = np.arange(x.shape[0])
        if which == "bottom":
            idx = np.argmin(~np.isnan(x), axis=1) - 1  # the last non-nan value
            return ElementNd(x[nx, idx], y[nx, idx])

        elif which == "top":
            return ElementNd(x[nx, 0], y[nx, 0])  # the first value is the uppermost value

    def bottom(self) -> ElementNd[shape[N], float_]:
        return self.pick("bottom")

    def top(self) -> ElementNd[shape[N], float_]:
        return self.pick("top")


def broadcast_nz(
    f: Callable[Concatenate[NZArray[np.float_], NZArray[np.float_], NZArray[np.float_], _P], _T],
) -> Callable[
    Concatenate[Pascal[NDArray[float_]], Kelvin[NDArray[float_]], Kelvin[NDArray[float_]], _P], _T
]:
    @functools.wraps(f)
    def wrapper(
        pressure,
        temperature,
        dewpoint,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        # TODO
        # - add support for squeezing what would have been a 1d input
        # - add support for reshaping:
        #   (T, Z, Y, X) -> (N, Z)
        #   (Z, Y, X) -> (N, Z)'
        pressure, temperature, dewpoint = F.exactly_2d(pressure, temperature, dewpoint)
        return f(pressure, temperature, dewpoint, *args, **kwargs)

    return wrapper


# =================================================================================================
# .....{ basic thermodynamics }.....
# None of the following require any type of array broadcasting or fancy indexing
# TODO: because the _multiple_el_lfc_options function is highly dependent on the output of
# find_intersections combining them into a class is probably the best idea.
# =================================================================================================
def mixing_ratio_from_specific_humidity(
    specific_humidity: Ratio[NDArray[float_]],
) -> Ratio[NDArray[float_]]:
    specific_humidity = np.where(specific_humidity == 0, 1e-10, specific_humidity)
    return specific_humidity / (1 - specific_humidity)


def mixing_ratio(
    partial_press: Pascal[NDArray[float_]],
    total_press: Pascal[NDArray[float_] | float],
    molecular_weight_ratio: Ratio[NDArray[float_] | float] = Rd / Rv,
) -> Ratio[NDArray[float_]]:
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


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
        r = mixing_ratio(uf.saturation_vapor_pressure(td0[:, newaxis]), p0[:, newaxis])
    else:
        raise NotImplementedError

    rt_profile = uf.dewpoint(uf.vapor_pressure(pressure, r))

    p, t = F.find_intersections(pressure, rt_profile, temperature, "increasing", log_x=True).pick(
        which
    )

    return p, t, uf.dry_lapse(p0, t, p)


# -------------------------------------------------------------------------------------------------
# el & lfc
# -------------------------------------------------------------------------------------------------
def _multiple_el_lfc_options(
    x: np.ndarray[shape[N, Z], np.dtype[float_]],
    y: np.ndarray[shape[N, Z], np.dtype[float_]],
    which: L["bottom", "top"] = "top",
) -> ElementNd[shape[N], float_]:
    """
    it is assumed that the x and y arrays are sorted in ascending order
    >>> [[76852.646 nan nan ... ] [45336.262 88486.399 nan ... ]]
    """
    if which == "bottom":
        idx = np.s_[
            np.arange(x.shape[0]),
            np.argmin(~np.isnan(x), axis=1) - 1,  # the last non-nan value
        ]

    elif which == "top":
        idx = np.s_[:, 0]

    return ElementNd(x[idx], y[idx])


@broadcast_nz
def el(
    pressure: Pascal[NZArray[float_]],
    temperature: Kelvin[NZArray[float_]],
    dewpoint: Kelvin[NZArray[float_]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]] | None = None,
    which: L["top", "bottom"] = "top",
    lcl_p: np.ndarray | None = None,
) -> ElementNd[shape[N], float_]:
    p0, t0, td0 = pressure[:, 0], temperature[:, 0], dewpoint[:, 0]

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    if lcl_p is None:
        lcl_p = uf.lcl_pressure(p0, t0, td0)

    el_p, el_t = F.find_intersections(
        pressure[:, 1:],
        parcel_temperature_profile[:, 1:],
        temperature[:, 1:],
        "decreasing",
        log_x=True,
    )

    idx = el_p < lcl_p[:, newaxis]  # ABOVE: the LCL
    el_p = np.where(idx, el_p, np.nan)  # BELOW: is LCL is set to nan
    el_t = np.where(idx, el_t, np.nan)

    return _multiple_el_lfc_options(el_p, el_t, which)


@broadcast_nz
def lfc(
    pressure: Pascal[NZArray[float_]],
    temperature: Kelvin[NZArray[float_]],
    dewpoint: Kelvin[NZArray[float_]],
    /,
    parcel_temperature_profile: np.ndarray | None = None,
    which: L["top", "bottom"] = "top",
    dewpoint_start: np.ndarray[shape[N], np.dtype[float_]] | None = None,
) -> ElementNd[shape[N], float_]:
    p0, t0 = pressure[:, 0], temperature[:, 0]
    if dewpoint_start is None:
        td0 = dewpoint[:, 0]  # (N,)
    else:
        td0 = dewpoint_start

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    lcl_p, lcl_t = (x[:, newaxis] for x in uf.lcl(p0, t0, td0))
    pressure, parcel_temperature_profile, temperature = (
        pressure[:, 1:],
        parcel_temperature_profile[:, 1:],
        temperature[:, 1:],
    )

    # .............................................................................................
    # The following was modified from the metpy implementation to handel 2 arrays
    # this is a bit tricky
    # .............................................................................................
    el_p = F.find_intersections(
        pressure,
        parcel_temperature_profile,
        temperature,
        "decreasing",
        log_x=True,
    )[0]

    lfc_p, lfc_t = F.find_intersections(
        pressure,
        parcel_temperature_profile,
        temperature,
        "increasing",
        log_x=True,
    )
    potential = lfc_p < lcl_p  # ABOVE: the LCL
    no_potential = ~potential
    is_lcl = np.all(np.isnan(lfc_p), axis=1, keepdims=True)
    positive_area_above_the_LCL = pressure < lcl_p

    # LFC does not exist or is LCL if len(x) == 0:
    no_lfc = is_lcl & F.logical_or_close(
        operator.lt,
        np.where(positive_area_above_the_LCL, parcel_temperature_profile, np.nan),
        np.where(positive_area_above_the_LCL, temperature, np.nan),
    ).any(axis=1, keepdims=True)

    # LFC exists. Make sure it is no lower than the LCL else:
    with warnings.catch_warnings():  # RuntimeWarning: All-NaN slice encountered
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        no_lfc |= np.nanmin(el_p, axis=1, keepdims=True) > lcl_p
        is_lcl |= np.all(no_potential, axis=1, keepdims=True)

    condlist = [no_lfc, is_lcl, potential]
    x_choice = [np.nan, lcl_p, lfc_p]
    y_choice = [np.nan, lcl_t, lfc_t]

    lfc_p = np.select(condlist, x_choice, np.nan)
    lfc_t = np.select(condlist, y_choice, np.nan)
    LFC = _multiple_el_lfc_options(lfc_p, lfc_t, which)

    return LFC


@broadcast_nz
def el_lfc(
    pressure: Pascal[NZArray[float_]],
    temperature: Kelvin[NZArray[float_]],
    dewpoint: Kelvin[NZArray[float_]],
    /,
    parcel_temperature_profile: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]] | None = None,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
):
    p0, t0, td0 = pressure[:, 0], temperature[:, 0], dewpoint[:, 0]

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    lcl_p, lcl_t = (x[:, newaxis] for x in uf.lcl(p0, t0, td0))
    pressure, parcel_temperature_profile, temperature = (
        pressure[:, 1:],
        parcel_temperature_profile[:, 1:],
        temperature[:, 1:],
    )

    # find the Equilibrium Level (EL)
    el_p, el_t = F.find_intersections(
        pressure,
        parcel_temperature_profile,
        temperature,
        "decreasing",
        log_x=True,
    )
    EL = _multiple_el_lfc_options(el_p, el_t, which_el)

    # find the Level of Free Convection (LFC)
    lfc_p, lfc_t = F.find_intersections(
        pressure,
        parcel_temperature_profile,
        temperature,
        "increasing",
        log_x=True,
    )
    # START: conditional logic to determine the LFC
    potential = lfc_p < lcl_p  # ABOVE: the LCL
    no_potential = ~potential
    is_lcl = np.all(np.isnan(lfc_p), axis=1, keepdims=True)
    positive_area_above_the_LCL = pressure < lcl_p

    # LFC does not exist or is LCL if len(x) == 0:
    no_lfc = is_lcl & F.logical_or_close(
        operator.lt,
        np.where(positive_area_above_the_LCL, parcel_temperature_profile, np.nan),
        np.where(positive_area_above_the_LCL, temperature, np.nan),
    ).any(axis=1, keepdims=True)

    # LFC exists. Make sure it is no lower than the LCL else:
    with warnings.catch_warnings():  # RuntimeWarning: All-NaN slice encountered
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        no_lfc |= np.nanmin(el_p, axis=1, keepdims=True) > lcl_p
        is_lcl |= np.all(no_potential, axis=1, keepdims=True)
    # END: conditional logic to determine the LFC

    condlist = [no_lfc, is_lcl, potential]
    x_choice = [np.nan, lcl_p, lfc_p]
    y_choice = [np.nan, lcl_t, lfc_t]

    lfc_p = np.select(condlist, x_choice, np.nan)
    lfc_t = np.select(condlist, y_choice, np.nan)

    LFC = _multiple_el_lfc_options(lfc_p, lfc_t, which_lfc)

    return EL, LFC


# -------------------------------------------------------------------------------------------------
# downdraft_cape
# -------------------------------------------------------------------------------------------------
@broadcast_nz
def downdraft_cape(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
) -> np.ndarray[shape[N], np.dtype[float_]]:
    N, _ = temperature.shape

    mid_layer_idx = ((pressure <= 7e4) & (pressure >= 5e4)).squeeze()
    p_layer, t_layer, td_layer = (x[:, mid_layer_idx] for x in (pressure, temperature, dewpoint))

    theta_e = uf.equivalent_potential_temperature(p_layer, t_layer, td_layer)
    nx, zx = np.arange(N), np.argmin(theta_e, axis=1)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    p_top = p_layer[0, zx]  # (N,)
    t_top = t_layer[nx, zx]  # (N,)
    td_top = td_layer[nx, zx]  # (N,)
    wb_top = uf.wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

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

    trace = core.moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    e_vt = uf.virtual_temperature(
        temperature, uf.saturation_mixing_ratio(pressure, dewpoint)
    )  # (N, Z)
    p_vt = uf.virtual_temperature(trace, uf.saturation_mixing_ratio(pressure, trace))  # (N, Z)

    delta = e_vt - p_vt
    logp = np.log(pressure)
    dcape = -(Rd * F.nantrapz(delta, logp, axis=1))

    return dcape


# -------------------------------------------------------------------------------------------------
# cape_cin
# -------------------------------------------------------------------------------------------------
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
    lcl_p = uf.lcl_pressure(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])  # ✔️

    # The mixing ratio of the parcel comes from the dewpoint below the LCL, is saturated
    # based on the temperature above the LCL
    parcel_mixing_ratio = np.where(
        pressure > lcl_p[:, newaxis],  # below_lcl
        uf.saturation_mixing_ratio(pressure, dewpoint),
        uf.saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    temperature = uf.virtual_temperature(
        temperature, uf.saturation_mixing_ratio(pressure, dewpoint)
    )
    parcel_profile = uf.virtual_temperature(parcel_profile, parcel_mixing_ratio)
    # Calculate the EL limit of integration
    (el_p, _), (lfc_p, _) = el_lfc(
        pressure, temperature, dewpoint, parcel_profile, which_lfc, which_el
    )
    # el_p = el(pressure, temperature, dewpoint, parcel_profile, which=which_el).x  # ✔️
    # # Calculate LFC limit of integration
    # lfc_p = lfc(pressure, temperature, dewpoint, parcel_profile, which=which_lfc).x  # ✔️

    pressure = np.broadcast_to(pressure, temperature.shape)
    p_top = np.nanmin(pressure, axis=1)
    el_p = np.where(np.isnan(el_p), p_top, el_p)

    lfc_p, el_p = np.reshape((lfc_p, el_p), (2, -1, 1))

    X, Y = F.find_append_zero_crossings(pressure, parcel_profile - temperature)  # ((N, Z), ...)

    mask = F.logical_or_close(operator.lt, X, lfc_p) & F.logical_or_close(operator.gt, X, el_p)
    x, y = np.where(mask[newaxis, ...], [X, Y], np.nan)

    cape = Rd * F.nantrapz(y, np.log(x), axis=1)
    cape[(cape < 0.0)] = 0.0

    mask = F.logical_or_close(operator.gt, X, lfc_p)
    x, y = np.where(mask[newaxis, ...], [X, Y], np.nan)

    cin = Rd * F.nantrapz(y, np.log(x), axis=1)
    cin[(cin > 0.0)] = 0.0

    return cape, cin


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
    # pressure = F.exactly_2d(pressure)
    depth = 100.0 if depth is None else depth
    p0 = pressure[:, 0] if bottom is None else bottom
    top = p0 - depth

    mask = F.logical_or_close(operator.lt, pressure, p0) & F.logical_or_close(
        operator.gt, pressure, top
    )
    p = pressure[:, mask]
    t = temperature[:, mask]
    td = dewpoint[:, mask]

    theta_e = uf.equivalent_potential_temperature(p, t, td)
    idx = np.argmax(theta_e, axis=1)
    n = np.arange(t.shape[0])

    return p[n, idx], t[n, idx], td[n, idx], np.array([n, idx])
