# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
This is an implementation of the metpy thermodynamics module but without the pint requirement and
additional support for higher dimensions in what would have normally been 1D arrays
"""

from __future__ import annotations
import operator
from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Generic,
    Literal,
    Literal as L,
    NamedTuple,
    TypeVar,
    overload,
    Any,
)
import warnings
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import Doc

from . import functional as F, _core as core, _ufunc as uf
from .const import P0, Cpd, Rd, Rv
from .typing import Kelvin, Kilogram, N, Pascal, Ratio, Z, shape

# .....{ types }.....
T = TypeVar("T")
Pair = tuple[T, T]
float_ = TypeVar("float_", bound=np.float_)
newaxis: Final[None] = np.newaxis


def _2d(x: np.ndarray[Any, np.dtype[float_]]) -> np.ndarray[shape[N, Z], np.dtype[float_]]:
    if x.ndim == 1:
        x = x[newaxis, :]
    elif x.ndim != 2:
        raise ValueError("pressure must be a 1D or 2D array")

    return x


# =================================================================================================
# .....{ basic thermodynamics }.....
# None of the following require any type of array broadcasting or fancy indexing
# =================================================================================================
def mixing_ratio_from_specific_humidity(specific_humidity: Kilogram[NDArray[float_]]) -> Kilogram[NDArray[float_]]:
    specific_humidity = np.where(specific_humidity == 0, 1e-10, specific_humidity)
    return specific_humidity / (1 - specific_humidity)


def vapor_pressure(
    pressure: Pascal[NDArray[float_]], mixing_ratio: Ratio[NDArray[float_] | float]
) -> Pascal[NDArray[float_]]:
    return pressure * mixing_ratio / ((Rd / Rv) + mixing_ratio)


def exner_function(
    pressure: Pascal[NDArray[float_]], refrence_pressure: Pascal[NDArray[float_] | float] = P0
) -> Pascal[NDArray[float_]]:
    r"""\Pi = \left( \frac{p}{p_0} \right)^{R_d/c_p} = \frac{T}{\theta}"""
    return (pressure / refrence_pressure) ** (Rd / Cpd)


def mixing_ratio(
    partial_press: Pascal[NDArray[float_]],
    total_press: Pascal[NDArray[float_] | float],
    molecular_weight_ratio: Ratio[NDArray[float_] | float] = Rd / Rv,
) -> Ratio[NDArray[float_]]:
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


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
    return uf.dewpoint(pressure * w / (eps + w))


def most_unstable_parcel(
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    *,
    depth: Pascal[float] = 30_000.0,
    bottom: Pascal[float] | None = None,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    np.ndarray[shape[N, Z], np.dtype[np.intp]],
]:
    depth = 100.0 if depth is None else depth
    p0 = pressure[0] if bottom is None else bottom
    top = p0 - depth

    mask = np.logical_or(pressure < p0, np.isclose(pressure, p0)) & np.logical_or(
        pressure > top, np.isclose(pressure, top)
    )
    p = pressure[mask]
    t = temperature[:, mask]
    td = dewpoint[:, mask]

    theta_e = uf.equivalent_potential_temperature(p, t, td)
    idx = np.argmax(theta_e, axis=1)
    n = np.arange(t.shape[0])

    return p[idx], t[n, idx], td[n, idx], np.array([n, idx])


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
        which: Literal["bottom", "top"] = "bottom",
    ) -> ConvectiveCondensationLevel[float_]:
        p, t, _ = intersect.bottom() if which == "bottom" else intersect.top()
        ct = uf.dry_lapse(p0, t, p)

        return ConvectiveCondensationLevel(p, t, ct)


@overload
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["all"] = ...,
) -> Pair[ConvectiveCondensationLevel[float_]]: ...
@overload
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["bottom", "top"] = ...,
) -> ConvectiveCondensationLevel[float_]: ...
def ccl(
    pressure: Pascal[NDArray[float_]],
    temperature: Kelvin[NDArray[float_]],
    dewpoint: Kelvin[NDArray[float_]],
    *,
    which: Literal["bottom", "top", "all"] = "bottom",
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
    td = uf.dewpoint(  # (N, Z)
        vapor_pressure(pressure, mixing_ratio(uf.saturation_vapor_pressure(td0[:, newaxis]), p0[:, newaxis]))
    )

    intersect = F.intersect_nz(pressure, td, temperature, log_x=True)  # (N, Z)

    if which != "all":
        return ConvectiveCondensationLevel.from_intersect(p0, intersect, which)

    return (
        ConvectiveCondensationLevel.from_intersect(p0, intersect, "bottom"),
        ConvectiveCondensationLevel.from_intersect(p0, intersect, "top"),
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

    theta_e = uf.equivalent_potential_temperature(p_layer, t_layer, td_layer)
    nx, zx = np.arange(N), np.argmin(theta_e, axis=1)
    # Tims suggestion was to allow for the parcel to potentially be conditionally based
    p_top = p_layer[0, zx]  # (N,)
    t_top = t_layer[nx, zx]  # (N,)
    td_top = td_layer[nx, zx]  # (N,)
    wb_top = uf.wet_bulb_temperature(p_top, t_top, td_top)  # (N,)

    # reshape our pressure into a 2d pressure grid and put the hard cap on everything above the hard cap
    cap = -(np.searchsorted(np.squeeze(pressure)[::-1], np.min(p_top)) - 1)
    pressure = pressure[0, :cap].repeat(N).reshape(-1, N).transpose()  # (N, Z -cap)
    if not np.any(pressure):
        return np.repeat(np.nan, N).astype(dtype)

    # our moist_lapse rate function has nan ignoring capabilities
    pressure[pressure < p_top[:, newaxis]] = np.nan
    temperature = temperature[:, :cap]
    dewpoint = dewpoint[:, :cap]

    trace = core.moist_lapse(pressure, wb_top, p_top)  # (N, Z)
    e_vt = uf.virtual_temperature(temperature, uf.saturation_mixing_ratio(pressure, dewpoint))  # (N, Z)
    p_vt = uf.virtual_temperature(trace, uf.saturation_mixing_ratio(pressure, trace))  # (N, Z)

    delta = e_vt - p_vt
    logp = np.log(pressure)
    dcape = -(Rd * F.nantrapz(delta, logp, axis=1))

    return dcape


# -------------------------------------------------------------------------------------------------
# parcel_profile
# -------------------------------------------------------------------------------------------------
class ParcelProfile(NamedTuple, Generic[float_]):
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Doc("[N, [...below, LCL, ...above]]"),
    ]
    temperature: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Doc("[N, [...below, LCL, ...above]]"),
    ]
    dewpoint: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Doc("[N, [...below, LCL, ...above]]"),
    ]
    temperature_profile: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Doc("[N, [...dry_lapse, LCL, ...moist_lapse]]"),
    ]
    lcl_index: tuple[
        np.ndarray[shape[N], np.dtype[np.intp]],
        np.ndarray[shape[Z], np.dtype[np.intp]],
    ]


# -------------------------------------------------------------------------------------------------
# el & lfc helpers
# -------------------------------------------------------------------------------------------------
# TODO: because the _multiple_el_lfc_options function is highly dependent on the output of
# find_intersections combining them into a class is probably the best idea.
def find_intersections(
    x: np.ndarray[shape[N, Z], np.dtype[float_]],
    a: np.ndarray[shape[N, Z], np.dtype[float_]],
    b: np.ndarray[shape[N, Z], np.dtype[float_]],
    direction: Literal["increasing", "decreasing"] = "increasing",
    log_x: bool = False,
):
    if log_x is True:
        x = np.log(x)

    x = np.broadcast_to(x.squeeze(), a.shape)

    ind, nearest_idx = np.nonzero(np.diff(np.sign(a - b), axis=1))
    next_idx = nearest_idx + 1
    sign_change = np.sign(a[ind, next_idx] - b[ind, next_idx])
    x0, x1 = x[ind, nearest_idx], x[ind, next_idx]
    a0, a1 = a[ind, nearest_idx], a[ind, next_idx]
    b0, b1 = b[ind, nearest_idx], b[ind, next_idx]
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1

    with np.errstate(divide="ignore", invalid="ignore"):
        x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)  # type: ignore
        y = ((x - x0) / (x1 - x0)) * (a1 - a0) + a0  # type: NDArray[float_] # type: ignore
        if log_x:
            x = np.exp(x)

    if direction == "increasing":
        x[sign_change <= 0] = np.nan  # increasing
    else:
        x[sign_change >= 0] = np.nan

    x_full = np.full_like(a, fill_value=np.nan)
    y_full = np.full_like(a, fill_value=np.nan)

    x_full[ind, nearest_idx] = x[...]
    y_full[ind, nearest_idx] = y[...]

    sort = np.argsort(x_full, axis=1)
    x = x_full[np.arange(a.shape[0])[:, newaxis], sort]  # [:, :-1]
    y = y_full[np.arange(a.shape[0])[:, newaxis], sort]  # [:, :-1]

    return x, y


def _multiple_el_lfc_options(
    x: np.ndarray[shape[N, Z], np.dtype[float_]],
    y: np.ndarray[shape[N, Z], np.dtype[float_]],
    which: Literal["bottom", "top"] = "top",
):
    nx = np.arange(x.shape[0])  # [:, newaxis]
    if which == "bottom":
        idx = np.argmin(~np.isnan(x), axis=1) - 1  # the last non-nan value
        return x[nx, idx], y[nx, idx]

    elif which == "top":
        return x[nx, 0], y[nx, 0]  # the first non-nan value

    raise ValueError("which must be either 'top' or 'bottom'")


# -------------------------------------------------------------------------------------------------
# el
# -------------------------------------------------------------------------------------------------
def el(
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]] | Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
        "isobaric pressure levels",
    ],  # TODO: add support for (N, Z) pressure arrays...
    temperature: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric temperature"],
    dewpoint: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric dewpoint temperature"],
    /,
    parcel_temperature_profile: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], ""] | None = None,
    which: Literal["top", "bottom"] = "top",
    lcl_p: np.ndarray | None = None,
):
    pressure = _2d(pressure)

    p0 = pressure[:, 0]
    t0 = temperature[:, 0]  # (N,)
    td0 = dewpoint[:, 0]  # (N,)

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    if lcl_p is None:
        lcl_p = uf.lcl_pressure(p0, t0, td0)

    x, y = find_intersections(
        pressure[:, 1:], parcel_temperature_profile[:, 1:], temperature[:, 1:], "decreasing", log_x=True
    )

    idx = x <= lcl_p[:, newaxis]  # ABOVE: the LCL
    x = np.where(idx, x, np.nan)  # BELOW: is LCL is set to nan
    y = np.where(idx, y, np.nan)

    # I'm not sure if this is entirely necessary but it's a good idea to sort the values
    # this will put all of the nan values towards the tail end of the array
    return _multiple_el_lfc_options(x, y, which)


# -------------------------------------------------------------------------------------------------
# lfc
# -------------------------------------------------------------------------------------------------
def lfc(
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]] | Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
        # Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
        "isobaric pressure levels",
    ],  # TODO: add support for (N, Z) pressure arrays...
    temperature: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric temperature"],
    dewpoint: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric dewpoint temperature"],
    /,
    parcel_temperature_profile: np.ndarray | None = None,
    which: Literal["top", "bottom"] = "top",
    lcl_p: np.ndarray | None = None,
    dewpoint_start: np.ndarray | None = None,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
]:
    pressure = _2d(pressure)

    p0 = pressure[:, 0]
    t0 = temperature[:, 0]  # (N,)
    if dewpoint_start is None:
        td0 = dewpoint[:, 0]  # (N,)
    else:
        td0 = dewpoint_start

    if parcel_temperature_profile is None:
        parcel_temperature_profile = core.parcel_profile(pressure, t0, td0)

    if lcl_p is None:
        lcl_p = uf.lcl_pressure(p0, t0, td0)

    with warnings.catch_warnings():  # RuntimeWarning: All-NaN slice encountered
        warnings.simplefilter("ignore", category=RuntimeWarning)
        el_p = np.nanmin(
            find_intersections(
                pressure[:, 1:], parcel_temperature_profile[:, 1:], temperature[:, 1:], "increasing", log_x=True
            )[0],
            axis=1,
        )

    x, y = find_intersections(
        pressure[:, 1:], parcel_temperature_profile[:, 1:], temperature[:, 1:], "increasing", log_x=True
    )
    # this get's a little confusing because a lower value represents a higher point in the atmosphere
    # so when I refer to ABOVE or BELOW I am not referring to the literal value of the pressure but the
    # altitude in the atmosphere.
    idx = x < lcl_p[:, newaxis]  # ABOVE: the LCL
    no_intersect = np.all(~idx, axis=1)  # LFC does not exist or is LCL

    x = np.where(idx, x, np.nan)  # BELOW: is LCL is set to nan
    x[no_intersect, 0] = el_p[no_intersect]  # where there is no intersection set the LFC to the EL
    idx[no_intersect, 0] = True  # and we need to update our index to reflect this

    y = np.where(idx, y, np.nan)

    return _multiple_el_lfc_options(x, y, which)


# TODO:...
# -------------------------------------------------------------------------------------------------
# cape_cin
# -------------------------------------------------------------------------------------------------
def cape_cin(
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]] | Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
        "isobaric pressure levels",
    ],  # TODO: add support for (N, Z) pressure arrays...
    temperature: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric temperature"],
    dewpoint: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric dewpoint temperature"],
    parcel_profile: np.ndarray,
    which_lfc: L["bottom", "top"] = "bottom",
    which_el: L["bottom", "top"] = "top",
) -> tuple[np.ndarray, np.ndarray]:
    pressure = _2d(pressure)
    lcl_p = uf.lcl_pressure(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])  # ✔️

    # The mixing ratio of the parcel comes from the dewpoint below the LCL, is saturated
    # based on the temperature above the LCL
    parcel_mixing_ratio = np.where(
        pressure > lcl_p[:, newaxis],  # below_lcl
        uf.saturation_mixing_ratio(pressure, dewpoint),
        uf.saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    temperature = uf.virtual_temperature(temperature, uf.saturation_mixing_ratio(pressure, dewpoint))
    parcel_profile = uf.virtual_temperature(parcel_profile, parcel_mixing_ratio)
    # Calculate LFC limit of integration
    lfc_p, _ = lfc(pressure, temperature, dewpoint, parcel_profile, which=which_lfc)  # ✔️

    # Calculate the EL limit of integration
    el_p, _ = el(pressure, temperature, dewpoint, parcel_profile, which=which_el)  # ✔️

    X, Y = F.find_append_zero_crossings(
        np.broadcast_to(pressure, temperature.shape), parcel_profile - temperature
    )  # ((N, Z), ...)

    # - need to spell this out i guess
    lfc_p = lfc_p.reshape(-1, 1)
    el_p = el_p.reshape(-1, 1)

    mask = F.logical_or_close(operator.lt, X, lfc_p) & F.logical_or_close(operator.gt, X, el_p)
    x, y = np.where(mask[None], [X, Y], np.nan)

    cape = Rd * F.nantrapz(y, np.log(x), axis=1)
    cape[cape < 0.0] = 0.0

    mask = F.logical_or_close(operator.gt, X, lfc_p)
    x, y = np.where(mask[None], [X, Y], np.nan)

    cin = Rd * F.nantrapz(y, np.log(x), axis=1)
    cin[cin > 0.0] = 0.0

    return cape, cin
