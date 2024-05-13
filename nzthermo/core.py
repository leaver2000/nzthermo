# ruff: noqa: F405,F403
# pyright: reportReturnType=none
# pyright: reportAssignmentType=none
"""
This is an implementation of the metpy thermodynamics module but without the pint requirement and
additional support for higher dimensions in what would have normally been 1D arrays
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
    Final,
    Generic,
    Literal,
    NamedTuple,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from typing_extensions import Doc

from . import functional as F
from ._c import lcl, moist_lapse
from ._typing import Kelvin, Kilogram, N, Pascal, Ratio, Z, shape
from .const import E0, P0, T0, Cpd, Rd, Rv

# .....{ types }.....
T = TypeVar("T")
Pair = tuple[T, T]
float_ = TypeVar("float_", bound=np.float_)
newaxis: Final[None] = np.newaxis


# =================================================================================================
# .....{ basic thermodynamics }.....
# None of the following require any type of array broadcasting or fancy indexing
# =================================================================================================
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
# wet_bulb_temperature
# -------------------------------------------------------------------------------------------------
def wet_bulb_temperature(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
) -> Kelvin[np.ndarray[shape[N], np.dtype[float_]]]:
    # TODO:...
    shape = pressure.shape
    pressure, temperature, dewpoint = map(np.ndarray.ravel, (pressure, temperature, dewpoint))
    lcl_p, lcl_t = lcl(pressure, temperature, dewpoint)

    return moist_lapse(pressure, lcl_t, lcl_p).reshape(shape)


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
        ct = dry_lapse(p0, t, p)

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
    td = _dewpoint(  # (N, Z)
        vapor_pressure(pressure, mixing_ratio(saturation_vapor_pressure(td0[:, newaxis]), p0[:, newaxis]))
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


# -------------------------------------------------------------------------------------------------
# parcel_profile
# -------------------------------------------------------------------------------------------------
def _insert(arr: np.ndarray, mask: np.ndarray, value: np.ndarray):
    N, Z = arr.shape
    x = np.column_stack([arr, np.full(N, np.nan)])
    out = np.full_like(x, np.nan)
    # [ below mask ]
    nx, zx = np.nonzero(mask)
    out[nx, zx] = arr[nx, zx]
    # [ above mask ]
    nx, zx = np.nonzero(~mask)
    zx = np.minimum(zx, Z - 1)
    out[nx, zx + 1] = arr[nx, zx]
    # [ mask position ]
    out[np.isnan(out)] = value

    return out


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

    @property
    def lcl_pressure(self) -> np.ndarray[shape[N], np.dtype[float_]]:
        return self.pressure[self.lcl_index]

    @property
    def lcl_temperature(self) -> np.ndarray[shape[N], np.dtype[float_]]:
        return self.temperature[self.lcl_index]

    def below_lcl(self, copy: bool = True):
        P, T = self.pressure, self.temperature
        N, Z = self.shape
        if copy:
            P, T = P.copy(), T.copy()
        indices = np.ones(N, dtype=np.int_)[:, None] * np.arange(Z)
        mask = indices >= self.lcl_index[1][:, None]

        P[mask] = np.nan
        T[mask] = np.nan
        return P, T

    def above_lcl(self, copy: bool = True):
        P, T = self.pressure, self.temperature
        N, Z = self.shape
        if copy:
            P, T = P.copy(), T.copy()
        indices = np.ones(N, dtype=np.int_)[:, None] * np.arange(Z)
        mask = indices < self.lcl_index[1][:, None]

        P[mask] = np.nan
        T[mask] = np.nan
        return P, T

    @property
    def shape(self) -> shape[N, Z]:
        return self.pressure.shape  # type: ignore

    def with_lcl(
        self,
    ) -> tuple[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    ]:
        # mpcalc.parcel_profile_with_lcl
        return (
            self.pressure,
            self.temperature,
            self.dewpoint,
            self.temperature_profile,
        )

    def el(
        self, which: Literal["top", "bottom"] = "top"
    ) -> tuple[
        Pascal[np.ndarray[shape[N], np.dtype[float_]]],
        Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
    ]:
        pressure, temperature, dewpoint = self.pressure, self.temperature, self.dewpoint
        parcel_temperature_profile = self.temperature_profile
        shape = temperature.shape
        assert pressure.shape == shape == dewpoint.shape == parcel_temperature_profile.shape
        lcl_p = self.lcl_pressure

        intersect = F.intersect_nz(
            pressure[:, 1:],
            temperature[:, 1:],
            parcel_temperature_profile[:, 1:],
            log_x=True,
        )
        if which == "top":
            x, y, _ = intersect.top()
        elif which == "bottom":
            x, y, _ = intersect.bottom()
        else:
            raise ValueError("which must be 'top' or 'bottom'")
        mask = x < lcl_p
        return np.where(mask, x, np.nan), np.where(mask, y, np.nan)

    def __repr__(self) -> str:
        text = f"{self.__class__.__name__}(\n"
        text += f"[pressure]\n{self.pressure}\n"
        text += f"[temperature]\n{self.temperature}\n"
        text += f"[dewpoint]\n{self.dewpoint}\n"
        text += f"[temperature_profile]\n{self.temperature_profile}\n)"

        return text


def parcel_profile(
    pressure: Annotated[
        Pascal[np.ndarray[shape[Z], np.dtype[float_]]], "isobaric pressure levels"
    ],  # TODO: add support for (N, Z) pressure arrays...
    temperature: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric temperature"],
    dewpoint: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric dewpoint temperature"],
    *,
    refrence_pressure: Annotated[Pascal[np.ndarray[shape[N], np.dtype[np.float_]]], "surface pressure"] | None = None,
    refrence_temperature: Annotated[Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]], "surface temperature"]
    | None = None,
    refrence_dewpoint: Annotated[Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]], "surface dewpoint temperature"]
    | None = None,
) -> ParcelProfile[float_]:
    # add a nan value to the end of the pressure array
    dtype = pressure.dtype
    pressure = np.append(pressure, np.nan)
    N, Z = temperature.shape[0], pressure.shape[0]
    indices = np.arange(N)
    P0 = pressure[:1].repeat(N) if refrence_pressure is None else refrence_pressure  # (N,)
    T0 = temperature[:, 0] if refrence_temperature is None else refrence_temperature  # (N,)
    Td0 = dewpoint[:, 0] if refrence_dewpoint is None else refrence_dewpoint  # (N,)
    assert T0.shape == Td0.shape == P0.shape == (N,)

    # [ lcl ]
    lcl_p, lcl_t = lcl(P0, T0, Td0)  # (N,)

    # [ pressure ]
    mask = pressure >= lcl_p.reshape(-1, 1)  # (N, Z)
    mask[indices, np.argmin(mask, axis=1) + 1] = 0
    P = np.full((N, Z), np.nan, dtype=dtype)
    nx, zx = np.nonzero(mask)
    P[nx, zx] = pressure[zx]
    nx, zx = np.nonzero(~mask & ~np.isnan(pressure))
    P[nx, zx + 1] = pressure[zx]
    lcl_index = np.nonzero(np.isnan(P))
    assert len(lcl_index) == 2 and len(lcl_index[0]) == N == len(lcl_index[1])
    P[lcl_index] = lcl_p

    # [ parcel temperature ]
    lower = np.column_stack(
        [T0, dry_lapse(np.where(mask, P, np.nan), T0[:, newaxis], P0[:, newaxis])[:, 1:]]
    )  #  sfc -> lcl
    upper = moist_lapse(np.where(mask, np.nan, P), lcl_t, lcl_p)  # lcl -> top

    Tp = np.where(mask, lower, upper)  # parcel temperature
    Tp[lcl_index] = lcl_t

    # [ temperature & dewpoint ]
    T, Td = F.interpolate_nz(
        lcl_p,
        pressure[:-1],  # drop the trailing nan value
        temperature,
        dewpoint,
    )
    T = _insert(temperature, mask, T)
    Td = _insert(dewpoint, mask, Td)

    return ParcelProfile(P, T, Td, Tp, lcl_index)


_parcel_profile: Final = parcel_profile  # alias for the parcel_profile function to mitigate namespace conflicts


# -------------------------------------------------------------------------------------------------
# el
# -------------------------------------------------------------------------------------------------
def el(
    pressure: Annotated[
        Pascal[np.ndarray[shape[Z], np.dtype[float_]]], "isobaric pressure levels"
    ],  # TODO: add support for (N, Z) pressure arrays...
    temperature: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric temperature"],
    dewpoint: Annotated[Kelvin[np.ndarray[shape[N, Z], np.dtype[np.float_]]], "isobaric dewpoint temperature"],
    *,
    parcel_profile: Annotated[ParcelProfile[float_], "parcel profile"] | None = None,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[float_]]],
    Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
]:
    if parcel_profile is None:
        parcel_profile = _parcel_profile(pressure, temperature, dewpoint)
    return parcel_profile.el()


# TODO:...
# -------------------------------------------------------------------------------------------------
# lfc
# -------------------------------------------------------------------------------------------------
# TODO:...
# -------------------------------------------------------------------------------------------------
# cape
# -------------------------------------------------------------------------------------------------
# TODO:...
