from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, Literal as L, TypeVar

import numpy as np

from . import functional as F
from ._core import Rd, g, parcel_profile
from ._ufunc import (
    between_or_close,
    dewpoint_from_specific_humidity,
    less_or_close,
    moist_static_energy,
    pressure_vector,
    saturation_mixing_ratio,
    wind_magnitude,
)
from .core import (
    el_lfc,
    mixed_layer_cape_cin,
    mixed_parcel,
    most_unstable_cape_cin,
    most_unstable_parcel,
    specific_humidity as _specific_humidity,
    surface_based_cape_cin,
)
from .typing import Dimensionless, Kelvin, Meter, MeterPerSecond, N, Pascal, Z, shape
from .utils import Parcel

_T = TypeVar("_T", bound=np.floating[Any])


def bunkers_storm_motion(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    u: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    v: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    height: Meter[np.ndarray[shape[N, Z], np.dtype[_T]]],
) -> tuple[np.ndarray[shape[L["(U, V)"], N], np.dtype[_T]], ...]:
    # mean wind from sfc-6km
    mask = less_or_close(height, 6000.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    wind_mean = F.nantrapz([u, v], pressure, axis=2, where=mask[np.newaxis]) / delta

    # mean wind from sfc-500m
    mask = less_or_close(height, 500.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    w0500 = F.nantrapz([u, v], x=pressure, axis=2, where=mask[np.newaxis]) / delta

    mask = between_or_close(height, 5250.0, 6250.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    w5500 = F.nantrapz([u, v], x=pressure, axis=2, where=mask[np.newaxis]) / delta

    shear = w5500 - w0500
    shear_cross = shear[::-1]
    shear_cross[1] *= -1
    mag = np.hypot(shear[0], shear[1])

    rdev = shear_cross * (7.5 / mag)
    # Add the deviations to the layer average wind to get the RM motion
    right_mover = wind_mean + rdev

    # Subtract the deviations to get the LM motion
    left_mover = wind_mean - rdev

    return right_mover, left_mover, wind_mean  # type: ignore


def _get_profile(
    fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, ...]],
    pressure,
    temperature,
    dewpoint,
):
    _, t, td, *_ = fn(pressure, temperature, dewpoint)
    return parcel_profile(pressure, t, td)


class entrainment(Generic[_T]):
    # TODO: some of the ecape type arguments need to be built into the constructor function
    # as many of the properties are dependent on the parcel type...
    if TYPE_CHECKING:
        shape: tuple[int, int]
        dtype: np.dtype[_T]
        pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]]
        height: Meter[np.ndarray[shape[N, Z], np.dtype[_T]]]
        temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]]
        u_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]]
        v_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]]
        specific_humidity: Dimensionless[np.ndarray[shape[N, Z], np.dtype[_T]]]
        dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]]
        wind: np.ndarray[shape[L["(U, V)"], N, Z], np.dtype[_T]]
        el: Parcel[_T]
        lfc: Parcel[_T]

    def __init__(
        self,
        pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[_T]]],
        height: Meter[np.ndarray[shape[N, Z], np.dtype[_T]]],
        temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
        u_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
        v_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
        /,
        specific_humidity: Dimensionless[np.ndarray[shape[N, Z], np.dtype[_T]]] | None = None,
        dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]] | None = None,
        dtype: type[_T] | np.dtype[_T] | None = None,
        cape_type: np.ndarray[shape[N], np.dtype[np.floating[Any]]]
        | L["most_unstable", "surface_based", "mixed_layer"] = "most_unstable",
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = temperature.dtype
        N, Z = temperature.shape
        self.shape = N, Z

        self.dtype = np.dtype(dtype)
        self.temperature = temperature.astype(dtype, copy=False)
        self.pressure = pressure.astype(dtype, copy=False).view(pressure_vector)
        self.height = height.astype(dtype, copy=False)

        if dewpoint is not None and specific_humidity is not None:
            raise ValueError("Only one of specific_humidity or dewpoint can be provided")
        elif specific_humidity is not None:
            self.specific_humidity = specific_humidity
            self.dewpoint = dewpoint_from_specific_humidity(pressure, specific_humidity)
        elif dewpoint is not None:
            self.dewpoint = dewpoint
            self.specific_humidity = _specific_humidity(pressure, dewpoint)  # type: ignore
        else:
            raise ValueError("Either specific_humidity or dewpoint must be provided")

        self.wind = np.array([u_wind, v_wind], dtype=dtype)

        if cape_type == "most_unstable":
            self._cape = most_unstable_cape_cin(pressure, temperature, self.dewpoint)[0]
            self.el, self.lfc = el_lfc(
                pressure,
                temperature,
                self.dewpoint,
                _get_profile(most_unstable_parcel, pressure, temperature, self.dewpoint),
            )
        elif cape_type == "mixed_layer":
            self._cape = mixed_layer_cape_cin(pressure, temperature, self.dewpoint)[0]
            self.el, self.lfc = el_lfc(
                pressure,
                temperature,
                self.dewpoint,
                _get_profile(mixed_parcel, pressure, temperature, self.dewpoint),
            )
        elif cape_type == "surface_based":
            self._cape = surface_based_cape_cin(pressure, temperature, self.dewpoint)[0]
            self.el, self.lfc = el_lfc(
                pressure,
                temperature,
                self.dewpoint,
            )
        else:
            assert cape_type.shape == (N,)
            self._cape = cape_type
            self.el, self.lfc = el_lfc(pressure, temperature, self.dewpoint)

    @property
    def u(self) -> MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]]:
        return self.wind[0]

    @property
    def v(self) -> MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]]:
        return self.wind[1]

    @property
    def moist_static_energy(
        self,
    ) -> tuple[
        np.ndarray[shape[N, Z], np.dtype[_T]],
        np.ndarray[shape[N, Z], np.dtype[_T]],
    ]:
        """Calculate the moist static energy terms of interest.

        Returns
        -------
            moist_static_energy_bar:
                Mean moist static energy from the surface to a layer
            moist_static_energy_star:
                Saturated moist static energy
        """
        Z = self.shape[1]
        # calculate MSE_bar
        bar = moist_static_energy(self.height, self.temperature, self.specific_humidity).cumsum(
            axis=1
        ) / np.arange(1, Z + 1)

        # calculate MSE_star
        star = moist_static_energy(
            self.height,
            self.temperature,
            saturation_mixing_ratio(self.pressure, self.temperature),
        )

        return bar.astype(self.dtype, copy=False), star.astype(self.dtype, copy=False)

    @property
    def lfc_index(self) -> np.ndarray[shape[N], np.dtype[np.intp]]:
        return self.lfc.pressure_index(self.pressure) - 1

    @property
    def lfc_height(self) -> Meter[np.ndarray[shape[N], np.dtype[_T]]]:
        return self.height[np.arange(self.shape[0]), self.lfc_index]

    @property
    def el_index(self) -> np.ndarray[shape[N], np.dtype[np.intp]]:
        return self.el.pressure_index(self.pressure) - 1

    @property
    def el_height(self) -> Meter[np.ndarray[shape[N], np.dtype[_T]]]:
        return self.height[np.arange(self.shape[0]), self.el_index]

    @property
    def storm_relative_motion(self) -> np.ndarray[shape[N], np.dtype[_T]]:
        height_agl = self.height - self.height[:, :1]
        right_mover = bunkers_storm_motion(self.pressure, self.u, self.v, height_agl)[0].reshape(
            2, -1, 1
        )  # ((U, V), N, 1)

        u, v = self.wind - right_mover  # ((U, V), N, Z)
        where = (height_agl >= 0) & (height_agl <= 1000)
        mag = wind_magnitude(u, v, where=where, out=np.full_like(u, fill_value=np.nan))

        return np.nanmean(mag, axis=1)

    @property
    def integral(self) -> np.ndarray[shape[N], np.dtype[_T]]:
        dry_air_spec_heat_ratio = 1.4

        cp_d = dry_air_spec_heat_ratio * Rd / (dry_air_spec_heat_ratio - 1)
        # dry_air_spec_heat_ratio = 0.14

        temperature = self.temperature
        bar, star = self.moist_static_energy
        # NCAPE eqn 54 integrand, see compute_NCAPE.m L32
        return -(g / (cp_d * temperature)) * (bar - star)  # type: ignore

    @property
    def ncape(self) -> np.ndarray[shape[N], np.dtype[_T]]:
        integral = self.integral
        height = self.height  # (N, Z)
        lfc_height = self.lfc_height.reshape(-1, 1)  # (N, 1)
        el_height = self.el_height.reshape(-1, 1)  # (N, 1)

        nx, z0 = between_or_close(height, lfc_height, el_height).nonzero()
        z1 = z0 + 1

        full = np.zeros_like(height)
        full[nx, z0] = (0.5 * integral[nx, z0] + 0.5 * integral[nx, z1]) * (
            height[nx, z1] - height[nx, z0]
        )

        ncape = np.sum(full, axis=1)
        ncape[(nx[z1 == height.shape[1] - 1])] = 0

        return ncape

    @property
    def psi(self) -> np.ndarray[shape[N], np.dtype[_T]]:
        sigma = 1.6
        alpha = 0.8
        l_mix = 120.0
        pr = 1.0 / 3.0
        ksq = 0.18

        return (ksq * alpha**2 * np.pi**2 * l_mix) / (4.0 * pr * sigma**2 * self.el_height)  # type: ignore

    @property
    def ecape_a(self) -> np.ndarray[shape[N], np.dtype[_T]]:
        srm = self.storm_relative_motion
        psi = self.psi
        ncape = self.ncape

        a = srm**2 / 2.0
        b = (-1 - psi - (2 * psi / srm**2) * ncape) / (4 * psi / srm**2)
        c = np.sqrt(
            (1 + psi + (2 * psi / srm**2) * ncape) ** 2
            + 8 * (psi / srm**2) * (self._cape - (psi * ncape))
        ) / (4 * psi / srm**2)

        ecape_a = np.sum([a, b, c], axis=0)
        # set to 0 if negative
        m = ecape_a < 0.0
        ecape_a[m] = 0.0

        return ecape_a


def ecape(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[_T]]],
    height: Meter[np.ndarray[shape[N, Z], np.dtype[_T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]],
    u_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    v_wind: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    /,
    specific_humidity: Dimensionless[np.ndarray[shape[N, Z], np.dtype[_T]]] | None = None,
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[_T]]] | None = None,
    cape_type: np.ndarray[shape[N], np.dtype[np.floating[Any]]]
    | L["most_unstable", "surface_based", "mixed_layer"] = "most_unstable",
) -> np.ndarray[shape[N], np.dtype[_T]]:
    e = entrainment(
        pressure,
        height,
        temperature,
        u_wind,
        v_wind,
        specific_humidity=specific_humidity,
        dewpoint=dewpoint,
        cape_type=cape_type,
    )

    return e.ecape_a
