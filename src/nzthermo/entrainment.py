from __future__ import annotations

from typing import Any, Generic, Literal as L, TypeVar

import numpy as np

# -----------------------------------------------------------------------------
from metpy.units import units

from . import _ufunc as uf, functional as F
from ._ufunc import between_or_close, less_or_close, pressure_vector
from .core import lfc, specific_humidity as _specific_humidity
from .typing import Dimensionless, Kelvin, Meter, MeterPerSecond, N, Pascal, Z, shape

K = units.kelvin
meter = units.meter
Pa = units.pascal
s = units.second
# -----------------------------------------------------------------------------
_T = TypeVar("_T", bound=np.floating[Any])


def bunkers_storm_motion(
    pressure: Pascal[pressure_vector[shape[N, Z], np.dtype[_T]]],
    u: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    v: MeterPerSecond[np.ndarray[shape[N, Z], np.dtype[_T]]],
    height: Meter[np.ndarray[shape[N, Z], np.dtype[_T]]],
) -> tuple[np.ndarray[shape[L["(U, V)"], N], np.dtype[_T]], ...]:
    dtype = pressure.dtype
    # mean wind from sfc-6km
    mask = less_or_close(height, 6000.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    args = np.where(mask[np.newaxis, :, :], [u, v], np.nan)
    wind_mean = F.nantrapz(args, x=pressure, axis=-1) / delta

    # mean wind from sfc-500m
    mask = less_or_close(height, 500.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    args = np.where(mask[np.newaxis, :, :], [u, v], np.nan)
    w0500 = F.nantrapz(args, x=pressure, axis=-1) / delta

    mask = between_or_close(height, 5250.0, 6250.0).astype(np.bool_)
    pbot = pressure[np.nanargmax(pressure.where(mask), axis=1)]
    ptop = pressure[np.nanargmin(pressure.where(mask), axis=1)]
    delta = ptop - pbot
    args = np.where(mask[np.newaxis, :, :], [u, v], np.nan)
    w5500 = F.nantrapz(args, x=pressure, axis=-1) / delta

    shear = w5500 - w0500
    shear_cross = shear[::-1]
    shear_cross[1] *= -1
    mag = np.hypot(shear[0], shear[1])

    rdev = shear_cross * (7.5 / mag)
    # Add the deviations to the layer average wind to get the RM motion
    right_mover = wind_mean + rdev

    # Subtract the deviations to get the LM motion
    left_mover = wind_mean - rdev

    return right_mover, left_mover, wind_mean.astype(dtype)


class entrainment(Generic[_T]):
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
    ) -> None:
        super().__init__()
        self.pressure = pressure.view(pressure_vector)
        self.height = height
        self.temperature = temperature
        N, Z = temperature.shape
        if dewpoint is not None and specific_humidity is not None:
            raise ValueError("Only one of specific_humidity or dewpoint can be provided")
        elif specific_humidity is not None:
            self.specific_humidity = specific_humidity
            self.dewpoint = uf.dewpoint_from_specific_humidity(pressure, specific_humidity)
        elif dewpoint is not None:
            self.dewpoint = dewpoint
            self.specific_humidity = _specific_humidity(pressure, dewpoint)
        else:
            raise ValueError("Either specific_humidity or dewpoint must be provided")

        self.wind = (u_wind, v_wind)
        self.shape = N, Z

    @property
    def moist_static_energy(self):
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
        bar = uf.moist_static_energy(self.height, self.temperature, self.specific_humidity).cumsum(
            axis=1
        ) / np.arange(1, Z + 1)

        # calculate MSE_star
        star = uf.moist_static_energy(
            self.height,
            self.temperature,
            uf.saturation_mixing_ratio(self.pressure, self.temperature),
        )

        return bar, star

    @property
    def lfc(self):
        return lfc(self.pressure, self.temperature, self.dewpoint)

    @property
    def lfc_index(self) -> np.ndarray[shape[N], np.dtype[np.intp]]:
        return self.lfc.pressure_index(self.pressure) - 1

    @property
    def lfc_height(self) -> Meter[np.ndarray[shape[N], np.dtype[_T]]]:
        return self.height[np.arange(self.shape[0]), self.lfc_index]

    def storm_relative_motion(
        self,
    ):
        """
        Calculate the mean storm relative (as compared to Bunkers right motion) wind magnitude in the 0-1 km AGL layer

        Args:
            pressure:
                Total atmospheric pressure
            u_wind:
                X component of the wind
            v_wind
                Y component of the wind
            height_msl:
                Atmospheric heights at the levels given by 'pressure'.

        Returns:
            sr_wind:
                0-1 km AGL average storm relative wind magnitude

        """
        u, v = self.wind
        height_agl = self.height - self.height[:, :1]
        # bunkers_right, _, _ = bunkers_storm_motion(
        #     self.pressure, u, v, height_agl
        # )  # right, left, mean

        # u_sr = u - bunkers_right[0]  # u-component
        # v_sr = v - bunkers_right[1]  # v-component

        # u_sr_1km = u_sr[np.nonzero((height_agl >= 0) & (height_agl <= 1000))]
        # v_sr_1km = v_sr[np.nonzero((height_agl >= 0) & (height_agl <= 1000))]

        # sr_wind = np.mean(uf.wind_magnitude(u_sr_1km, v_sr_1km))

        # return sr_wind
