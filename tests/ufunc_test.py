import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo._ufunc import (
    equivalent_potential_temperature,
    lcl,
    potential_temperature,
    wet_bulb_potential_temperature,
    wet_bulb_temperature,
    wind_components,
    dewpoint_from_specific_humidity,
)

Pa = units.pascal
K = units.kelvin
dimensionless = units.dimensionless
WIND_DIRECTIONS = np.array([0, 90, 180, 270, 360])
WIND_MAGNITUDES = np.array([10, 20, 30, 40, 50])


def test_dewpoints() -> None:
    pressure = 101325.0
    sh = 0.01

    assert_allclose(
        dewpoint_from_specific_humidity(pressure, sh),
        mpcalc.dewpoint_from_specific_humidity(pressure * Pa, sh * dimensionless).to(K).m,
        rtol=1e-4,
    )


def test_wind_components() -> None:
    assert_allclose(
        wind_components(WIND_DIRECTIONS, WIND_MAGNITUDES),
        [
            x.m
            for x in mpcalc.wind_components(
                WIND_MAGNITUDES * units.meter / units.second, WIND_DIRECTIONS * units.degrees
            )
        ],
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wet_bulb_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0  # (N,) :: surface pressure
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,) :: surface temperature
    dewpoint = np.array([220.31, 240.0], dtype=dtype)  # (N,) :: surface temperature

    assert_allclose(
        wet_bulb_temperature(pressure, temperature, dewpoint),
        [
            mpcalc.wet_bulb_temperature(pressure[i] * Pa, temperature[i] * K, dewpoint[i] * K).m
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_potential_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)

    assert_allclose(
        potential_temperature(pressure, temperature),
        [
            mpcalc.potential_temperature(pressure[i] * Pa, temperature[i] * K).m
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_lcl(dtype) -> None:
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)
    dewpoint = np.array([220.31, 240.0], dtype=dtype)

    lcl_p, lcl_t = lcl(pressure, temperature, dewpoint)

    for i in range(len(temperature)):
        lcl_p_, lcl_t_ = mpcalc.lcl(pressure[i] * Pa, temperature[i] * K, dewpoint[i] * K)
        assert_allclose(lcl_p[i], lcl_p_.m, rtol=1e-4)  # type: ignore
        assert_allclose(lcl_t[i], lcl_t_.m, rtol=1e-4)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_equivalent_potential_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)
    dewpoint = np.array([220.31, 240.0], dtype=dtype)
    assert_allclose(
        equivalent_potential_temperature(pressure, temperature, dewpoint),
        [
            mpcalc.equivalent_potential_temperature(
                pressure[i] * Pa, temperature[i] * K, dewpoint[i] * K
            ).m
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wet_bulb_potential_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)
    dewpoint = np.array([220.31, 240.0], dtype=dtype)
    assert_allclose(
        wet_bulb_potential_temperature(pressure, temperature, dewpoint),
        [
            mpcalc.wet_bulb_potential_temperature(
                pressure[i] * Pa, temperature[i] * K, dewpoint[i] * K
            ).m  # type: ignore
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )
