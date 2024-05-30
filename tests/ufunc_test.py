import pytest
import numpy as np
from numpy.testing import assert_allclose
import metpy.calc as mpcalc
from metpy.units import units
from nzthermo._ufunc import (
    wet_bulb_temperature,
    potential_temperature,
    equivalent_potential_temperature,
    wet_bulb_potential_temperature,
)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wet_bulb_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0  # (N,) :: surface pressure
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,) :: surface temperature
    dewpoint = np.array([220.31, 240.0], dtype=dtype)  # (N,) :: surface temperature
    print(wet_bulb_temperature.ntypes)
    assert_allclose(
        wet_bulb_temperature(pressure, temperature, dewpoint),
        [
            mpcalc.wet_bulb_temperature(
                pressure[i] * units.pascal, temperature[i] * units.kelvin, dewpoint[i] * units.kelvin
            ).m
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
            mpcalc.potential_temperature(pressure[i] * units.pascal, temperature[i] * units.kelvin).m
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_equivalent_potential_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)
    dewpoint = np.array([220.31, 240.0], dtype=dtype)
    assert_allclose(
        equivalent_potential_temperature(pressure, temperature, dewpoint),
        [
            mpcalc.equivalent_potential_temperature(
                pressure[i] * units.pascal, temperature[i] * units.kelvin, dewpoint[i] * units.kelvin
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
                pressure[i] * units.pascal, temperature[i] * units.kelvin, dewpoint[i] * units.kelvin
            ).m
            for i in range(len(temperature))
        ],
        rtol=1e-4,
    )
