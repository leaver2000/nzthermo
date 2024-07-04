import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose, assert_array_equal

from nzthermo._core import pressure_vector
from nzthermo._ufunc import (
    dewpoint_from_specific_humidity,
    dry_static_energy,
    equivalent_potential_temperature,
    lcl,
    moist_static_energy,
    potential_temperature,
    standard_height,
    standard_pressure,
    wet_bulb_potential_temperature,
    wet_bulb_temperature,
    wind_components,
    wind_vector,
)

Pa = units.pascal
K = units.kelvin
dimensionless = units.dimensionless
WIND_DIRECTIONS = np.array([0, 90, 180, 270, 0])
WIND_MAGNITUDES = np.array([10, 20, 30, 40, 50])


# =============================================================================================== #
# load test data
# =============================================================================================== #
# load up the test data
data = np.load("tests/data.npz", allow_pickle=False)
step = np.s_[:]
P: np.ndarray = data["P"]
T: np.ndarray = data["T"][step]
Td: np.ndarray = data["Td"][step]
# In very rare cases the data accessed from the HRRR model had dewpoint temperatures greater than
# the actual temperature. This is not physically possible and is likely due to rounding errors.
# This also makes testing quite difficult because in many cases metpy will report a nan values
# and throw interpolation warnings. To avoid this we will set the dewpoint temperature to be less
# than the actual temperature.
_super_saturation = Td > T
Td[_super_saturation] = T[_super_saturation]
Q = mpcalc.specific_humidity_from_dewpoint(P * Pa, Td * K).to("g/g").m


def test_height_conversion() -> None:
    height = standard_height(P)
    pressure = standard_pressure(height)
    assert_array_equal(height, P.view(pressure_vector).to_standard_height())
    assert_array_equal(pressure, pressure_vector.from_standard_height(height))

    assert_allclose(
        height,
        mpcalc.pressure_to_height_std(P * Pa).to("m").m,
        rtol=1e-3,
    )
    assert_allclose(
        pressure,
        mpcalc.height_to_pressure_std(height * units.meter).to(Pa).m,
        rtol=1e-3,
    )


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

    u, v = wind_components(WIND_DIRECTIONS, WIND_MAGNITUDES)
    assert_allclose([WIND_DIRECTIONS, WIND_MAGNITUDES], wind_vector(u, v), atol=1e-9)


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


def test_static_energy():
    z = mpcalc.pressure_to_height_std(P * Pa)[np.newaxis, :].to("m")

    assert_allclose(
        dry_static_energy(z.m, T),
        mpcalc.dry_static_energy(z, T * K).to("J/kg").m,
        atol=1e-1,
    )

    assert_allclose(
        moist_static_energy(z.m, T, Q),
        mpcalc.moist_static_energy(z, T * K, Q * dimensionless).to("J/kg").m,
        atol=1e-1,
    )
