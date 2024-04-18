from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo.core import ccl, lcl, wet_bulb_temperature


def pressure_levels(sfc=1013.25, dtype: Any = np.float64):
    pressure = [sfc, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    pressure += [725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200]
    return np.array(pressure, dtype=dtype) * 100.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_wet_bulb_temperature(dtype):
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0  # (N,) :: surface pressure
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,) :: surface temperature
    dewpoint = np.array([220.31, 240.0], dtype=dtype)  # (N,) :: surface temperature

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
def test_ccl(dtype) -> None:
    P = (
        ([993.0, 957.0, 925.0, 886.0, 850.0, 813.0, 798.0, 732.0, 716.0, 700.0] * units.hPa)
        .to("pascal")
        .m.astype(dtype)
    )
    T = ([34.6, 31.1, 27.8, 24.3, 21.4, 19.6, 18.7, 13, 13.5, 13] * units.degC).to("kelvin").m.astype(dtype)
    Td = ([19.6, 18.7, 17.8, 16.3, 12.4, -0.4, -3.8, -6, -13.2, -11] * units.degC).to("kelvin").m.astype(dtype)
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="lower")

    def get_metpy_ccl(p, t, td):
        return [x.m for x in mpcalc.ccl(p * units.pascal, t * units.kelvin, td * units.kelvin, which="bottom")]

    assert_allclose(
        np.ravel((ccl_t, ccl_p, ct)),
        get_metpy_ccl(P, T, Td),
    )
    P = np.array([P, P, P])
    T = np.array([T, T - 1, T])
    Td = np.array([Td, Td, Td - 0.5])
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="lower")

    for i in range(len(P)):
        assert_allclose(
            (ccl_t[i], ccl_p[i], ct[i]),
            get_metpy_ccl(P[i], T[i], Td[i]),
            rtol=1e-4,
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_lcl(dtype) -> None:
    pressure = np.array([912.12, 1012.93], dtype=dtype) * 100.0
    temperature = np.array([225.31, 254.0], dtype=dtype)
    dewpoint = np.array([220.31, 240.0], dtype=dtype)
    lcl_p, lcl_t = lcl(pressure, temperature, dewpoint)
    for i in range(len(temperature)):
        lcl_p_, lcl_t_ = mpcalc.lcl(
            pressure[i] * units.pascal, temperature[i] * units.kelvin, dewpoint[i] * units.kelvin
        )
        assert_allclose(lcl_p[i], lcl_p_.m, rtol=1e-4)  # type: ignore
        assert_allclose(lcl_t[i], lcl_t_.m, rtol=1e-4)
