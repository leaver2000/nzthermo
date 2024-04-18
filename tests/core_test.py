from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo.core import ccl, downdraft_cape, lcl, wet_bulb_temperature


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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_downdraft_cape(dtype):

    pressure = [
        101300.0,
        100000.0,
        97500.0,
        95000.0,
        92500.0,
        90000.0,
        87500.0,
        85000.0,
        82500.0,
        80000.0,
        77500.0,
        75000.0,
        72500.0,
        70000.0,
        65000.0,
        60000.0,
        55000.0,
        50000.0,
        45000.0,
        40000.0,
        35000.0,
        30000.0,
        25000.0,
        20000.0,
    ]
    temperature = [
        [243.28, 242.68, 241.51, 240.32, 239.11, 237.86, 236.59, 235.29, 233.96, 232.59, 231.19, 229.75, 228.28]
        + [226.76, 235.58, 236.2, 234.42, 231.1, 226.84, 221.98, 217.31, 211.33, 212.78, 217.8],
        [
            250.51,
            249.9,
            248.7,
            247.47,
            246.22,
            244.94,
            243.63,
            242.29,
            240.92,
            239.51,
            238.07,
            236.59,
            235.07,
            233.5,
            240.81,
            239.41,
            236.21,
            232.36,
            227.77,
            223.2,
            217.49,
            211.49,
            212.93,
            217.69,
        ],
        [
            293.98,
            292.88,
            290.72,
            288.65,
            287.23,
            285.78,
            284.23,
            282.75,
            281.25,
            279.78,
            279.28,
            280.3,
            279.95,
            278.4,
            275.25,
            270.91,
            268.2,
            264.33,
            260.04,
            254.35,
            246.67,
            237.88,
            227.78,
            219.37,
        ],
        [
            300.57,
            299.51,
            297.36,
            295.2,
            293.09,
            291.83,
            292.27,
            291.97,
            291.37,
            289.98,
            288.11,
            286.49,
            285.72,
            285.16,
            281.68,
            278.12,
            273.73,
            268.95,
            264.32,
            258.5,
            251.85,
            242.82,
            232.49,
            220.12,
        ],
    ]
    dewpoint = [
        [
            224.91,
            224.8,
            224.58,
            224.35,
            224.11,
            223.87,
            223.62,
            223.37,
            223.11,
            222.84,
            222.57,
            222.29,
            221.99,
            221.69,
            233.71,
            233.66,
            231.75,
            228.18,
            223.59,
            218.66,
            213.55,
            207.44,
            206.11,
            195.79,
        ],
        [
            233.27,
            233.14,
            232.9,
            232.65,
            232.39,
            232.13,
            231.87,
            231.59,
            231.31,
            231.02,
            230.72,
            230.41,
            230.09,
            229.77,
            237.89,
            236.41,
            233.05,
            229.07,
            223.99,
            219.29,
            213.78,
            207.71,
            206.9,
            195.79,
        ],
        [288.79, 288.28, 287.65, 286.91, 281.19, 280.57, 279.05, 277.57, 276.44, 275.11, 270.62, 258.29]
        + [244.87, 247.88, 243.01, 254.39, 262.04, 248.96, 229.05, 232.7, 229.15, 224.4, 217.69, 207.63],
        [
            294.46,
            294.14,
            293.52,
            292.86,
            291.99,
            289.71,
            285.49,
            282.07,
            280.66,
            280.89,
            281.56,
            281.15,
            278.22,
            274.49,
            273.87,
            269.75,
            259.85,
            246.24,
            240.11,
            241.7,
            226.75,
            219.81,
            218.45,
            213.12,
        ],
    ]
