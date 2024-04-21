from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo.core import moist_lapse


def pressure_levels(sfc=1013.25, dtype: Any = np.float64):
    pressure = [sfc, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    pressure += [725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200]
    return np.array(pressure, dtype=dtype) * 100.0


# ELEMENT_WISE: (N,) x (N,) x (N,)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moist_lapse_mode_1(dtype):
    # this mode requires that reference pressure is provided for each temperature value
    dtype = np.dtype(dtype)
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,) :: surface temperature
    pressure = np.array([912.12, 732.93], dtype=dtype) * 100.0
    ref_pressure = np.array([1013.12, 1013.14], dtype=dtype) * 100.0
    assert_allclose(
        moist_lapse(pressure, temperature, ref_pressure).squeeze(),
        [mpcalc.moist_lapse(pressure[i] * units.pascal, temperature[i] * units.kelvin, ref_pressure[i] * units.pascal).m for i in range(len(temperature))],  # type: ignore
        rtol=1e-4,
    )


# BROADCAST: (N,) x (Z,)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moist_lapse_broadcasting(dtype):
    dtype = np.dtype(dtype)
    pressure = pressure_levels(dtype=dtype)  # (Z,)
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,)

    ml = moist_lapse(pressure.reshape(1, -1), temperature)
    assert ml.dtype == np.dtype(dtype)
    assert_allclose(
        ml,
        [mpcalc.moist_lapse(pressure * units.pascal, temperature[i] * units.kelvin).m for i in range(len(temperature))],  # type: ignore
        rtol=1e-2,
    )


# MATRIX: (N,) x (N, Z)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moist_lapse(dtype):
    dtype = np.dtype(dtype)

    pressure = np.array(
        [
            pressure_levels(1013.12, dtype=dtype),
            pressure_levels(1013.93, dtype=dtype),
        ],
        dtype=dtype,
    )
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,)
    ml = moist_lapse(pressure, temperature)
    assert ml.dtype == np.dtype(dtype)

    assert_allclose(
        ml,
        [mpcalc.moist_lapse(pressure[i] * units.pascal, temperature[i] * units.kelvin).m for i in range(len(temperature))],  # type: ignore
        rtol=1e-4,
    )

    # .....{ working with nan values }.....
    # broadcasting: (N,) x (N, Z,) :: axis = 1 :: with nans
    # the nan value can actually be very useful in a situations where we want to ``mask`` out values in the vertical profile.
    # consider a grid global grid where we only want to calculate the moist lapse rate conditionally below a certain pressure
    # level. We can simply set the pressure values above that level to nan and the function will ignore them.
    pressure = (
        np.array(
            [
                [
                    1013.12,
                    1000,
                    975,
                    950,
                    925,
                    900,
                    875,
                    850,
                    825,
                    800,
                    775,
                    750,
                    725,
                    700,
                    650,
                    600,
                    550,
                    500,
                    450,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    1013.93,
                    1000,
                    975,
                    950,
                    925,
                    900,
                    875,
                    850,
                    825,
                    800,
                    775,
                    750,
                    725,
                    700,
                    650,
                    600,
                    550,
                    500,
                    450,
                    400,
                    350,
                    300,
                    250,
                    200,
                ],
            ],
            dtype=float,
        )
        * 100.0
    )

    with_nans = moist_lapse(pressure, temperature)

    for i in range(len(temperature)):
        nans = np.isnan(pressure[i])
        mp = mpcalc.moist_lapse(pressure[i, ~nans] * units.pascal, temperature[i] * units.kelvin).m  # type: ignore
        assert_allclose(
            with_nans[i][~nans],
            mp,
            rtol=1e-4,
        )

    pressure = (
        np.array(
            [
                [
                    np.nan,
                    np.nan,
                    975,
                    950,
                    925,
                    900,
                    875,
                    850,
                    825,
                    800,
                    775,
                    750,
                    725,
                    700,
                    650,
                    600,
                    550,
                    500,
                    450,
                    400,
                    350,
                    300,
                    250,
                    200,
                ],
                [
                    1013.93,
                    1000,
                    975,
                    950,
                    925,
                    900,
                    875,
                    850,
                    825,
                    800,
                    775,
                    750,
                    725,
                    700,
                    650,
                    600,
                    550,
                    500,
                    450,
                    400,
                    350,
                    300,
                    250,
                    200,
                ],
            ],
            dtype=dtype,
        )
        * 100.0
    )
    a, b = moist_lapse(pressure, temperature)

    assert_allclose(
        a[2:],
        mpcalc.moist_lapse(pressure[0, 2:] * units.pascal, temperature[0] * units.kelvin),
        rtol=1e-4,
    )
    assert_allclose(
        b,
        mpcalc.moist_lapse(pressure[1] * units.pascal, temperature[1] * units.kelvin),
        rtol=1e-4,
    )
