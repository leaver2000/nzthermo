# noqa
from __future__ import annotations

import itertools
from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units, units as U
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import nzthermo._core as _C
from nzthermo._core import moist_lapse
from nzthermo.core import (
    cape_cin,
    ccl,
    downdraft_cape,
    el,
    lfc,
    mixed_layer,
    mixed_layer_cape_cin,
    mixed_parcel,
    most_unstable_cape_cin,
    most_unstable_parcel,
    most_unstable_parcel_index,
)

np.set_printoptions(
    precision=6,
    suppress=True,
    threshold=150,
    linewidth=150,
    edgeitems=10,
)

Pa = U.pascal
hPa = U.hectopascal
K = U.kelvin
C = U.celsius

PRESSURE_ABSOLUTE_TOLERANCE = 1e-3
PRESSURE_RELATIVE_TOLERANCE = 1.5e-1
TEMPERATURE_ABSOLUTE_TOLERANCE = 1.0  # temperature is within 1 degree
CAPE_ABSOLUTE_TOLERANCE = 20.0
CIN_ABSOLUTE_TOLERANCE = 40.0


def assert_nan(value: np.ndarray, value_units=None):
    """Check for nan with proper units."""
    check = np.isnan(value)
    if check.size == 1:
        assert check.item(), f"Expected NaN, got {value}"
    else:
        assert np.all(check), f"Expected NaN, got {value}"


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


def pressure_levels(sfc=1013.25, dtype: Any = np.float64):
    pressure = [sfc, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    pressure += [725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200]
    return np.array(pressure, dtype=dtype) * 100.0


# =============================================================================================== #
# nzthermo._core
# =============================================================================================== #
# ............................................................................................... #
# nzthermo._core.moist_lapse
# ............................................................................................... #
# ELEMENT_WISE: (N,) x (N,) x (N,)
@pytest.mark.moist_lapse
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moist_lapse_element_wise(dtype):
    # this mode requires that reference pressure is provided for each temperature value
    dtype = np.dtype(dtype)
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,) :: surface temperature
    pressure = np.array([912.12, 732.93], dtype=dtype) * 100.0
    ref_pressure = np.array([1013.12, 1013.14], dtype=dtype) * 100.0
    assert_allclose(
        moist_lapse(pressure, temperature, ref_pressure).squeeze(),
        [
            mpcalc.moist_lapse(
                pressure[i] * units.pascal,
                temperature[i] * units.kelvin,
                ref_pressure[i] * units.pascal,
            ).m  # type: ignore
            for i in range(len(temperature))
        ],  # type: ignore
        rtol=1e-4,
    )


# BROADCAST: (N,) x (Z,)
@pytest.mark.moist_lapse
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_moist_lapse_broadcasting(dtype):
    dtype = np.dtype(dtype)
    pressure = P.astype(dtype)  # (Z,)
    temperature = np.array([225.31, 254.0], dtype=dtype)  # (N,)

    ml = moist_lapse(pressure.reshape(1, -1), temperature)
    assert ml.dtype == np.dtype(dtype)
    assert_allclose(
        ml,
        [
            mpcalc.moist_lapse(pressure * units.pascal, temperature[i] * units.kelvin).m  # type: ignore
            for i in range(len(temperature))
        ],
        rtol=1e-2,
    )


# MATRIX: (N,) x (N, Z)
@pytest.mark.moist_lapse
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
        [
            mpcalc.moist_lapse(pressure[i] * units.pascal, temperature[i] * units.kelvin).m  # type: ignore
            for i in range(len(temperature))
        ],
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
        [
            x.m
            for x in mpcalc.moist_lapse(
                pressure[0, 2:] * units.pascal, temperature[0] * units.kelvin
            )
        ],
        rtol=1e-4,
    )
    assert_allclose(
        b,
        [
            x.m
            for x in mpcalc.moist_lapse(pressure[1] * units.pascal, temperature[1] * units.kelvin)
        ],
        rtol=1e-4,
    )


# ............................................................................................... #
# nzthermo._core.parcel_profile
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.parcel_profile
def test_parcel_profile_broadcasting() -> None:
    assert_array_equal(
        _C.parcel_profile(P, T[:, 0], Td[:, 0]),
        _C.parcel_profile(np.broadcast_to(P, T.shape), T[:, 0], Td[:, 0]),
    )


@pytest.mark.regression
@pytest.mark.parcel_profile
def test_parcel_profile_metpy_regression() -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])
    for i in range(T.shape[0]):
        prof_ = mpcalc.parcel_profile(
            P * Pa,
            T[i, 0] * K,
            Td[i, 0] * K,
        )
        assert_allclose(prof[i], prof_.m, atol=TEMPERATURE_ABSOLUTE_TOLERANCE)


# ............................................................................................... #
# nzthermo._core.parcel_profile_with_lcl
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.parcel_profile
def test_parcel_profile_with_lcl_broadcasting() -> None:
    p, t, td, tp = _C.parcel_profile_with_lcl(P, T, Td)
    p_, t_, td_, tp = _C.parcel_profile_with_lcl(np.broadcast_to(P, T.shape), T, Td)
    assert_array_equal(p, p_)
    assert_array_equal(t, t_)
    assert_array_equal(td, td_)
    assert_array_equal(tp, tp)


@pytest.mark.regression
@pytest.mark.parcel_profile
def test_parcel_profile_with_lcl_metpy_regression() -> None:
    ep, et, etd, ptp = _C.parcel_profile_with_lcl(P, T[:, :], Td[:, :])
    for i in range(ep.shape[0]):
        ep_, et_, etd_, pt_ = mpcalc.parcel_profile_with_lcl(
            P * Pa,
            T[i, :] * K,
            Td[i, :] * K,
        )
        assert_allclose(ep[i], ep_.m, rtol=1e-3)
        assert_allclose(et[i], et_.m, rtol=1e-3)
        assert_allclose(etd[i], etd_.m, rtol=1e-3)
        assert_allclose(ptp[i], pt_.m, rtol=1e-3)


# =============================================================================================== #
# nzthermo.core
# =============================================================================================== #
# ............................................................................................... #
# nzthermo.core.ccl
# ............................................................................................... #
@pytest.mark.ccl
@pytest.mark.parametrize(
    "dtype,mixed_layer_depth,which",
    itertools.product([np.float64, np.float32], [None, 10000], ["bottom", "top"]),  # type: ignore
)
def test_ccl_metpy_regression(dtype, mixed_layer_depth, which) -> None:
    if mixed_layer_depth is not None:
        # this is dependent on the interpolation of the mixed_layer function which is not implemented
        with pytest.raises(NotImplementedError):
            CCL_P, CCL_T, CT = ccl(
                P.astype(dtype),
                T.astype(dtype),
                Td.astype(dtype),
                mixed_layer_depth=mixed_layer_depth,
                which=which,
            )
    else:
        CCL_P, CCL_T, CT = ccl(
            P.astype(dtype),
            T.astype(dtype),
            Td.astype(dtype),
            mixed_layer_depth=mixed_layer_depth,
            which=which,
        )
        for i in range(T.shape[1]):
            CCL_P_, CCL_T_, CT_ = mpcalc.ccl(
                P.astype(dtype) * Pa,
                T[i].astype(dtype) * K,
                Td[i].astype(dtype) * K,
                mixed_layer_depth=mixed_layer_depth
                if mixed_layer_depth is None
                else mixed_layer_depth * Pa,
                which=which,
            )

            assert_allclose(CCL_P[i], CCL_P_.m, atol=2)
            assert_allclose(CCL_T[i], CCL_T_.m, atol=2)
            assert_allclose(CT[i], CT_.m, atol=2)


# ............................................................................................... #
# nzthermo.core.downdraft_cape
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.downdraft_cape
def test_downdraft_cape_with_broadcasted_pressure() -> None:
    assert_array_equal(
        downdraft_cape(np.broadcast_to(P, T.shape), T, Td),
        downdraft_cape(P, T, Td),
    )


@pytest.mark.regression
@pytest.mark.downdraft_cape
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_downdraft_cape_metpy_regression(dtype) -> None:
    DCAPE = downdraft_cape(P.astype(dtype), T.astype(dtype), Td.astype(dtype))
    for i in range(T.shape[1]):
        DCAPE_ = mpcalc.downdraft_cape(
            P * Pa,
            T[i] * K,
            Td[i] * K,
        )[0].m
        assert_allclose(DCAPE[i], DCAPE_, rtol=1e-2)


# ............................................................................................... #
# nzthermo.core.el
# ............................................................................................... #
@pytest.mark.el
def test_el() -> None:
    """Test equilibrium layer calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * C
    el_pressure, el_temperature = el(
        levels.to(Pa).m,
        temperatures.to(K).m,
        dewpoints.to(K).m,
    )
    assert_almost_equal(el_pressure, 47100.0, -2)  # 3
    assert_almost_equal(el_temperature, (-11.5603 * C).to(K).m, 1)  # 2


@pytest.mark.el
def test_el_kelvin() -> None:
    """Test that EL temperature returns Kelvin if Kelvin is provided."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = (np.array([22.2, 14.6, 12.0, 9.4, 7.0, -38.0]) + 273.15) * K
    dewpoints = (np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) + 273.15) * K
    el_pressure, el_temp = el(
        levels.to(Pa).m,
        temperatures.to(K).m,
        dewpoints.to(K).m,
    )
    assert_almost_equal(el_pressure, 47100.0, -2)  # 3
    assert_almost_equal(el_temp, (-11.5603 * C).to(K).m, 1)  # 2
    # assert el_temp.units == temperatures.units


@pytest.mark.el
@pytest.mark.skip(reason="the mixed parcel profile is not implemented yet")
def test_el_ml() -> None:
    """Test equilibrium layer calculation for a mixed parcel."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 400.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -25.0, -35.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -35.0, -53.2]) * C
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = _C.parcel_profile(levels, t_mixed, td_mixed)
    el_pressure, el_temperature = el(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(el_pressure, 350.0561 * hPa, 3)
    assert_almost_equal(el_temperature, -28.36156 * C, 3)


@pytest.mark.el
def test_no_el() -> None:
    """Test equilibrium layer calculation when there is no EL in the data."""
    levels = np.array([959.0, 867.9, 779.2, 647.5, 472.5, 321.9, 251.0]) * hPa
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * C
    dewpoints = np.array([19.0, 14.3, -11.2, -16.7, -21.0, -43.3, -56.7]) * C
    el_pressure, el_temperature = el(
        levels,
        temperatures,
        dewpoints,
    )
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


@pytest.mark.el
def test_no_el_multi_crossing() -> None:
    """Test el calculation with no el and several parcel path-profile crossings."""
    levels = [
        918.0,
        911.0,
        880.0,
        873.9,
        850.0,
        848.0,
        843.5,
        818.0,
        813.8,
        785.0,
        773.0,
        763.0,
        757.5,
        730.5,
        700.0,
        679.0,
        654.4,
        645.0,
        643.9,
    ] * hPa
    temperatures = [
        24.2,
        22.8,
        19.6,
        19.1,
        17.0,
        16.8,
        16.5,
        15.0,
        14.9,
        14.4,
        16.4,
        16.2,
        15.7,
        13.4,
        10.6,
        8.4,
        5.7,
        4.6,
        4.5,
    ] * C
    dewpoints = [
        19.5,
        17.8,
        16.7,
        16.5,
        15.8,
        15.7,
        15.3,
        13.1,
        12.9,
        11.9,
        6.4,
        3.2,
        2.6,
        -0.6,
        -4.4,
        -6.6,
        -9.3,
        -10.4,
        -10.5,
    ] * C
    el_pressure, el_temperature = el(
        levels,
        temperatures,
        dewpoints,
    )

    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


@pytest.mark.el
@pytest.mark.lfc
def test_lfc_and_el_below_lcl() -> None:
    """Test that LFC and EL are returned as NaN if both are below LCL."""
    dewpoint = [264.5351, 261.13443, 259.0122, 252.30063, 248.58017, 242.66582] * K
    temperature = [273.09723, 268.40173, 263.56207, 260.257, 256.63538, 252.91345] * K
    pressure = [1017.16, 950, 900, 850, 800, 750] * units.hPa

    el_pressure, el_temperature = el(pressure, temperature, dewpoint)
    lfc_pressure, lfc_temperature = lfc(pressure, temperature, dewpoint)

    assert_nan(lfc_pressure, pressure.units)
    assert_nan(lfc_temperature, temperature.units)

    assert_nan(el_pressure, pressure.units)
    assert_nan(el_temperature, temperature.units)


@pytest.mark.el
def test_el_lfc_equals_lcl() -> None:
    """Test equilibrium layer calculation when the lfc equals the lcl."""
    levels = [
        912.0,
        905.3,
        874.4,
        850.0,
        815.1,
        786.6,
        759.1,
        748.0,
        732.3,
        700.0,
        654.8,
        606.8,
        562.4,
        501.8,
        500.0,
        482.0,
        400.0,
        393.3,
        317.1,
        307.0,
        300.0,
        252.7,
        250.0,
        200.0,
        199.3,
        197.0,
        190.0,
        172.0,
        156.6,
        150.0,
        122.9,
        112.0,
        106.2,
        100.0,
    ] * hPa
    temperatures = [
        29.4,
        28.7,
        25.2,
        22.4,
        19.4,
        16.8,
        14.3,
        13.2,
        12.6,
        11.4,
        7.1,
        2.2,
        -2.7,
        -10.1,
        -10.3,
        -12.4,
        -23.3,
        -24.4,
        -38.0,
        -40.1,
        -41.1,
        -49.8,
        -50.3,
        -59.1,
        -59.1,
        -59.3,
        -59.7,
        -56.3,
        -56.9,
        -57.1,
        -59.1,
        -60.1,
        -58.6,
        -56.9,
    ] * C
    dewpoints = [
        18.4,
        18.1,
        16.6,
        15.4,
        13.2,
        11.4,
        9.6,
        8.8,
        0.0,
        -18.6,
        -22.9,
        -27.8,
        -32.7,
        -40.1,
        -40.3,
        -42.4,
        -53.3,
        -54.4,
        -68.0,
        -70.1,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
        -70.0,
    ] * C
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)

    assert_almost_equal(el_pressure, 17573.273, 3)  # 175.7663 * hPa, 3)
    assert_almost_equal(el_temperature, 216.117, 3)  # -57.03994 * C, 3)


@pytest.mark.el
@pytest.mark.skip
def test_el_small_surface_instability() -> None:
    """Test that no EL is found when there is a small pocket of instability at the sfc."""
    levels = [
        959.0,
        931.3,
        925.0,
        899.3,
        892.0,
        867.9,
        850.0,
        814.0,
        807.9,
        790.0,
        779.2,
        751.3,
        724.3,
        700.0,
        655.0,
        647.5,
        599.4,
        554.7,
        550.0,
        500.0,
    ] * hPa
    temperatures = [
        22.2,
        20.2,
        19.8,
        18.4,
        18.0,
        17.4,
        17.0,
        15.4,
        15.4,
        15.6,
        14.6,
        12.0,
        9.4,
        7.0,
        2.2,
        1.4,
        -4.2,
        -9.7,
        -10.3,
        -14.9,
    ] * C
    dewpoints = [
        20.0,
        18.5,
        18.1,
        17.9,
        17.8,
        15.3,
        13.5,
        6.4,
        2.2,
        -10.4,
        -10.2,
        -9.8,
        -9.4,
        -9.0,
        -15.8,
        -15.7,
        -14.8,
        -14.0,
        -13.9,
        -17.9,
    ] * C
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


@pytest.mark.el
def test_no_el_parcel_colder() -> None:
    """Test no EL when parcel stays colder than environment. INL 20170925-12Z."""
    levels = [
        974.0,
        946.0,
        925.0,
        877.2,
        866.0,
        850.0,
        814.6,
        785.0,
        756.6,
        739.0,
        729.1,
        700.0,
        686.0,
        671.0,
        641.0,
        613.0,
        603.0,
        586.0,
        571.0,
        559.3,
        539.0,
        533.0,
        500.0,
        491.0,
        477.9,
        413.0,
        390.0,
        378.0,
        345.0,
        336.0,
    ] * hPa
    temperatures = [
        10.0,
        8.4,
        7.6,
        5.9,
        7.2,
        7.6,
        6.8,
        7.1,
        7.7,
        7.8,
        7.7,
        5.6,
        4.6,
        3.4,
        0.6,
        -0.9,
        -1.1,
        -3.1,
        -4.7,
        -4.7,
        -6.9,
        -7.5,
        -11.1,
        -10.9,
        -12.1,
        -20.5,
        -23.5,
        -24.7,
        -30.5,
        -31.7,
    ] * C
    dewpoints = [
        8.9,
        8.4,
        7.6,
        5.9,
        7.2,
        7.0,
        5.0,
        3.6,
        0.3,
        -4.2,
        -12.8,
        -12.4,
        -8.4,
        -8.6,
        -6.4,
        -7.9,
        -11.1,
        -14.1,
        -8.8,
        -28.1,
        -18.9,
        -14.5,
        -15.2,
        -15.1,
        -21.6,
        -41.5,
        -45.5,
        -29.6,
        -30.6,
        -32.1,
    ] * C
    el_pressure, el_temperature = el(levels, temperatures, dewpoints)
    assert_nan(el_pressure, levels.units)
    assert_nan(el_temperature, temperatures.units)


@pytest.mark.el
def test_el_below_lcl() -> None:
    """Test LFC when there is positive area below the LCL (#1003)."""
    p = [
        902.1554,
        897.9034,
        893.6506,
        889.4047,
        883.063,
        874.6284,
        866.2387,
        857.887,
        849.5506,
        841.2686,
        833.0042,
        824.7891,
        812.5049,
        796.2104,
        776.0027,
        751.9025,
        727.9612,
        704.1409,
        680.4028,
        656.7156,
        629.077,
        597.4286,
        565.6315,
        533.5961,
        501.2452,
        468.493,
        435.2486,
        401.4239,
        366.9387,
        331.7026,
        295.6319,
        258.6428,
        220.9178,
        182.9384,
        144.959,
        106.9778,
        69.00213,
    ] * units.hPa
    t = [
        -3.039381,
        -3.703779,
        -4.15996,
        -4.562574,
        -5.131827,
        -5.856229,
        -6.568434,
        -7.276881,
        -7.985013,
        -8.670911,
        -8.958063,
        -7.631381,
        -6.05927,
        -5.083627,
        -5.11576,
        -5.687552,
        -5.453021,
        -4.981445,
        -5.236665,
        -6.324916,
        -8.434324,
        -11.58795,
        -14.99297,
        -18.45947,
        -21.92021,
        -25.40522,
        -28.914,
        -32.78637,
        -37.7179,
        -43.56836,
        -49.61077,
        -54.24449,
        -56.16666,
        -57.03775,
        -58.28041,
        -60.86264,
        -64.21677,
    ] * C
    td = [
        -22.08774,
        -22.18181,
        -22.2508,
        -22.31323,
        -22.4024,
        -22.51582,
        -22.62526,
        -22.72919,
        -22.82095,
        -22.86173,
        -22.49489,
        -21.66936,
        -21.67332,
        -21.94054,
        -23.63561,
        -27.17466,
        -31.87395,
        -38.31725,
        -44.54717,
        -46.99218,
        -43.17544,
        -37.40019,
        -34.3351,
        -36.42896,
        -42.1396,
        -46.95909,
        -49.36232,
        -48.94634,
        -47.90178,
        -49.97902,
        -55.02753,
        -63.06276,
        -72.53742,
        -88.81377,
        -93.54573,
        -92.92464,
        -91.57479,
    ] * C
    prof = _C.parcel_profile(
        p.to(Pa).m,
        t[:1].to(K).m,
        td[:1].to(K).m,
    )
    el_p, el_t = el(
        p.to(Pa).m,
        t.to(K).m,
        td.to(K).m,
        prof,
    )
    assert_nan(el_p, p.units)
    assert_nan(el_t, t.units)


@pytest.mark.el
@pytest.mark.skip(reason="nan values are not handled properly")
def test_el_profile_nan() -> None:
    """Test EL when the profile includes NaN values."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, np.nan, 9.4, 7.0, -38.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, np.nan, -53.2]) * C
    el_pressure, el_temperature = el(
        levels.to(Pa).m,
        temperatures.to(K).m,
        dewpoints.to(K).m,
    )
    # assert_almost_equal(el_pressure, 673.0104 * hPa, 3)
    # assert_almost_equal(el_temperature, 5.8853 * C, 3)
    assert_almost_equal(
        el_pressure,
        67301.04,
        3,
    )
    assert_almost_equal(
        el_temperature,
        5.8853,  # 5.8853 * C,
        3,
    )


@pytest.mark.el
@pytest.mark.skip(reason="nan values are not handled properly")
def test_el_profile_nan_with_parcel_profile() -> None:
    """Test EL when the profile includes NaN values, and a parcel temp profile is specified."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, np.nan, 9.4, 7.0, -38.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, np.nan, -53.2]) * C
    parcel_temps = _C.parcel_profile(
        levels.to(Pa).m,
        temperatures[:1].to(K).m,
        dewpoints[:1].to(K).m,
    )  # .to("degC")
    el_pressure, el_temperature = el(
        levels.to(Pa).m,
        temperatures.to(K).m,
        dewpoints.to(K).m,
        parcel_temps,
    )

    assert_almost_equal(el_pressure, 673.0104 * hPa, 3)
    assert_almost_equal(el_temperature, 5.8853 * C, 3)


@pytest.mark.el
@pytest.mark.regression
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_el_metpy_regression(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    el_p, el_t = el(P, T, Td, prof, which=which)
    for i in range(T.shape[0]):
        el_p_, el_t_ = mpcalc.el(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which=which,
        )

        assert_allclose(el_p[i], el_p_.m, atol=PRESSURE_ABSOLUTE_TOLERANCE)
        assert_allclose(el_t[i], el_t_.m, atol=TEMPERATURE_ABSOLUTE_TOLERANCE)


# ............................................................................................... #
# nzthermo.core.lfc
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.lfc
def test_lfc_broadcasting() -> None:
    assert_array_equal(
        lfc(P, T, Td),
        lfc(np.broadcast_to(P, T.shape), T, Td),
    )


@pytest.mark.lfc
def test_lfc_basic() -> None:
    """Test LFC calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * C
    lfc_p, lfc_t = lfc(
        levels.reshape(1, -1).to(Pa).magnitude,
        temperatures.reshape(1, -1).to(K).magnitude,
        dewpoints.reshape(1, -1).to(K).magnitude,
    )
    lfc_p = (lfc_p.squeeze() * Pa).to(hPa).magnitude
    lfc_t = (lfc_t.squeeze() * K).to(C).magnitude

    assert_almost_equal(lfc_p, 727.371, 2)
    assert_almost_equal(lfc_t, 9.705, 2)


@pytest.mark.lfc
@pytest.mark.skip(reason="the mixed parcel profile is not implemented yet")
def test_lfc_ml() -> None:
    """Test Mixed-Layer LFC calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * C
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = _C.parcel_profile(levels, t_mixed, td_mixed)
    lfc_p, lfc_t = lfc(levels, temperatures, dewpoints, mixed_parcel_prof)

    assert_almost_equal(lfc_p, 601.225 * hPa, 2)
    assert_almost_equal(lfc_t, -1.90688 * C, 2)


@pytest.mark.lfc
def test_no_lfc() -> None:
    """Test LFC calculation when there is no LFC in the data."""
    levels = np.array([959.0, 867.9, 779.2, 647.5, 472.5, 321.9, 251.0]) * hPa
    temperatures = np.array([22.2, 17.4, 14.6, 1.4, -17.6, -39.4, -52.5]) * C
    dewpoints = np.array([9.0, 4.3, -21.2, -26.7, -31.0, -53.3, -66.7]) * C
    lfc_p, lfc_t = lfc(
        levels.reshape(1, -1).to(Pa).magnitude,
        temperatures.reshape(1, -1).to(K).magnitude,
        dewpoints.reshape(1, -1).to(K).magnitude,
    )

    np.testing.assert_(np.isnan(lfc_p))
    np.testing.assert_(np.isnan(lfc_t))


@pytest.mark.lfc
def test_lfc_inversion() -> None:
    """Test LFC when there is an inversion to be sure we don't pick that."""
    levels = (
        np.array([963.0, 789.0, 782.3, 754.8, 728.1, 727.0, 700.0, 571.0, 450.0, 300.0, 248.0])
        * hPa
    )
    temperatures = (
        np.array([25.4, 18.4, 17.8, 15.4, 12.9, 12.8, 10.0, -3.9, -16.3, -41.1, -51.5]) * C
    )
    dewpoints = np.array([20.4, 0.4, -0.5, -4.3, -8.0, -8.2, -9.0, -23.9, -33.3, -54.1, -63.5]) * C
    lfc_p, lfc_t = lfc(
        levels.reshape(1, -1).to(Pa).magnitude,
        temperatures.reshape(1, -1).to(K).magnitude,
        dewpoints.reshape(1, -1).to(K).magnitude,
    )
    lfc_p = (lfc_p.squeeze() * Pa).to(hPa).magnitude
    lfc_t = (lfc_t.squeeze() * K).to(C).magnitude

    assert_almost_equal(lfc_p, 705.8806, 1)  # RETURNS: 705.86196532424
    assert_almost_equal(lfc_t, 10.6232, 2)


@pytest.mark.lfc
def test_lfc_equals_lcl() -> None:
    """Test LFC when there is no cap and the lfc is equal to the lcl."""
    levels = (
        np.array([912.0, 905.3, 874.4, 850.0, 815.1, 786.6, 759.1, 748.0, 732.2, 700.0, 654.8])
        * hPa
    )
    temperatures = np.array([29.4, 28.7, 25.2, 22.4, 19.4, 16.8, 14.0, 13.2, 12.6, 11.4, 7.1]) * C
    dewpoints = np.array([18.4, 18.1, 16.6, 15.4, 13.2, 11.4, 9.6, 8.8, 0.0, -18.6, -22.9]) * C
    lfc_p, lfc_t = lfc(
        levels.reshape(1, -1).to(Pa).magnitude,
        temperatures.reshape(1, -1).to(K).magnitude,
        dewpoints.reshape(1, -1).to(K).magnitude,
    )
    lfc_p = (lfc_p.squeeze() * Pa).to(hPa).magnitude
    lfc_t = (lfc_t.squeeze() * K).to(C).magnitude

    assert_almost_equal(lfc_p, 777.0786, 2)
    assert_almost_equal(lfc_t, 15.8714, 2)


@pytest.mark.lfc
def test_lfc_profile_nan() -> None:
    """Test LFC when the profile includes NaN values."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, np.nan, 9.4, 7.0, -38.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, np.nan, -53.2]) * C
    lfc_p, lfc_t = lfc(
        levels.reshape(1, -1).to(Pa).magnitude,
        temperatures.reshape(1, -1).to(K).magnitude,
        dewpoints.reshape(1, -1).to(K).magnitude,
    )
    lfc_p = (lfc_p.squeeze() * Pa).to(hPa).magnitude
    lfc_t = (lfc_t.squeeze() * K).to(C).magnitude

    assert_almost_equal(lfc_p, 727.3365, -1)  # RETURNS: 725.265238330742
    assert_almost_equal(lfc_t, 9.6977, 0)  # RETURNS: 9.58921545636997


@pytest.mark.regression
@pytest.mark.lfc
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_lfc_metpy_regression(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    lfc_p, lfc_t = lfc(P, T, Td, prof, which=which)
    for i in range(T.shape[0]):
        lfc_p_, lfc_t_ = mpcalc.lfc(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which=which,
        )
        assert_allclose(lfc_p[i], lfc_p_.m, atol=500.0)  # type: ignore
        assert_allclose(lfc_t[i], lfc_t_.m, atol=1.0)


# ----------------------------------------------------------------------------------------------- #
# CAPE CIN
# ----------------------------------------------------------------------------------------------- #
# ............................................................................................... #
# nzthermo.core.cape_cin
# ............................................................................................... #
@pytest.mark.cape_cin
@pytest.mark.broadcasting
def test_cape_cin_broadcasting():
    assert_array_equal(
        cape_cin(P, T, Td, _C.parcel_profile(P, T[:, 0], Td[:, 0])),
        cape_cin(
            np.broadcast_to(P, T.shape),
            T,
            Td,
            _C.parcel_profile(np.broadcast_to(P, T.shape), T[:, 0], Td[:, 0]),
        ),
    )


@pytest.mark.cape_cin
@pytest.mark.regression
@pytest.mark.parametrize(
    "which_lfc, which_el",
    itertools.product(["top", "bottom"], ["top", "bottom"]),
)
def test_cape_cin_metpy_regression(which_lfc, which_el) -> None:
    """
    TODO currently this test is passing on 95% of the cases, need to investigate the.
    there error appears to be something in the logic block of the el_lfc function.

    The current test cases run 500 samples and we are failing on 17 of them specifically when
    `which_el=bottom` parameter is used. realistically using the lower EL is not a typical use
    case but it should still be tested.
    """
    parcel_profile = _C.parcel_profile(P, T[:, 0], Td[:, 0])
    CAPE, CIN = cape_cin(
        P,
        T,
        Td,
        parcel_profile,
        which_lfc=which_lfc,
        which_el=which_el,
    )

    for i in range(T.shape[0]):
        CAPE_, CIN_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            parcel_profile[i] * K,
            which_lfc=which_lfc,
            which_el=which_el,
        )

        assert_allclose(CAPE[i], CAPE_.m, atol=10)
        assert_allclose(CIN[i], CIN_.m, atol=10)


# ----------------------------------------------------------------------------------------------- #
# MOST UNSTABLE
# ----------------------------------------------------------------------------------------------- #
# ............................................................................................... #
# nzthermo.core.most_unstable_parcel_index
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.most_unstable_parcel
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_parcel_index_broadcasting(depth) -> None:
    assert_array_equal(
        most_unstable_parcel_index(P, T, Td, depth=depth),
        most_unstable_parcel_index(np.broadcast_to(P, T.shape), T, Td, depth=depth),
    )


@pytest.mark.regression
@pytest.mark.most_unstable_parcel
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_parcel_index(depth) -> None:
    assert_array_equal(
        most_unstable_parcel_index(P, T, Td, depth=depth),
        [
            mpcalc.most_unstable_parcel(P * Pa, T[i] * K, Td[i] * K, depth=depth * Pa)[-1]
            for i in range(T.shape[0])
        ],
    )


# ............................................................................................... #
# nzthermo.core.most_unstable_parcel
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.most_unstable_parcel
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_parcel_broadcasting(depth) -> None:
    assert_array_equal(
        most_unstable_parcel(P, T, Td, depth=depth),
        most_unstable_parcel(np.broadcast_to(P, T.shape), T, Td, depth=depth),
    )


@pytest.mark.regression
@pytest.mark.most_unstable_parcel
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_parcel_regression(depth) -> None:
    p, t, td, idx = most_unstable_parcel(P, T, Td, depth=depth)

    for i in range(T.shape[0]):
        p_, t_, td_, idx_ = mpcalc.most_unstable_parcel(
            P * Pa, T[i] * K, Td[i] * K, depth=depth * Pa
        )
        assert_array_equal(p[i], p_.m)
        assert_array_equal(t[i], t_.m)
        assert_array_equal(td[i], td_.m)
        assert_array_equal(idx[i], idx_)


# ............................................................................................... #
# nzthermo.core.most_unstable_cape_cin
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.most_unstable_cape_cin
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_cape_cin_broadcasting(depth) -> None:
    assert_array_equal(
        most_unstable_cape_cin(P, T, Td, depth=depth),
        most_unstable_cape_cin(np.broadcast_to(P, T.shape), T, Td, depth=depth),
    )


@pytest.mark.regression
@pytest.mark.most_unstable_cape_cin
@pytest.mark.parametrize("depth", [30000.0])
def test_most_unstable_cape_cin_metpy_regression(depth) -> None:
    CAPE, CIN = most_unstable_cape_cin(
        P,
        T,
        Td,
        depth=depth,
    )

    for i in range(T.shape[0]):
        CAPE_, CIN_ = mpcalc.most_unstable_cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            depth=depth * Pa,
        )
        assert_allclose(CAPE[i], CAPE_.m, atol=10)
        assert_allclose(CIN[i], CIN_.m, atol=20)


# ----------------------------------------------------------------------------------------------- #
# MIXED LAYER
# ----------------------------------------------------------------------------------------------- #
# ............................................................................................... #
# nzthermo.core.mixed_layer
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.mixed_layer
def test_mixed_layer_broadcasting() -> None:
    assert_array_equal(
        mixed_layer(P, T, Td),
        mixed_layer(np.broadcast_to(P, T.shape), T, Td),
    )


@pytest.mark.regression
@pytest.mark.mixed_layer
@pytest.mark.parametrize("interpolate", [True, False])
def test_mixed_layer_regression(interpolate) -> None:
    if interpolate:
        with pytest.raises(NotImplementedError):
            mixed_layer(P, T, Td, interpolate=interpolate)
    else:
        t, td = mixed_layer(P, T, Td, interpolate=interpolate)
        for i in range(T.shape[0]):
            t_, td_ = mpcalc.mixed_layer(P * Pa, T[i] * K, Td[i] * K, interpolate=interpolate)
            assert_allclose(
                t[i],
                t_.m,
                atol=TEMPERATURE_ABSOLUTE_TOLERANCE,
            )
            assert_allclose(
                td[i],
                td_.m,
                atol=TEMPERATURE_ABSOLUTE_TOLERANCE,
            )


# ............................................................................................... #
# nzthermo.core.mixed_parcel
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.mixed_parcel
def test_mixed_parcel_broadcasting() -> None:
    assert_array_equal(
        mixed_parcel(P, T, Td),
        mixed_parcel(np.broadcast_to(P, T.shape), T, Td),
    )


@pytest.mark.regression
@pytest.mark.mixed_parcel
def test_mixed_parcel_regression() -> None:
    p, t, td = mixed_parcel(P, T, Td)

    for i in range(T.shape[0]):
        p_, t_, td_ = mpcalc.mixed_parcel(P * Pa, T[i] * K, Td[i] * K, interpolate=False)
        assert_allclose(
            p[i],
            p_.m,
            atol=PRESSURE_ABSOLUTE_TOLERANCE,
        )

        assert_allclose(
            t[i],
            t_.m,
            atol=TEMPERATURE_ABSOLUTE_TOLERANCE,
        )
        assert_allclose(
            td[i],
            td_.m,
            atol=TEMPERATURE_ABSOLUTE_TOLERANCE,
        )


# ............................................................................................... #
# nzthermo.core.mixed_layer_cape_cin
# ............................................................................................... #
@pytest.mark.broadcasting
@pytest.mark.mixed_layer_cape_cin
def test_mixed_layer_cape_cin_broadcasting() -> None:
    assert_array_equal(
        mixed_layer_cape_cin(P, T, Td),
        mixed_layer_cape_cin(np.broadcast_to(P, T.shape), T, Td),
    )


@pytest.mark.regression
@pytest.mark.mixed_layer_cape_cin
def test_mixed_layer_cape_cin_regression() -> None:
    CAPE, CIN = mixed_layer_cape_cin(P, T, Td)

    for i in range(T.shape[0]):
        CAPE_, CIN__ = mpcalc.mixed_layer_cape_cin(P * Pa, T[i] * K, Td[i] * K, interpolate=False)
        assert_allclose(CAPE[i], CAPE_.m, atol=20)
        assert_allclose(CIN[i], CIN__.m, atol=200)
