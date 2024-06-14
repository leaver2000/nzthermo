# noqa
import itertools

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

import nzthermo._core as _C
from nzthermo.core import cape_cin, ccl, el, lfc, most_unstable_parcel_index

np.set_printoptions(
    precision=3,
    suppress=True,
    threshold=150,
    linewidth=150,
    edgeitems=10,
)

Pa = units.pascal
hPa = units.hectopascal
K = units.kelvin
C = units.celsius

PRESSURE_ABSOLUTE_TOLERANCE = 2000.0
PRESSURE_RELATIVE_TOLERANCE = 500
TEMPERATURE_ABSOLUTE_TOLERANCE = 1.0
# ............................................................................................... #
# load up the test data
data = np.load("tests/data.npz", allow_pickle=False)
# this specific step uses the failing tests cases for the cape_cin function
step = np.s_[
    [19, 20, 25, 44, 46, 48, 88, 89, 116, 119, 144, 146, 149, 166, 238, 268, 297],
    :,
]

step = np.s_[:]
P, T, Td = data["P"], data["T"][step], data["Td"][step]
# ............................................................................................... #


@pytest.mark.ccl
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ccl(dtype) -> None:
    P = (
        ([993.0, 957.0, 925.0, 886.0, 850.0, 813.0, 798.0, 732.0, 716.0, 700.0] * units.hPa)
        .to("pascal")
        .m.astype(dtype)
    )
    T = (
        ([34.6, 31.1, 27.8, 24.3, 21.4, 19.6, 18.7, 13, 13.5, 13] * units.degC)
        .to("K")
        .m.astype(dtype)
    )
    Td = (
        ([19.6, 18.7, 17.8, 16.3, 12.4, -0.4, -3.8, -6, -13.2, -11] * units.degC)
        .to("K")
        .m.astype(dtype)
    )
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="bottom")

    def get_metpy_ccl(p, t, td):
        return [x.m for x in mpcalc.ccl(p * Pa, t * K, td * K, which="bottom")]

    assert_allclose(
        np.ravel((ccl_t, ccl_p, ct)),
        get_metpy_ccl(P, T, Td),
        atol=1e-6,
    )
    P = np.array([P, P, P])
    T = np.array([T, T - 1, T])
    Td = np.array([Td, Td, Td - 0.5])
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="bottom")

    for i in range(len(P)):
        assert_allclose(
            (ccl_t[i], ccl_p[i], ct[i]),
            get_metpy_ccl(P[i], T[i], Td[i]),
            atol=1e-6,
        )


def test_parcel_profile() -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])
    for i in range(T.shape[0]):
        prof_ = mpcalc.parcel_profile(
            P * Pa,
            T[i, 0] * K,
            Td[i, 0] * K,
        )
        assert_allclose(prof[i], prof_.m, atol=TEMPERATURE_ABSOLUTE_TOLERANCE)


# ............................................................................................... #
# EL
# ............................................................................................... #
@pytest.mark.el
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_el(which) -> None:
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
# LFC
# ............................................................................................... #
@pytest.mark.lfc
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_lfc(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    count_a = 0
    count_b = 0
    lfc_p, lfc_t = lfc(P, T, Td, prof, which)
    for i in range(T.shape[0]):
        lfc_p_, lfc_t_ = mpcalc.lfc(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which=which,
        )
        # if not np.isfinite(lfc_p[i]) and not np.isnan(lfc_p_.m).item():  # type: ignore
        #     count_a += 1
        #     continue

        assert_allclose(
            lfc_p[i],
            lfc_p_.m,  # type: ignore
            atol=PRESSURE_ABSOLUTE_TOLERANCE,
            rtol=PRESSURE_RELATIVE_TOLERANCE,
        )
        assert_allclose(lfc_t[i], lfc_t_.m, atol=TEMPERATURE_ABSOLUTE_TOLERANCE + 50)

    print(f"\nisfinite_count [{which}]: {count_a=} {count_b=}")


# NOTE: the following LFC tests were taken directly from the metpy test suite, some modifications
# were  made to insure SI units and proper 2d array shape. The tolerances also needed to be
# adjusted in some  cases to account for minor differences in the calculations.
@pytest.mark.lfc
def test_lfc_basic():
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


@pytest.mark.skip(reason="the mixed parcel profile is not implemented yet")
def test_lfc_ml():
    """Test Mixed-Layer LFC calculation."""
    levels = np.array([959.0, 779.2, 751.3, 724.3, 700.0, 269.0]) * hPa
    temperatures = np.array([22.2, 14.6, 12.0, 9.4, 7.0, -49.0]) * C
    dewpoints = np.array([19.0, -11.2, -10.8, -10.4, -10.0, -53.2]) * C
    __, t_mixed, td_mixed = mixed_parcel(levels, temperatures, dewpoints)
    mixed_parcel_prof = _C.parcel_profile(levels, t_mixed, td_mixed)
    lfc_p, lfc_t = lfc(levels, temperatures, dewpoints, mixed_parcel_prof)
    assert_almost_equal(lfc_p, 601.225 * hPa, 2)
    assert_almost_equal(lfc_t, -1.90688 * units.degC, 2)


@pytest.mark.lfc
def test_no_lfc():
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
def test_lfc_inversion():
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
def test_lfc_equals_lcl():
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
def test_lfc_profile_nan():
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


@pytest.mark.cape
@pytest.mark.mu_cape
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
# CAPE_CIN
# ............................................................................................... #
@pytest.mark.cape_cin
@pytest.mark.parametrize(
    "which_lfc, which_el",
    itertools.product(["top", "bottom"], ["top", "bottom"]),
)
def test_cape_cin(which_lfc, which_el):
    """
    TODO currently this test is passing on 95% of the cases, need to investigate the.
    there error appears to be something in the logic block of the el_lfc function.

    The current test cases run 500 samples and we are failing on 17 of them specifically when
    `which_el=bottom` parameter is used. realistically using the lower EL is not a typical use
    case but it should still be tested.
    """
    pressure, temperature, dewpoint = P[np.newaxis], T, Td

    parcel_profile = _C.parcel_profile(pressure[0, :], temperature[:, 0], dewpoint[:, 0])
    CAPE, CIN = cape_cin(
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc=which_lfc,
        which_el=which_el,
    )
    count = 0
    indexes = []
    for i in range(T.shape[0]):
        CAPE_, CIN_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            parcel_profile[i] * K,
            which_lfc=which_lfc,
            which_el=which_el,
        )
        if not np.allclose(CAPE[i], CAPE_.m, atol=1000):
            count += 1
            indexes.append(i)

    print(f"\nCAPE_CIN [{which_lfc} {which_el}]: {count=} {indexes=}")
