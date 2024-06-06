# noqa
import itertools
import json
from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.calc.thermo import _find_append_zero_crossings  # noqa: E402
from metpy.units import units
from numpy.testing import assert_allclose

import nzthermo._core as _C
import nzthermo._ufunc as uf  # noqa: E402
import nzthermo.functional as F  # noqa: E402
from nzthermo._ufunc import lcl_pressure
from nzthermo.core import cape_cin, ccl, dewpoint_from_specific_humidity, el, lfc

np.set_printoptions(
    precision=3,
    suppress=True,
    threshold=150,
    linewidth=150,
    edgeitems=10,
)

K = units.kelvin
Pa = units.pascal
FAST_APPROXIMATE = True
# somewhat surprisingly, the MRMS data is able to be resolved to 1e-5 short cutting a large
# number of conditionals that were needed to best mimic the metpy implementation.
# For grids of shape (40, 1059, 1799) the fast approximation method reduces the time to
# calculate the CAPE/CIN by 25% (~5 seconds).
if FAST_APPROXIMATE:
    RTOL = 1e-5
    with open("tests/data.json", "r") as f:
        data = json.load(f)
        _P = data["pressure"]
        _T = data["temperature"]
        _Q = data["specific_humidity"]

    P = np.array(_P, dtype=np.float64)  # * 100.0
    T = np.array(_T, dtype=np.float64)[::]
    Q = np.array(_Q, dtype=np.float64)[::]
    Td = dewpoint_from_specific_humidity(T, Q)

else:
    RTOL = 1e-1
    _P = [
        1013,
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
    ]

    _T = [
        [243, 242, 241, 240, 239, 237, 236, 235, 233, 232, 231, 229, 228, 226, 235, 236, 234, 231, 226, 221, 217, 211],
        [250, 249, 248, 247, 246, 244, 243, 242, 240, 239, 238, 236, 235, 233, 240, 239, 236, 232, 227, 223, 217, 211],
        [293, 292, 290, 288, 287, 285, 284, 282, 281, 279, 279, 280, 279, 278, 275, 270, 268, 264, 260, 254, 246, 237],
        [300, 299, 297, 295, 293, 291, 292, 291, 291, 289, 288, 286, 285, 285, 281, 278, 273, 268, 264, 258, 251, 242],
        # [299, 298, 296, 294, 292, 290, 289, 288, 287, 285, 284, 282, 281, 280, 276, 273, 268, 263, 258, 252, 245, 237],
    ]
    _Td = [
        [224, 224, 224, 224, 224, 223, 223, 223, 223, 222, 222, 222, 221, 221, 233, 233, 231, 228, 223, 218, 213, 207],
        [233, 233, 232, 232, 232, 232, 231, 231, 231, 231, 230, 230, 230, 229, 237, 236, 233, 229, 223, 219, 213, 207],
        [288, 288, 287, 286, 281, 280, 279, 277, 276, 275, 270, 258, 244, 247, 243, 254, 262, 248, 229, 232, 229, 224],
        [294, 294, 293, 292, 291, 289, 285, 282, 280, 280, 281, 281, 278, 274, 273, 269, 259, 246, 240, 241, 226, 219],
        # [298, 298, 297, 296, 295, 293, 291, 289, 287, 285, 284, 282, 280, 276, 273, 268, 263, 258, 252, 245, 237, 229],
    ]
    P = np.array(_P, dtype=np.float64) * 100.0
    T = np.array(_T, dtype=np.float64)
    Td = np.array(_Td, dtype=np.float64)


LCL = lcl_pressure(P[0], T[:, 0], Td[:, 0])
assert np.all(Td <= T)


def pressure_levels(sfc=1013.25, dtype: Any = np.float64):
    pressure = [sfc, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    pressure += [725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200]
    return np.array(pressure, dtype=dtype) * 100.0


@pytest.mark.ccl
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ccl(dtype) -> None:
    P = (
        ([993.0, 957.0, 925.0, 886.0, 850.0, 813.0, 798.0, 732.0, 716.0, 700.0] * units.hPa)
        .to("pascal")
        .m.astype(dtype)
    )
    T = ([34.6, 31.1, 27.8, 24.3, 21.4, 19.6, 18.7, 13, 13.5, 13] * units.degC).to("K").m.astype(dtype)
    Td = ([19.6, 18.7, 17.8, 16.3, 12.4, -0.4, -3.8, -6, -13.2, -11] * units.degC).to("K").m.astype(dtype)
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="bottom")

    def get_metpy_ccl(p, t, td):
        return [x.m for x in mpcalc.ccl(p * Pa, t * K, td * K, which="bottom")]

    assert_allclose(
        np.ravel((ccl_t, ccl_p, ct)),
        get_metpy_ccl(P, T, Td),
        rtol=1e-6,
    )
    P = np.array([P, P, P])
    T = np.array([T, T - 1, T])
    Td = np.array([Td, Td, Td - 0.5])
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="bottom")

    for i in range(len(P)):
        assert_allclose(
            (ccl_t[i], ccl_p[i], ct[i]),
            get_metpy_ccl(P[i], T[i], Td[i]),
            rtol=1e-4,
        )


def test_parcel_profile() -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])
    for i in range(T.shape[0]):
        prof_ = mpcalc.parcel_profile(
            P * Pa,
            T[i, 0] * K,
            Td[i, 0] * K,
        )
        assert_allclose(prof[i], prof_.m, rtol=1e-3)


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

        assert_allclose(el_p[i], el_p_.m, rtol=1e-3)
        assert_allclose(el_t[i], el_t_.m, rtol=1e-3)


# ............................................................................................... #
# LFC
# ............................................................................................... #
@pytest.mark.lfc
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_lfc(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    lfc_p, lfc_t = lfc(P, T, Td, prof, which, fast_approximate=FAST_APPROXIMATE)
    for i in range(T.shape[0]):
        lfc_p_, lfc_t_ = mpcalc.lfc(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which=which,
        )

        assert_allclose(lfc_p[i], lfc_p_.m, rtol=1e-5)  # type: ignore
        assert_allclose(lfc_t[i], lfc_t_.m, rtol=1e-5)


# ............................................................................................... #
# CAPE_CIN
# ............................................................................................... #
@pytest.mark.cape
@pytest.mark.parametrize("which_lfc, which_el", itertools.product(["top", "bottom"], ["top", "bottom"]))
def test_insert_zero_crossings(which_lfc, which_el):
    """This represents the first part of the cape_cin calculation"""
    pressure, temperature, dewpoint = P[np.newaxis], T, Td
    parcel_profile = _C.parcel_profile(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])
    lcl_p = uf.lcl_pressure(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])  # ✔️
    parcel_mixing_ratio = np.where(
        pressure > lcl_p[:, np.newaxis],  # below_lcl
        uf.saturation_mixing_ratio(pressure, dewpoint),
        uf.saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    temperature = uf.virtual_temperature(temperature, uf.saturation_mixing_ratio(pressure, dewpoint))
    parcel_profile = uf.virtual_temperature(parcel_profile, parcel_mixing_ratio)
    lfc_p = lfc(pressure, temperature, dewpoint, parcel_profile, which=which_lfc).pressure  # ✔️

    # Calculate the EL limit of integration
    el_p = el(pressure, temperature, dewpoint, parcel_profile, which=which_el).pressure  # ✔️
    lfc_p, el_p = np.reshape((lfc_p, el_p), (2, -1, 1))
    INPUT = np.broadcast_to(pressure, temperature.shape), parcel_profile - temperature

    X, Y = F.find_append_zero_crossings(INPUT[0], INPUT[1])  # ((N, Z), ...)

    for i in range(len(X)):
        x, y = X[i], Y[i]
        x_, y_ = _find_append_zero_crossings(INPUT[0][i] * Pa, INPUT[1][i] * K)
        x = x[: len(x_)]
        y = y[: len(x_)]
        assert_allclose(x, x_.m, rtol=1e-5)
        assert_allclose(y, y_.m, rtol=1e-5)


# top equilibrium level
@pytest.mark.cape
@pytest.mark.cape_el
@pytest.mark.cape_el_top
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_cape_cin_el_top(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    cape, cin = cape_cin(P, T, Td, prof, which_lfc=which, which_el="top", fast_approximate=FAST_APPROXIMATE)
    for i in range(T.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which_lfc=which,
            which_el="top",
        )
        print(cape[i], cape_.m)
        assert_allclose(cape[i], cape_.m, rtol=RTOL)  # type: ignore
        assert_allclose(cin[i], cin_.m, rtol=RTOL)


# bottom equilibrium level
@pytest.mark.cape
@pytest.mark.cape_el
@pytest.mark.cape_el_bottom
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_cape_cin_el_bottom(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    cape, cin = cape_cin(P, T, Td, prof, which_lfc=which, which_el="bottom", fast_approximate=FAST_APPROXIMATE)
    for i in range(T.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which_lfc=which,
            which_el="bottom",
        )

        assert_allclose(cape[i], cape_.m, rtol=RTOL)  # type: ignore
        assert_allclose(cin[i], cin_.m, rtol=RTOL)


# bottom lfc
@pytest.mark.cape
@pytest.mark.cape_lfc
@pytest.mark.cape_lfc_bottom
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_cape_cin_lfc_bottom(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    cape, cin = cape_cin(P, T, Td, prof, which_el=which, which_lfc="bottom", fast_approximate=FAST_APPROXIMATE)
    for i in range(T.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which_el=which,
            which_lfc="bottom",
        )

        assert_allclose(cape[i], cape_.m, rtol=RTOL)  # type: ignore
        assert_allclose(cin[i], cin_.m, rtol=RTOL)


# top lfc
@pytest.mark.cape
@pytest.mark.cape_lfc
@pytest.mark.cape_lfc_top
@pytest.mark.parametrize("which", ["top", "bottom"])
def test_cape_cin_lfc_top(which) -> None:
    prof = _C.parcel_profile(P, T[:, 0], Td[:, 0])

    cape, cin = cape_cin(P, T, Td, prof, which_el=which, which_lfc="top", fast_approximate=FAST_APPROXIMATE)
    for i in range(T.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            P * Pa,
            T[i] * K,
            Td[i] * K,
            prof[i] * K,
            which_el=which,
            which_lfc="top",
        )

        assert_allclose(cape[i], cape_.m, rtol=RTOL)  # type: ignore
        assert_allclose(cin[i], cin_.m, rtol=RTOL)
