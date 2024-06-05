from typing import Any
import nzthermo._core as _C
from nzthermo.core import lfc

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose
from nzthermo.core import ccl, el, lcl, parcel_profile, interpolate_nz, cape_cin

np.set_printoptions(
    precision=3,
    suppress=True,
    threshold=150,
    linewidth=150,
    edgeitems=10,
)
_P = [1013, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300]

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
PRESSURE = np.array(_P, dtype=np.float64) * 100.0
TEMPERATURE = np.array(_T, dtype=np.float64)
DEWPOINT = np.array(_Td, dtype=np.float64)
P, T, Td = PRESSURE, TEMPERATURE, DEWPOINT


def pressure_levels(sfc=1013.25, dtype: Any = np.float64):
    pressure = [sfc, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750]
    pressure += [725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200]
    return np.array(pressure, dtype=dtype) * 100.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_ccl(dtype) -> None:
    P = (
        ([993.0, 957.0, 925.0, 886.0, 850.0, 813.0, 798.0, 732.0, 716.0, 700.0] * units.hPa)
        .to("pascal")
        .m.astype(dtype)
    )
    T = ([34.6, 31.1, 27.8, 24.3, 21.4, 19.6, 18.7, 13, 13.5, 13] * units.degC).to("kelvin").m.astype(dtype)
    Td = ([19.6, 18.7, 17.8, 16.3, 12.4, -0.4, -3.8, -6, -13.2, -11] * units.degC).to("kelvin").m.astype(dtype)
    ccl_t, ccl_p, ct = ccl(P, T, Td, which="bottom")

    def get_metpy_ccl(p, t, td):
        return [x.m for x in mpcalc.ccl(p * units.pascal, t * units.kelvin, td * units.kelvin, which="bottom")]

    assert_allclose(
        np.ravel((ccl_t, ccl_p, ct)),
        get_metpy_ccl(P, T, Td),
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


def test_parcel_profile_with_lcl() -> None:
    P, T, Td, Tp = parcel_profile(PRESSURE, TEMPERATURE, DEWPOINT).with_lcl()
    for i in range(TEMPERATURE.shape[0]):
        metpy_pp = mpcalc.parcel_profile_with_lcl(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.degK,
            DEWPOINT[i] * units.degK,
        )
        P_, T_, Td_, Tp_ = (x.m for x in metpy_pp)
        assert_allclose(P[i], P_, rtol=1e-3)
        assert_allclose(T[i], T_, rtol=1e-4)
        assert_allclose(Td[i], Td_, rtol=1e-4)
        assert_allclose(Tp[i], Tp_, rtol=1e-4)


@pytest.mark.parametrize("which", ["top", "bottom"])
def test_el(which) -> None:
    prof = _C.parcel_profile(PRESSURE, TEMPERATURE[:, 0], DEWPOINT[:, 0])
    el_p, el_t = el(PRESSURE, TEMPERATURE, DEWPOINT, prof, which=which)
    for i in range(TEMPERATURE.shape[0]):
        el_p_, el_t_ = mpcalc.el(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.degK,
            DEWPOINT[i] * units.degK,
            prof[i] * units.degK,
            which=which,
        )

        assert_allclose(el_p[i], el_p_.m, rtol=1e-1)
        assert_allclose(el_t[i], el_t_.m, rtol=1e-1)


@pytest.mark.parametrize("which", ["top", "bottom"])
def test_lfc(which) -> None:
    pp = parcel_profile(PRESSURE, TEMPERATURE, DEWPOINT)

    lfc_p, lfc_t = pp.lfc(which)
    for i in range(TEMPERATURE.shape[0]):
        lfc_p_, lfc_t_ = mpcalc.lfc(
            PRESSURE * units.pascal, TEMPERATURE[i] * units.kelvin, DEWPOINT[i] * units.kelvin, which=which
        )
        np.testing.assert_allclose(lfc_p[i], lfc_p_.m, rtol=1e-3)  # type: ignore
        np.testing.assert_allclose(lfc_t[i], lfc_t_.m, rtol=1e-3)


@pytest.mark.parametrize("which", ["top", "bottom"])
def test_lfc_new(which) -> None:
    prof = _C.parcel_profile(PRESSURE, TEMPERATURE[:, 0], DEWPOINT[:, 0])

    lfc_p, lfc_t = lfc(PRESSURE, TEMPERATURE, DEWPOINT, prof, which)
    for i in range(TEMPERATURE.shape[0]):
        lfc_p_, lfc_t_ = mpcalc.lfc(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.kelvin,
            DEWPOINT[i] * units.kelvin,
            prof[i] * units.kelvin,
            which=which,
        )
        np.testing.assert_allclose(lfc_p[i], lfc_p_.m, rtol=1e-3)  # type: ignore
        np.testing.assert_allclose(lfc_t[i], lfc_t_.m, rtol=1e-3)


@pytest.mark.parametrize("which", ["top", "bottom"])
def test_cape_cin(which) -> None:
    prof = _C.parcel_profile(PRESSURE, TEMPERATURE[:, 0], DEWPOINT[:, 0])
    # top equilibrium level
    cape, cin = cape_cin(PRESSURE, TEMPERATURE, DEWPOINT, prof, which_lfc=which, which_el="top")
    for i in range(TEMPERATURE.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.kelvin,
            DEWPOINT[i] * units.kelvin,
            prof[i] * units.kelvin,
            which_lfc=which,
            which_el="top",
        )
        print("CAPE", cape[i], cape_.m, cape[i] - cape_.m)
        np.testing.assert_allclose(cape[i], cape_.m, rtol=1)  # type: ignore
        print("CIN", cin[i], cin_.m)
        np.testing.assert_allclose(cin[i], cin_.m, rtol=1)

    # bottom equilibrium level
    cape, cin = cape_cin(PRESSURE, TEMPERATURE, DEWPOINT, prof, which_lfc=which, which_el="bottom")
    for i in range(TEMPERATURE.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.kelvin,
            DEWPOINT[i] * units.kelvin,
            prof[i] * units.kelvin,
            which_lfc=which,
            which_el="bottom",
        )
        print("CAPE", cape[i], cape_.m, cape[i] - cape_.m)
        np.testing.assert_allclose(cape[i], cape_.m, rtol=1)  # type: ignore
        print("CIN", cin[i], cin_.m)
        np.testing.assert_allclose(cin[i], cin_.m, rtol=1)

    # bottom lfc
    cape, cin = cape_cin(PRESSURE, TEMPERATURE, DEWPOINT, prof, which_el=which, which_lfc="bottom")
    for i in range(TEMPERATURE.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.kelvin,
            DEWPOINT[i] * units.kelvin,
            prof[i] * units.kelvin,
            which_el=which,
            which_lfc="bottom",
        )
        print("CAPE", cape[i], cape_.m, cape[i] - cape_.m)
        np.testing.assert_allclose(cape[i], cape_.m, rtol=1)  # type: ignore
        print("CIN", cin[i], cin_.m)
        np.testing.assert_allclose(cin[i], cin_.m, rtol=1)

    # top lfc
    cape, cin = cape_cin(PRESSURE, TEMPERATURE, DEWPOINT, prof, which_el=which, which_lfc="top")
    for i in range(TEMPERATURE.shape[0]):
        cape_, cin_ = mpcalc.cape_cin(
            PRESSURE * units.pascal,
            TEMPERATURE[i] * units.kelvin,
            DEWPOINT[i] * units.kelvin,
            prof[i] * units.kelvin,
            which_el=which,
            which_lfc="top",
        )
        print("CAPE", cape[i], cape_.m, cape[i] - cape_.m)
        np.testing.assert_allclose(cape[i], cape_.m, rtol=1)  # type: ignore
        print("CIN", cin[i], cin_.m)
        np.testing.assert_allclose(cin[i], cin_.m, rtol=1)


def test_interpolate_nz() -> None:
    lcl_p = np.array([93290.11, 92921.01, 92891.83, 93356.17, 94216.14])  # (N,)
    pressure_levels = np.array(
        [101300.0, 100000.0, 97500.0, 95000.0, 92500.0, 90000.0, 87500.0, 85000.0, 82500.0, 80000.0],
    )  # (Z,)
    temperature = np.array(
        [
            [303.3, 302.36, 300.16, 298.0, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
            [303.58, 302.6, 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6],
            [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
            [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1],
            [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47],
        ],
    )  # (N, Z)
    dewpoint = np.array(
        [
            [297.61, 297.36, 296.73, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            [297.62, 297.36, 296.79, 296.18, 294.5, 292.07, 287.74, 286.67, 285.15, 284.02],
            [297.76, 297.51, 296.91, 296.23, 295.05, 292.9, 288.86, 287.12, 285.99, 283.98],
            [297.82, 297.56, 296.95, 296.23, 295.0, 292.47, 289.97, 288.45, 287.09, 285.17],
            [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31],
        ],
    )  # (N, Z)
    values = interpolate_nz(  # ((N,), ...)
        lcl_p, pressure_levels, temperature, dewpoint
    )  # temp & dwpt values interpolated at LCL pressure

    assert_allclose(
        values,
        [[296.69, 296.78, 296.68, 296.97, 297.44], [295.12, 294.78, 295.23, 295.42, 296.22]],
        atol=1e-2,
    )
