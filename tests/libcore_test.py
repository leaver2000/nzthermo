import json

import metpy.calc as mpcalc
import numpy as np
from metpy.calc.tools import interpolate_1d
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo._core import interpolate_nz, parcel_profile, parcel_profile_with_lcl
from nzthermo._ufunc import dewpoint_from_specific_humidity, lcl_pressure

Pa = units.pascal
K = units.kelvin
np.set_printoptions(
    precision=3,
    suppress=True,
    threshold=150,
    linewidth=150,
    edgeitems=10,
)


with open("tests/data.json", "r") as f:
    data = json.load(f)
    _P = data["pressure"]
    _T = data["temperature"]
    _Q = data["specific_humidity"]


P = np.array(_P, dtype=np.float64)
T = np.array(_T, dtype=np.float64)[::10]
Q = np.array(_Q, dtype=np.float64)[::10]
Td = dewpoint_from_specific_humidity(T, Q)
LCL = lcl_pressure(P[0], T[:, 0], Td[:, 0])


def test_interpolate_nz() -> None:
    t, td = interpolate_nz(LCL, P, T, Td)
    for i in range(t.shape[0]):
        t_, td_ = interpolate_1d(LCL[i], P, T[i], Td[i])
        assert_allclose(t[i], t_, rtol=1e-5)
        assert_allclose(td[i], td_, rtol=1e-5)


def test_parcel_profile() -> None:
    pp = parcel_profile(P, T[:, 0], Td[:, 0])
    for i in range(pp.shape[0]):
        pp_ = mpcalc.parcel_profile(
            P * Pa,
            T[i, 0] * K,
            Td[i, 0] * K,
        ).m

        assert_allclose(pp[i], pp_, rtol=1e-3)


def test_parcel_profile_with_lcl() -> None:
    ep, et, etd, ptp = parcel_profile_with_lcl(P, T[:, :], Td[:, :])
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
