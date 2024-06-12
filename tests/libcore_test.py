import json

import numpy as np
from metpy.calc.tools import interpolate_1d
from numpy.testing import assert_allclose

from nzthermo._core import interpolate_nz
from nzthermo._ufunc import dewpoint_from_specific_humidity, lcl_pressure

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
T = np.array(_T, dtype=np.float64)
Q = np.array(_Q, dtype=np.float64)
Td = dewpoint_from_specific_humidity(T, Q)
LCL = lcl_pressure(P[0], T[:, 0], Td[:, 0])


def test_interpolate_nz() -> None:
    t, td = interpolate_nz(LCL, P, T, Td)
    for i in range(t.shape[0]):
        t_, td_ = interpolate_1d(LCL[i], P, T[i], Td[i])
        assert_allclose(t[i], t_, rtol=1e-5)
        assert_allclose(td[i], td_, rtol=1e-5)
