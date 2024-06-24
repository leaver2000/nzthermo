from __future__ import annotations

import metpy.calc as mpcalc
import numpy as np
import nzthermo.entrainment as entrainment
from metpy.units import units
from numpy.testing import assert_allclose

from .source.ecape import calc_mse

assert_allclose
Pa = units.pascal
K = units.kelvin
m = units.meters
dimensionsless = units("g/g")
print(dimensionsless)
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
Q = mpcalc.specific_humidity_from_dewpoint(P * Pa, Td * K).to(dimensionsless).m
msl = mpcalc.pressure_to_height_std(P * Pa)[np.newaxis, :].to("m").m


def test_entrainment_mse():
    BAR, STAR = entrainment.moist_static_energy(P, msl, T, Q)
    for i in range(T.shape[0]):
        BAR_, STAR_ = calc_mse(P * Pa, msl * m, T[i] * K, Q[i] * dimensionsless)
        print(BAR[i], BAR_.m)

        assert_allclose(BAR[i, -1], BAR_[-1].m, atol=1)
        break
        break
