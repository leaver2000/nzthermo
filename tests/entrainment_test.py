from __future__ import annotations

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo._ufunc import pressure_vector
from nzthermo.core import specific_humidity
from nzthermo.entrainment import bunkers_storm_motion, entrainment

from .source.ecape import calc_lfc_height, calc_mse

Pa = units.pascal
K = units.kelvin
m = units.meters
dimensionsless = units.dimensionless

data = np.load("tests/data.npz", allow_pickle=False)

step = np.s_[:]

P: pressure_vector = data["P"].view(pressure_vector)
T: np.ndarray = data["T"][step]
Td: np.ndarray = data["Td"][step]
_super_saturation = Td > T
Td[_super_saturation] = T[_super_saturation]

Q = mpcalc.specific_humidity_from_dewpoint(P * Pa, Td * K).to(dimensionsless).m
assert_allclose(Q, specific_humidity(P, Td), atol=1e-4)  # .to(dimensionsless).m
U = data["U"][step]
V = data["V"][step]
print(np.abs(Q - data["Q"][step]).max())

MSL: np.ndarray = data["Z"][step]

assert P[0] > P[-1]
assert np.all(T[:, 0] > T[:, -1])

print(f"P: {P.shape}, T: {T.shape}, Td: {Td.shape}")
# In very rare cases the data accessed from the HRRR model had dewpoint temperatures greater than
# the actual temperature. This is not physically possible and is likely due to rounding errors.
# This also makes testing quite difficult because in many cases metpy will report a nan values
# and throw interpolation warnings. To avoid this we will set the dewpoint temperature to be less
# than the actual temperature.
# MSL = mpcalc.pressure_to_height_std(P * Pa).to("m").m

entraining = entrainment(P, MSL, T, U, V, specific_humidity=Q)


@pytest.mark.entrainment
def test_entrainment_moist_static_energy() -> None:
    BAR, STAR = entraining.moist_static_energy

    for i in range(T.shape[0]):
        BAR_, STAR_ = calc_mse(P * Pa, MSL[i] * m, T[i] * K, Q[i] * dimensionsless)
        assert_allclose(BAR[i], BAR_.to("J/kg").m, atol=1)
        assert_allclose(STAR[i], STAR_.to("J/kg").m, atol=5)


@pytest.mark.entrainment
def test_entrainment_lfc_index() -> None:
    idx = entraining.lfc_index
    z = entraining.lfc_height

    offby_1 = 0
    for i in range(T.shape[0]):
        assert offby_1 <= 10

        idx_, z_ = calc_lfc_height(P * Pa, MSL[i] * m, T[i] * K, Td[i] * K, None)
        # TODO: in some circumstances there is no LFC which our target code didnt really account for
        # so we need to skip any cases in the target code that produce a nan value. because this breaks
        # an index operation they had implmented. Also we can short circuit some of our own code with
        # by adding a where argument to the function call.
        if np.isnan(idx_):
            continue
        elif np.abs(idx[i] - idx_) >= 1:
            offby_1 += 1
            continue
        assert idx[i] == idx_
        assert z[i] == z_.m


@pytest.mark.entrainment
def test_bunkers_storm_motion():
    pressure, u, v, height = P, U, V, MSL
    right_mover, left_mover, wind_mean = bunkers_storm_motion(pressure, u, v, height)

    for i in range(T.shape[0]):
        p, u, v, msl, _ = (P * Pa, U[i] * units("m/s"), V[i] * units("m/s"), MSL[i] * m, 500 * m)

        right_mover_, left_mover_, wind_mean_ = mpcalc.bunkers_storm_motion(
            p,
            u,
            v,
            msl,
        )
        assert right_mover_.units == "meter / second"
        assert left_mover_.units == "meter / second"
        assert wind_mean_.units == "meter / second"
        assert_allclose(
            wind_mean[:, i],
            wind_mean_.m,
            atol=1.5,
        )

        assert_allclose(
            right_mover[:, i],
            right_mover_.m,
            atol=5,
        )
        assert_allclose(
            left_mover[:, i],
            left_mover_.m,
            atol=5,
        )


@pytest.mark.entrainment
def test_storm_relative_motion():
    entraining.storm_relative_motion()
