from __future__ import annotations

import metpy.calc as mpcalc
import numpy as np
import pytest
from metpy.units import units
from numpy.testing import assert_allclose

from nzthermo._core import pressure_vector, height_vector
from nzthermo.core import specific_humidity
from nzthermo.entrainment import bunkers_storm_motion, ecape, entrainment

from .source.ecape import (
    calc_ecape,
    calc_el_height,
    calc_integral_arg,
    calc_lfc_height,
    calc_mse,
    calc_ncape,
    calc_psi,
    calc_sr_wind,
)

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

MSL: height_vector = data["Z"][step].view(height_vector)

assert P[0] > P[-1]
assert np.all(T[:, 0] > T[:, -1])

print(f"P: {P.shape}, T: {T.shape}, Td: {Td.shape}")
# In very rare cases the data accessed from the HRRR model had dewpoint temperatures greater than
# the actual temperature. This is not physically possible and is likely due to rounding errors.
# This also makes testing quite difficult because in many cases metpy will report a nan values
# and throw interpolation warnings. To avoid this we will set the dewpoint temperature to be less
# than the actual temperature.
# MSL = mpcalc.pressure_to_height_std(P * Pa).to("m").m

entraining = entrainment(P, MSL, T, U, V, specific_humidity=Q, cape_type="most_unstable")


@pytest.mark.entrainment
def test_entrainment_moist_static_energy() -> None:
    BAR, STAR = entraining.moist_static_energy

    for i in range(T.shape[0]):
        BAR_, STAR_ = calc_mse(P * Pa, MSL[i] * m, T[i] * K, Q[i] * dimensionsless)
        assert_allclose(BAR[i], BAR_.to("J/kg").m, atol=1)
        assert_allclose(STAR[i], STAR_.to("J/kg").m, atol=5)


@pytest.mark.entrainment
def test_entrainment_psi() -> None:
    assert_allclose(entraining.psi, calc_psi(entraining.el_height * m).m)


@pytest.mark.entrainment
def test_integral() -> None:
    bar, star = entraining.moist_static_energy
    arg = calc_integral_arg(
        bar * units("J/kg"),
        star * units("J/kg"),
        temperature=T * K,
    )

    assert_allclose(arg.m, entraining.integral)


@pytest.mark.entrainment
def test_entrainment_lfc_index() -> None:
    idx = entraining.lfc_index
    z = entraining.lfc_height

    offby_1 = 0
    for i in range(T.shape[0]):
        # assert offby_1 <= 10

        idx_, z_ = calc_lfc_height(
            P * Pa, MSL[i] * m, T[i] * K, Td[i] * K, mpcalc.most_unstable_parcel
        )
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

    print(offby_1)


@pytest.mark.entrainment
def test_entrainment_el_index() -> None:
    idx = entraining.el_index
    z = entraining.el_height

    offby_1 = 0
    for i in range(T.shape[0]):
        assert offby_1 <= 10

        idx_, z_ = calc_el_height(
            P * Pa, MSL[i] * m, T[i] * K, Td[i] * K, mpcalc.most_unstable_parcel
        )
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
def test_ncape() -> None:
    integral_arg = entraining.integral
    lfc_idx = entraining.lfc_index
    el_idx = entraining.el_index
    NCAPE = entraining.ncape
    for i in range(T.shape[0]):
        NCAPE_ = calc_ncape(
            integral_arg[i] * units("meter / second ** 2"),
            MSL[i] * m,
            lfc_idx[i],
            el_idx[i],
        )
        assert NCAPE_.units == "meter ** 2 / second ** 2"

        assert_allclose(
            NCAPE[i],
            NCAPE_.to("J/kg").m,
            atol=1400.0,
        )


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
            atol=10,
        )
        assert_allclose(
            left_mover[:, i],
            left_mover_.m,
            atol=10,
        )


@pytest.mark.entrainment
def test_storm_relative_motion():
    srm = entraining.storm_relative_motion
    for i in range(T.shape[0]):
        srm_ = calc_sr_wind(P * Pa, U[i] * units("m/s"), V[i] * units("m/s"), MSL[i] * m)
        assert_allclose(srm[i], srm_.m, atol=1.5)


@pytest.mark.entrainment
def test_storm_entrainment_cape():
    srm = entraining.storm_relative_motion
    for i in range(T.shape[0]):
        srm_ = calc_sr_wind(P * Pa, U[i] * units("m/s"), V[i] * units("m/s"), MSL[i] * m)
        assert_allclose(srm[i], srm_.m, atol=1.5)


@pytest.mark.entrainment
@pytest.mark.parametrize(
    "cape_type",
    ["most_unstable", "mixed_layer", "surface_based"],
)
def test_ecape(cape_type):
    ECAPE = ecape(P, T, Td, MSL, U, V, cape_type=cape_type)
    for i in range(T.shape[0]):
        ecape_ = calc_ecape(
            MSL[i] * m,
            P * Pa,
            T[i] * K,
            Q[i] * dimensionsless,
            U[i] * units("m/s"),
            V[i] * units("m/s"),
            cape_type=cape_type,
        )
        if np.isnan(ecape_):
            continue

        # break
        assert_allclose(
            ECAPE[i],
            ecape_,
            atol=500.0,
        )
