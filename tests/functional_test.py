from __future__ import annotations

import metpy.calc.tools as mtools
import numpy as np
import pytest
from metpy.calc.thermo import _find_append_zero_crossings
from metpy.units import units
from numpy.testing import assert_allclose

import nzthermo.functional as F
from nzthermo._core import parcel_profile as _parcel_profile  # noqa
from nzthermo._ufunc import (  # noqa
    lcl_pressure,
    saturation_mixing_ratio,
    virtual_temperature,
)
from nzthermo.core import FASTPATH, el_lfc  # noqa

K = units.kelvin
Pa = units.pascal

data = np.load("tests/data.npz", allow_pickle=False)
step = np.s_[:]
P, T, Td = data["P"], data["T"][step], data["Td"][step]


@pytest.mark.parametrize(
    "x,a,b",
    [
        (
            [1013.0, 1000.0, 975.00, 950.00, 925.00, 900.00, 875.00, 850.00, 825.00, 800.00],
            [
                [303.30, 302.36, 300.16, 298.00, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
            ],  # a
            [
                [297.61, 297.36, 300.18, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
        ),
        (
            [1013.0, 1000.0, 975.00, 950.00, 925.00, 900.00, 875.00, 850.00, 825.00, 800.00],  # x
            [
                [303.30, 302.36, 300.16, 298.00, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
                [303.3, 302.36, 300.16, 298.0, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
                [303.58, 302.6, 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6],
                [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
                [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1],
                [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47],
                # [297.61, 300.36, 300.18, 300.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
            [
                [297.61, 297.36, 300.18, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
                [297.61, 297.36, 300.18, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
                [297.62, 297.36, 296.79, 296.18, 294.5, 292.07, 287.74, 286.67, 285.15, 284.02],
                [297.76, 297.51, 296.91, 296.23, 295.05, 292.9, 288.86, 287.12, 285.99, 283.98],
                [297.82, 297.56, 296.95, 296.23, 295.0, 292.47, 289.97, 288.45, 287.09, 285.17],
                [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31],
                # [297.62, 300.36, 300.18, 300.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
        ),
    ],
)
def test_intersect_nz_increasing(x, a, b) -> None:
    direction = "increasing"
    pressure_levels = np.array(x)  # (Z,)
    temperature = np.array(a)  # (N, Z)
    dewpoint = np.array(b)  # (N, Z)
    # TODO: update this test with the correct upper intersect values, there was an off by
    # one error in the determination of the upper index based on the metpy implementation.
    intersect = F.intersect_nz(
        pressure_levels,
        temperature,
        dewpoint,
        direction=direction,
        log_x=True,
    )
    bottom = intersect.bottom()

    for i in range(temperature.shape[0]):
        x_, y_ = mtools.find_intersections(
            pressure_levels * units.pascal,
            temperature[i] * units.kelvin,
            dewpoint[i] * units.kelvin,
            direction,
            log_x=True,
        )

        assert_allclose(bottom.pressure[i], x_.m)
        assert_allclose(bottom.temperature[i], y_.m)


@pytest.mark.parametrize(
    "x,a,b",
    [
        (
            [1013.0, 1000.0, 975.00, 950.00, 925.00, 900.00, 875.00, 850.00, 825.00, 800.00],
            [
                [303.30, 302.36, 300.16, 298.00, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
            ],  # a
            [
                [297.61, 297.36, 300.18, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
        ),
        (
            [1013.0, 1000.0, 975.00, 950.00, 925.00, 900.00, 875.00, 850.00, 825.00, 800.00],  # x
            [
                [303.3, 302.36, 300.16, 298.0, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
                [303.58, 302.6, 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6],
                [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
                [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1],
                [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47],
                [297.61, 300.36, 300.18, 300.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
            [
                [297.61, 297.36, 300.18, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
                [297.62, 297.36, 296.79, 296.18, 294.5, 292.07, 287.74, 286.67, 285.15, 284.02],
                [297.76, 297.51, 296.91, 296.23, 295.05, 292.9, 288.86, 287.12, 285.99, 283.98],
                [297.82, 297.56, 296.95, 296.23, 295.0, 292.47, 289.97, 288.45, 287.09, 285.17],
                [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31],
                [297.62, 300.36, 300.18, 300.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            ],  # b
        ),
    ],
)
def test_intersect_nz_decreasing(x, a, b) -> None:
    direction = "decreasing"
    pressure_levels = np.array(x)  # (Z,)
    temperature = np.array(a)  # (N, Z)
    dewpoint = np.array(b)  # (N, Z)
    # TODO: update this test with the correct upper intersect values, there was an off by
    # one error in the determination of the upper index based on the metpy implementation.
    intersect = F.intersect_nz(
        pressure_levels, temperature, dewpoint, direction=direction, log_x=True
    )

    bottom = intersect.bottom()

    for i in range(temperature.shape[0]):
        x_, y_ = mtools.find_intersections(
            pressure_levels * units.pascal,
            temperature[i] * units.kelvin,
            dewpoint[i] * units.kelvin,
            direction,
            log_x=True,
        )

        assert_allclose(bottom.pressure[i], x_.m, rtol=1e-3)
        assert_allclose(bottom.temperature[i], y_.m, rtol=1e-3)


@pytest.mark.parametrize(
    "direction, expected",
    [
        ("increasing", [[[24.44, np.nan]], [[1794.53, np.nan]]]),
        ("decreasing", [[[8.89, np.nan]], [[238.84, np.nan]]]),
    ],
)
def test_find_intersections(direction, expected):
    """Test finding the intersection of two curves functionality."""
    x = np.linspace(5, 30, 17)
    y1 = 3 * x**2
    y2 = 100 * x - 650
    # Note: Truth is what we will get with this sampling, not the mathematical intersection
    x_int, y_int = F.intersect_nz(x, y1, y2, direction=direction)
    assert_allclose(x_int, expected[0], rtol=1e-2)
    assert_allclose(y_int, expected[1], rtol=1e-2)


X = [
    [
        101300.0,
        100000.0,
        97500.0,
        95000.0,
        92500.0,
        90000.0,
        87500.0,
        85000.0,
        82500.0,
        80000.0,
        77500.0,
        75000.0,
        72500.0,
        70000.0,
        65000.0,
        60000.0,
        55000.0,
        50000.0,
        45000.0,
        40000.0,
        35000.0,
        30000.0,
    ],
    [
        101300.0,
        100000.0,
        97500.0,
        95000.0,
        92500.0,
        90000.0,
        87500.0,
        85000.0,
        82500.0,
        80000.0,
        77500.0,
        75000.0,
        72500.0,
        70000.0,
        65000.0,
        60000.0,
        55000.0,
        50000.0,
        45000.0,
        40000.0,
        35000.0,
        30000.0,
    ],
    [
        101300.0,
        100000.0,
        97500.0,
        95000.0,
        92500.0,
        90000.0,
        87500.0,
        85000.0,
        82500.0,
        80000.0,
        77500.0,
        75000.0,
        72500.0,
        70000.0,
        65000.0,
        60000.0,
        55000.0,
        50000.0,
        45000.0,
        40000.0,
        35000.0,
        30000.0,
    ],
    [
        101300.0,
        100000.0,
        97500.0,
        95000.0,
        92500.0,
        90000.0,
        87500.0,
        85000.0,
        82500.0,
        80000.0,
        77500.0,
        75000.0,
        72500.0,
        70000.0,
        65000.0,
        60000.0,
        55000.0,
        50000.0,
        45000.0,
        40000.0,
        35000.0,
        30000.0,
    ],
]
Y = [
    [
        0.0,
        0.10490034640355361,
        -0.640105270326103,
        -1.4173715943893797,
        -2.2283682108609923,
        -2.0746375692942536,
        -2.957946557104748,
        -3.880094592235338,
        -3.843032758266702,
        -4.848946189466773,
        -5.900162810857239,
        -5.999177292107561,
        -7.1389520224321075,
        -7.325149527591435,
        -20.87840169212663,
        -26.697389772706345,
        -29.825206686214926,
        -32.303983155797624,
        -33.19347616248663,
        -34.572819205896195,
        -37.550460619437445,
        -39.282830179520204,
    ],
    [
        0.0,
        0.07911830677912235,
        -0.7161864792301174,
        -1.544683663942294,
        -2.4078860182064545,
        -2.3073472132873576,
        -3.2449309019359305,
        -4.222492679966194,
        -4.241984079278325,
        -5.305748249794419,
        -6.416066088328506,
        -6.542190388438371,
        -7.711014316772179,
        -7.942640728217526,
        -19.595796124717083,
        -23.53889891479386,
        -25.80450245051111,
        -27.436859469938838,
        -28.494167217249498,
        -31.059915100442197,
        -32.2441184858487,
        -34.2047538741576,
    ],
    [
        0.0,
        -0.0797889474139879,
        -0.18444716911261594,
        -0.3281836293301126,
        -0.19757510081490182,
        0.5717704440022544,
        0.41293754490976653,
        1.1856710910926154,
        0.9324674184024957,
        1.5405620440196799,
        0.48573331460755753,
        -1.4067978075917722,
        -1.7617025648233948,
        -2.407970206437369,
        -2.955019599172317,
        -2.212123757020777,
        -4.876724067576163,
        -5.720884901348313,
        -7.554888789734804,
        -8.595547153076609,
        -8.83767415110151,
        -9.467990707876623,
    ],
    [
        0.0,
        -0.106052173951241,
        -0.2617172730227253,
        -0.45745672115992875,
        -0.20493893151274278,
        0.7988380787404594,
        -0.5239867486703247,
        -0.38997422751697286,
        -1.2570524531375327,
        -0.6820032346079188,
        -1.0876665082764134,
        -0.5866252781567596,
        -0.6983051651785104,
        -1.6952311226601182,
        -0.9793486917313317,
        -1.2357861848597054,
        0.14699283094955717,
        0.9278433580207093,
        -0.0718859454202061,
        -0.1274059729805117,
        -0.3182393979348035,
        -0.24344423360349765,
    ],
]


def test_insert_zero_crossings() -> None:
    crossings = F.zero_crossings(np.array(X), np.array(Y))
    for i in range(len(X)):
        x_, y_ = _find_append_zero_crossings(X[i] * Pa, Y[i] * K)
        x, y = crossings[0][i], crossings[1][i]
        x = x[: len(x_)]
        y = y[: len(y_)]
        assert_allclose(x, x_.m)
        assert_allclose(y, y_.m)


def test_insert_zero_crossings_specifically_for_cape_cin() -> None:
    """The top half of this test cases replicates a partial implementation of our cape_cin function
    which is currently failing under similar test conditions.  However, the test asserts that
    the zero_crossing function is not the source of the error."""
    pressure = P[np.newaxis, :]
    temperature = T[:, :]
    dewpoint = Td[:, :]
    lcl_p = lcl_pressure(pressure[:, 0], temperature[:, 0], dewpoint[:, 0])  # ✔️
    parcel_profile = _parcel_profile(pressure.squeeze(), temperature[:, 0], dewpoint[:, 0])  # ✔️

    # The mixing ratio of the parcel comes from the dewpoint below the LCL, is saturated
    # # based on the temperature above the LCL
    parcel_mixing_ratio = np.where(
        pressure > lcl_p[:, np.newaxis],  # below_lcl
        saturation_mixing_ratio(pressure, dewpoint),
        saturation_mixing_ratio(pressure, temperature),
    )
    # Convert the temperature/parcel profile to virtual temperature
    temperature = virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint))
    parcel_profile = virtual_temperature(parcel_profile, parcel_mixing_ratio)
    # Calculate the EL limit of integration
    (el_p, _), (lfc_p, _) = el_lfc(
        pressure,
        temperature,
        dewpoint,
        parcel_profile,
        which_lfc="bottom",
        which_el="top",
        **FASTPATH,
    )

    lfc_p, el_p = np.reshape((lfc_p, el_p), (2, -1, 1))  # reshape for broadcasting
    delta = parcel_profile - temperature
    X, Y = F.zero_crossings(pressure, delta)  # ((N, Z), ...)

    for i in range(temperature.shape[0]):
        x_, y_ = _find_append_zero_crossings(
            pressure[0] * Pa,
            delta[i] * K,
        )

        assert_allclose(
            X[i][~np.isnan(X[i])],
            x_.m,
        )

        assert_allclose(
            Y[i][~np.isnan(Y[i])],
            y_.m,
        )
