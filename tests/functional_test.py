from __future__ import annotations

import numpy as np
import nzthermo.functional as F
from numpy.testing import assert_allclose
import metpy.calc.tools as mtools
from metpy.units import units
import pytest

PRESSURE = np.array(
    [1013, 1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300],
    dtype=np.float64,
)
PRESSURE *= 100.0
TEMPERATURE = np.array(
    [
        [243, 242, 241, 240, 239, 237, 236, 235, 233, 232, 231, 229, 228, 226, 235, 236, 234, 231, 226, 221, 217, 211],
        [250, 249, 248, 247, 246, 244, 243, 242, 240, 239, 238, 236, 235, 233, 240, 239, 236, 232, 227, 223, 217, 211],
        [293, 292, 290, 288, 287, 285, 284, 282, 281, 279, 279, 280, 279, 278, 275, 270, 268, 264, 260, 254, 246, 237],
        [300, 299, 297, 295, 293, 291, 292, 291, 291, 289, 288, 286, 285, 285, 281, 278, 273, 268, 264, 258, 251, 242],
    ],
    dtype=np.float64,
)
DEWPOINT = np.array(
    [
        [224, 224, 224, 224, 224, 223, 223, 223, 223, 222, 222, 222, 221, 221, 233, 233, 231, 228, 223, 218, 213, 207],
        [233, 233, 232, 232, 232, 232, 231, 231, 231, 231, 230, 230, 230, 229, 237, 236, 233, 229, 223, 219, 213, 207],
        [288, 288, 287, 286, 281, 280, 279, 277, 276, 275, 270, 258, 244, 247, 243, 254, 262, 248, 229, 232, 229, 224],
        [294, 294, 293, 292, 291, 289, 285, 282, 280, 280, 281, 281, 278, 274, 273, 269, 259, 246, 240, 241, 226, 219],
    ],
    dtype=np.float64,
)


# def test_zero_crossing():
#     self = nzt.parcel_profile(PRESSURE, TEMPERATURE, DEWPOINT)
#     pressure, temperature, dewpoint, parcel_profile = self.with_lcl()
#     y = parcel_profile - temperature
#     x, y = F.zero_crossing(pressure.copy(), y)  # (N, Z)

#     for i in range(y.shape[0]):
#         x_, y_ = _find_append_zero_crossings(
#             pressure[i] * units.pascal, (parcel_profile - temperature)[i] * units.kelvin
#         )

#         assert_allclose(
#             x[i][~np.isnan(x[i])],
#             x_.m[:],  # , rtol=1e-3
#             # sep='\n'
#         )

#         assert_allclose(
#             y[i][~np.isnan(x[i])],
#             y_.m[:],  # , rtol=1e-3
#             # sep='\n'
#         )


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
    intersect = F.intersect_nz(pressure_levels, temperature, dewpoint, direction=direction, log_x=True, mask_nans=True)
    bottom = intersect.bottom()
    top = intersect.bottom()
    for i in range(temperature.shape[0]):
        x_, y_ = mtools.find_intersections(
            pressure_levels * units.pascal,
            temperature[i] * units.kelvin,
            dewpoint[i] * units.kelvin,
            direction,
            log_x=True,
        )

        if x_.size != 0:
            assert_allclose(bottom.x[i], x_.m[0], rtol=1e-3)
            assert_allclose(bottom.y[i], y_.m[0], rtol=1e-3)
            assert_allclose(top.x[i], x_.m[-1], rtol=1e-3)
            assert_allclose(top.y[i], y_.m[-1], rtol=1e-3)
        else:
            assert np.isnan(bottom.x[i])
            assert np.isnan(bottom.y[i])
            assert np.isnan(top.x[i])
            assert np.isnan(top.y[i])


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
    # one error in the determination of the upper index based on the metpy implmentation.
    intersect = F.intersect_nz(pressure_levels, temperature, dewpoint, direction=direction, log_x=True, mask_nans=True)

    bottom = intersect.bottom()
    top = intersect.bottom()

    for i in range(temperature.shape[0]):
        x_, y_ = mtools.find_intersections(
            pressure_levels * units.pascal,
            temperature[i] * units.kelvin,
            dewpoint[i] * units.kelvin,
            direction,
            log_x=True,
        )
        if x_.size != 0:
            assert_allclose(bottom.x[i], x_[0], rtol=1e-3)
            assert_allclose(bottom.y[i], y_[0], rtol=1e-3)
            assert_allclose(top.x[i], x_[-1], rtol=1e-3)
            assert_allclose(top.y[i], y_[-1], rtol=1e-3)
        else:
            assert np.isnan(bottom.x[i])
            assert np.isnan(bottom.y[i])
            assert np.isnan(top.x[i])
            assert np.isnan(top.y[i])


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
    values = F.interpolate_nz(  # ((N,), ...)
        lcl_p, pressure_levels, temperature, dewpoint
    )  # temp & dwpt values interpolated at LCL pressure

    assert_allclose(
        values,
        [[296.69, 296.78, 296.68, 296.97, 297.44], [295.12, 294.78, 295.23, 295.42, 296.22]],
        atol=1e-2,
    )
