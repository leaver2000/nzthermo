from __future__ import annotations
import numpy as np
from nzthermo.utils import timeseries
from numpy.testing import assert_array_equal, assert_allclose


def test_timeseries() -> None:
    values = [
        "2020-02-29T00:00:00",  # leap year
        "2021-02-28T00:00:00",
        "2022-02-28T00:00:00",
        "2023-02-28T00:00:00",
        "2024-02-29T00:00:00",  # leap year
    ]
    ts = timeseries(values)
    assert_array_equal(ts.years, [2020, 2021, 2022, 2023, 2024])
    assert_array_equal(ts.months, [2, 2, 2, 2, 2])
    assert_array_equal(ts.days, [29, 28, 28, 28, 29])
    assert_array_equal(ts.hours, [0, 0, 0, 0, 0])
    assert_array_equal(ts.minutes, [0, 0, 0, 0, 0])
    assert_array_equal(ts.seconds, [0, 0, 0, 0, 0])
    assert_array_equal(ts.microseconds, [0, 0, 0, 0, 0])
    assert_array_equal(ts.to_datetime64(), np.array(values, dtype="datetime64[us]"))
    assert_allclose(ts.delta_t, [71.66730358, 72.22001983, 72.78391408, 73.35898633, 73.94523658])
    assert_array_equal(ts.leap_year, [True, False, False, False, True])
    assert_array_equal(ts.leap_day, [True, False, False, False, True])
