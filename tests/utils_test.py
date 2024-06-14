from __future__ import annotations

import numpy as np
from numpy.testing import assert_array_equal

from nzthermo.utils import calendar, datetime64, julian, timeseries, unix

DATE_STRINGS = [
    # # A.D. 2023 January 1	00:00:00.0	2459945.500000
    "2023-01-01T00:00:00",
    # A.D. 2024 January 1	00:00:00.0	2460310.500000
    # A.D. 2024 January 1	00:00:00.0	2460310.500000
    "2024-01-01T00:00:00",  # leap year
    # A.D. 2024 January 1	12:00:00.0	2460311.000000
    "2024-01-01T12:00:00",  # leap year
    # # A.D. 2023 April 4	20:06:00.0	2460039.337500
    "2023-04-04T20:06:00",
    "2023-04-04T20:06:58",
    "2023-04-04T20:06:58",
]


def test_timeseries() -> None:
    ts = timeseries(DATE_STRINGS, "datetime64[s]")

    assert_array_equal(ts, ts.to_datetime64())
    assert_array_equal(
        ts.to_julian_day(),
        [2459945.5, 2460310.5, 2460311.0, 2460039.3375, 2460039.3381712963, 2460039.3381712963],
    )
    assert_array_equal(
        ts.to_unixtime(),
        [1672531200, 1704067200, 1704110400, 1680638760, 1680638818, 1680638818],
    )
    assert_array_equal(
        ts.to_calendar(),
        np.array(
            [
                (2023, 1, 1, 0, 0, 0, 0),
                (2024, 1, 1, 0, 0, 0, 0),
                (2024, 1, 1, 12, 0, 0, 0),
                (2023, 4, 4, 20, 6, 0, 0),
                (2023, 4, 4, 20, 6, 58, 0),
                (2023, 4, 4, 20, 6, 58, 0),
            ],
            dtype=np.dtype(
                [
                    ("year", np.int32),
                    ("month", np.int32),
                    ("day", np.int32),
                    ("hour", np.int32),
                    ("minute", np.int32),
                    ("second", np.int32),
                    ("microsecond", np.int32),
                ]
            ),
        ),
    )

    JD = ts.to(julian)
    UT = ts.to(unix)
    CD = ts.to(calendar)
    DT = ts.to(datetime64)
    for dtype in (julian, "julian"):
        assert np.all(
            (JD == ts.to(dtype))
            & (JD == JD.to(dtype))
            & (JD == UT.to(dtype))
            & (JD == CD.to(dtype))
            & (JD == DT.to(dtype))
        )

    for dtype in (unix, "unix"):
        assert np.all(
            (UT == ts.to(dtype))
            & (UT == JD.to(dtype))
            & (UT == UT.to(dtype))
            & (UT == CD.to(dtype))
            & (UT == DT.to(dtype))
        )
    for dtype in (calendar, "calendar"):
        assert np.all(
            (CD == ts.to(dtype))
            & (CD == JD.to(dtype))
            & (CD == UT.to(dtype))
            & (CD == CD.to(dtype))
            & (CD == DT.to(dtype))
        )
    for dtype in (datetime64, "datetime64"):
        assert np.all(
            (DT == ts.to(dtype))
            & (DT == JD.to(dtype))
            & (DT == UT.to(dtype))
            & (DT == CD.to(dtype))
            & (DT == DT.to(dtype))
        )


def test_delta_t() -> None:
    ts = timeseries(DATE_STRINGS, "datetime64[s]")
    assert_array_equal(
        ts.delta_t(),
        [
            73.31063670312504,
            73.89595545312505,
            73.89595545312505,
            73.45591845312504,
            73.45591845312504,
            73.45591845312504,
        ],
    )
