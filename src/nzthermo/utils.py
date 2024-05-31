from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
import datetime

from ._core import delta_t
from ._typing import SupportsArray, NestedSequence


class Timeseries(NamedTuple):
    years: NDArray[np.uint32]
    months: NDArray[np.uint32]
    days: NDArray[np.uint32]
    hours: NDArray[np.uint32]
    minutes: NDArray[np.uint32]
    seconds: NDArray[np.uint32]
    microseconds: NDArray[np.uint32]

    @property
    def leap_year(self) -> NDArray[np.bool_]:
        return (self.years % 4 == 0) & ((self.years % 100 != 0) | (self.years % 400 != 0))

    @property
    def leap_day(self) -> NDArray[np.bool_]:
        return (self.months == 2) & (self.days == 29)

    def to_datetime64(self) -> NDArray[np.datetime64]:
        Y, M, D, h, m, s, us = (
            a.astype(f"timedelta64[{x}]") for a, x in zip(self, ("Y", "M", "D", "h", "m", "s", "us"))
        )
        out = np.zeros(Y.shape, dtype="datetime64[Y]")
        out += Y - 1970

        out = out.astype("datetime64[M]")
        out += M - 1

        out = out.astype("datetime64[D]")
        out += D - 1

        out = out.astype("datetime64[h]")
        out += h

        out = out.astype("datetime64[m]")
        out += m

        out = out.astype("datetime64[s]")
        out += s

        out = out.astype("datetime64[us]")
        out += us

        return out  # type: ignore

    @property
    def delta_t(self) -> NDArray[np.float64]:
        return delta_t(self.years, self.months)


def timeseries(
    datetime: (
        SupportsArray[np.int_ | np.float_ | np.str_]
        | NestedSequence[SupportsArray[np.int_ | np.float_ | np.str_]]
        | NestedSequence[str | datetime.datetime | int | float]
        | str
        | datetime.datetime
        | int
        | float
        | np.int_
        | np.float_
        | np.str_
    ),
) -> Timeseries:
    dt = np.asarray(datetime, dtype=np.datetime64)
    out = np.empty((7,) + dt.shape, dtype="u4")
    Y, M, D, h, m, s = (dt.astype(f"M8[{x}]") for x in "YMDhms")
    out[0] = Y + 1970  # Gregorian Year
    out[1] = (M - Y) + 1  # month
    out[2] = (D - M) + 1  # day
    out[3] = (dt - D).astype("m8[h]")  # hour
    out[4] = (dt - h).astype("m8[m]")  # minute
    out[5] = (dt - m).astype("m8[s]")  # second
    out[6] = (dt - s).astype("m8[us]")  # microsecond
    return Timeseries(*out)
