from __future__ import annotations

import datetime
import functools
import textwrap
from typing import (
    Callable,
    Concatenate,
    Generic,
    Literal as L,
    NamedTuple,
    ParamSpec,
    Self,
    TypeVar,
    overload,
    Any,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ufunc import delta_t
from .typing import Kelvin, N, NestedSequence, NZArray, Pascal, SupportsArray, Z, shape

_T = TypeVar("_T")
_P = ParamSpec("_P")
Pair = tuple[_T, _T]
float_ = TypeVar("float_", np.float_, np.floating[Any], covariant=True)


class VectorNd(NamedTuple, Generic[_T, float_]):
    pressure: Pascal[np.ndarray[_T, np.dtype[float_]]]
    temperature: Kelvin[np.ndarray[_T, np.dtype[float_]]]

    def where(
        self,
        condition: np.ndarray[_T, np.dtype[np.bool_]],
        x_fill: ArrayLike = np.nan,
        y_fill: ArrayLike | None = None,
    ) -> Self:
        if y_fill is None:
            y_fill = x_fill
        return self.__class__(
            np.where(condition, self.pressure, x_fill),
            np.where(condition, self.temperature, y_fill),
        )

    def __repr__(self) -> str:
        prefix = " " * 2
        return "\n".join(
            f"{name} {textwrap.indent(np.array2string(x), prefix).removeprefix(prefix)}"
            for name, x in zip(["pressure", "temperature"], self)
        )

    def __str__(self) -> str:
        return self.__repr__()


class Vector1d(VectorNd[shape[N], float_]): ...


class Vector2d(VectorNd[shape[N, Z], float_]):
    def pick(self, which: L["bottom", "top"]) -> Vector1d[float_]:
        p, t = self.pressure, self.temperature
        nx = np.arange(p.shape[0])
        if which == "bottom":
            idx = np.argmin(~np.isnan(p), axis=1) - 1  # the last non-nan value
            return Vector1d(p[nx, idx], t[nx, idx])

        elif which == "top":
            return Vector1d(p[nx, 0], t[nx, 0])  # the first value is the uppermost value

    def bottom(self) -> Vector1d[float_]:
        return self.pick("bottom")

    def top(self) -> Vector1d[float_]:
        return self.pick("top")


@overload
def exactly_2d(x: NDArray[float_], /) -> np.ndarray[shape[N, Z], np.dtype[float_]]: ...
@overload
def exactly_2d(
    *args: NDArray[float_],
) -> tuple[np.ndarray[shape[N, Z], np.dtype[float_]], ...]: ...
def exactly_2d(
    *args: NDArray[float_],
) -> (
    np.ndarray[shape[N, Z], np.dtype[float_]]
    | tuple[np.ndarray[shape[N, Z], np.dtype[float_]], ...]
):
    values = []
    for x in args:
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[np.newaxis, :]
        elif x.ndim != 2:
            raise ValueError("pressure must be a 1D or 2D array")
        values.append(x)

    if len(values) == 1:
        return values[0]

    return tuple(values)


def broadcast_nz(
    f: Callable[Concatenate[NZArray[float_], NZArray[float_], NZArray[float_], _P], _T],
) -> Callable[
    Concatenate[
        Pascal[NDArray[float_] | NestedSequence[float]],
        Kelvin[NDArray[float_]] | NestedSequence[float],
        Kelvin[NDArray[float_]] | NestedSequence[float],
        _P,
    ],
    _T,
]:
    @functools.wraps(f)
    def wrapper(
        pressure: NDArray[float_] | NestedSequence[float],
        temperature: NDArray[float_] | NestedSequence[float],
        dewpoint: NDArray[float_] | NestedSequence[float],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _T:
        # TODO
        # - add support for squeezing what would have been a 1d input
        # - add support for reshaping:
        #   (T, Z, Y, X) -> (N, Z)
        #   (Z, Y, X) -> (N, Z)'
        if kwargs.pop("__fastpath", False):
            return f(pressure, temperature, dewpoint, *args, **kwargs)  # type: ignore

        pressure, temperature, dewpoint = exactly_2d(
            np.asarray(pressure), np.asarray(temperature), np.asarray(dewpoint)
        )
        return f(pressure, temperature, dewpoint, *args, **kwargs)

    return wrapper


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
            a.astype(f"timedelta64[{x}]")
            for a, x in zip(self, ("Y", "M", "D", "h", "m", "s", "us"))
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
        | NestedSequence[SupportsArray[np.int_ | float_ | np.str_]]
        | NestedSequence[str | datetime.datetime | int | float]
        | str
        | datetime.datetime
        | int
        | float
        | np.int_
        | float_
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
