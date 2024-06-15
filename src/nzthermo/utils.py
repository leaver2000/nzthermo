from __future__ import annotations

import functools
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Literal as L,
    Mapping,
    NamedTuple,
    ParamSpec,
    Self,
    Sequence,
    TypeGuard,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._ufunc import delta_t, greater_or_close, less_or_close
from .typing import Kelvin, N, NestedSequence, Pascal, SupportsDType, Z, shape

_T = TypeVar("_T")
_P = ParamSpec("_P")
float_ = TypeVar("float_", np.float_, np.floating[Any], covariant=True)


class PVectorNd(NamedTuple, Generic[_T, float_]):
    pressure: Pascal[np.ndarray[_T, np.dtype[float_]]]
    temperature: Kelvin[np.ndarray[_T, np.dtype[float_]]]

    def where(
        self,
        condition: np.ndarray[_T, np.dtype[np.bool_]]
        | Callable[[Self], np.ndarray[_T, np.dtype[np.bool_]]],
        x_fill: ArrayLike = np.nan,
        y_fill: ArrayLike | None = None,
    ) -> Self:
        if callable(condition):
            condition = condition(self)

        if y_fill is None:
            y_fill = x_fill
        return self.__class__(
            np.where(condition, self.pressure, x_fill),
            np.where(condition, self.temperature, y_fill),
        )

    def is_below(
        self, pressure: Pascal[NDArray[np.float_]] | PVectorNd, *, close: bool = False
    ) -> NDArray[np.bool_]:
        if isinstance(pressure, PVectorNd):
            pressure = pressure.pressure
        if not close:
            return self.pressure > pressure

        return less_or_close(self.pressure, pressure).astype(np.bool_)

    def is_above(
        self, pressure: Pascal[NDArray[np.float_]] | PVectorNd, *, close: bool = False
    ) -> NDArray[np.bool_]:
        if isinstance(pressure, PVectorNd):
            pressure = pressure.pressure
        if not close:
            return self.pressure < pressure

        return greater_or_close(self.pressure, pressure).astype(np.bool_)

    def is_nan(self) -> NDArray[np.bool_]:
        return np.isnan(self.pressure)

    def select(
        self,
        condlist: Sequence[NDArray[np.bool_]],
        x_choice: Sequence[NDArray[float_]],
        y_choice: Sequence[NDArray[float_]],
        x_default: ArrayLike = np.nan,
        y_default: ArrayLike | None = None,
    ) -> Self:
        if y_default is None:
            y_default = x_default
        return self.__class__(
            np.select(condlist, x_choice, default=x_default),
            np.select(condlist, y_choice, default=y_default),
        )

    def __repr__(self) -> str:
        return "\n".join(
            f"[{name}, {x.shape}]\n{np.array2string(x, precision=2)}"
            for name, x in zip(["pressure", "temperature"], self)
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_func(
        cls,
        func: Callable[
            _P,
            tuple[Pascal[NDArray[float_]], Kelvin[NDArray[float_]]],
        ],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Self:
        x, y = func(*args, **kwargs)
        return cls(x, y)

    def reshape(self, *shape: int) -> tuple[Pascal[NDArray[float_]], Kelvin[NDArray[float_]]]:
        p, t = np.reshape([self.pressure, self.temperature], (2, *shape))
        return p, t


class Vector1d(PVectorNd[shape[N], float_]):
    def unsqueeze(self) -> Vector2d[float_]:
        s = np.s_[:, np.newaxis]
        return Vector2d(self.pressure[s], self.temperature[s])


class Vector2d(PVectorNd[shape[N, Z], float_]):
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

    def sort(self) -> Self:
        N = self.pressure.shape[0]
        sort = np.arange(N)[:, np.newaxis], np.argsort(self.pressure, axis=1, kind="quicksort")
        return self.__class__(self.pressure[sort], self.temperature[sort])


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
    f: Callable[
        Concatenate[
            Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
            Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
            Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
            _P,
        ],
        _T,
    ],
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


# ............................................................................................... #
# TIME SERIES UTILITIES
# ............................................................................................... #
UNIX_EPOCH = 2440587.5

calendar = np.void
unix = np.int64
julian = np.float64
datetime64 = np.datetime64


_DT1 = TypeVar("_DT1", datetime64, calendar, julian, unix)
_DT2 = TypeVar("_DT2", datetime64, calendar, julian, unix)
DType_DT1 = np.dtype[_DT1] | type[_DT1] | SupportsDType[_DT1]


@overload
def isdtype(x: NDArray[Any], /, dtype: DType_DT1[_DT1]) -> TypeGuard[NDArray[_DT1]]: ...
@overload
def isdtype(x: np.dtype[_DT1], /, dtype: DType_DT1[_DT1]) -> TypeGuard[np.dtype[_DT1]]: ...
@overload
def isdtype(x: type[_DT1], /, dtype: DType_DT1[_DT1]) -> TypeGuard[np.dtype[_DT1]]: ...
def isdtype(
    x: NDArray[Any] | np.dtype[_DT1] | type[_DT1], /, dtype: DType_DT1[_DT1]
) -> TypeGuard[NDArray[_DT1]] | TypeGuard[np.dtype[_DT1]]:
    if isinstance(x, np.dtype):
        x_dtype = x
    elif isinstance(x, type):
        x_dtype = np.dtype(x)
    else:
        x_dtype = x.dtype

    if dtype is calendar:
        return (
            np.issubdtype(x_dtype, calendar)
            and x_dtype.names is not None
            and (
                {"year", "month", "day", "hour", "minute", "second", "microsecond"}.issubset(
                    x_dtype.names
                )
            )
        )
    return np.issubdtype(x_dtype, dtype)


def leap_year(x: NDArray[datetime64 | calendar | julian | unix]) -> NDArray[np.bool_]:
    x = to_date(x)
    year = x["year"]
    return (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))


def leap_day(
    x: NDArray[datetime64 | calendar | julian | unix],
) -> NDArray[np.bool_]:
    x = to_date(x)
    year, month, day = x["year"], x["month"], x["day"]
    return leap_year(year) & (month == 2) & (day == 29)


# ............................................................................................... #
# calendar
# ............................................................................................... #
def to_date(
    x: NDArray[datetime64 | calendar | julian | unix],
) -> np.recarray[Any, np.dtype[calendar]]:
    if isdtype(x, calendar):
        return x.view(np.recarray)
    elif isdtype(x, unix) or isdtype(x, julian):
        x = to_datetime64(x)

    out = np.recarray(
        x.shape, dtype=np.dtype([("year", np.int32), ("month", np.int32), ("day", np.int32)])
    )

    Y, M, D = (x.astype(f"datetime64[{s}]") for s in "YMD")

    out["year"] = Y + 1970
    out["month"] = (M - Y) + 1
    out["day"] = (D - M) + 1

    return out


def to_calendar(
    x: NDArray[datetime64 | calendar | julian | unix],
) -> np.recarray[Any, np.dtype[calendar]]:
    if isdtype(x, calendar):
        return x.view(np.recarray)
    elif isdtype(x, unix) or isdtype(x, julian):
        x = to_datetime64(x)

    out = np.recarray(
        x.shape,
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
    )

    Y, M, D, h, m, s = (x.astype(f"datetime64[{s}]") for s in "YMDhms")

    out["year"] = Y + 1970
    out["month"] = (M - Y) + 1
    out["day"] = (D - M) + 1
    out["hour"] = (x - D).astype("timedelta64[h]", copy=False)
    out["minute"] = (x - h).astype("timedelta64[m]", copy=False)
    out["second"] = (x - m).astype("timedelta64[s]", copy=False)
    out["microsecond"] = (x - s).astype("timedelta64[us]", copy=False)

    return out


# ............................................................................................... #
# datetime64
# ............................................................................................... #
def to_datetime64(
    x: NDArray[datetime64 | calendar | julian | unix],
) -> NDArray[datetime64]:
    if isdtype(x, datetime64):
        return x
    elif isdtype(x, julian):
        return np.ceil((x - UNIX_EPOCH) * 86400).astype("datetime64[s]", copy=False)
    elif isdtype(x, unix):
        return x.astype("datetime64[s]", copy=False)
    elif isdtype(x, calendar):
        out = np.zeros(x.shape, dtype="datetime64[Y]")
        out += x["year"].astype("timedelta64[Y]") - np.timedelta64(1970, "Y")

        out = out.astype("datetime64[M]", copy=False)
        out += x["month"] - np.timedelta64(1, "M")

        out = out.astype("datetime64[D]", copy=False)
        out += x["day"] - np.timedelta64(1, "D")

        out = out.astype("datetime64[h]", copy=False)
        out += x["hour"].astype("timedelta64[h]")

        out = out.astype("datetime64[m]", copy=False)
        out += x["minute"].astype("timedelta64[m]")

        out = out.astype("datetime64[s]", copy=False)
        out += x["second"].astype("timedelta64[s]")

        out = out.astype("datetime64[us]", copy=False)
        out += x["microsecond"].astype("timedelta64[us]")

        return out  # type: ignore

    raise NotImplementedError


# ............................................................................................... #
# unix
# ............................................................................................... #
def to_unixtime(x: NDArray[datetime64 | calendar | julian | unix]) -> NDArray[unix]:
    if isdtype(x, unix):
        return x
    elif isdtype(x, datetime64):
        return x.astype(unix, copy=True)
    elif isdtype(x, julian):
        return np.ceil((x - UNIX_EPOCH) * 86400).astype(unix, copy=False)
    elif isdtype(x, calendar):
        return (
            to_datetime64(x)
            .astype("datetime64[s]", copy=False)  # drop microsecond precision
            .astype(unix, copy=False)
        )

    raise NotImplementedError


# ............................................................................................... #
# julian
# ............................................................................................... #
def to_julian_day(x: NDArray[datetime64 | calendar | julian | unix]) -> NDArray[julian]:
    if isdtype(x, julian):
        return x

    cd = to_calendar(x)

    Y = cd["year"]
    M = cd["month"]
    D = cd["day"]
    h = cd["hour"]
    m = cd["minute"]
    s = cd["second"]
    ms = cd["microsecond"]

    f = np.ceil((M - 14) / 12)
    ymd = (
        np.ceil((1461 * (Y + 4800 + f)) / 4)
        + np.ceil((367 * (M - 2 - 12 * f)) / 12)
        - np.ceil((3 * np.ceil((Y + 4900 + f) / 100)) / 4)
        + D
        - 32075
    )
    hms = (h * 3600 + m * 60 + (s + (ms / 1_000_000))) / 86400
    return ymd + hms - 0.5


_string_map = {
    "datetime64": datetime64,
    "calendar": calendar,
    "julian": julian,
    "unix": unix,
}

_function_map: Mapping[
    type[datetime64 | calendar | julian | unix], Callable[[NDArray[Any]], NDArray[Any]]
] = {datetime64: to_datetime64, calendar: to_calendar, julian: to_julian_day, unix: to_unixtime}


def cast_to(
    x: NDArray[datetime64 | calendar | julian | unix], dtype: type[_DT2] | str
) -> NDArray[_DT2]:
    if isinstance(dtype, str):
        return _function_map[_string_map[dtype]](x)

    return _function_map[dtype](x)


class timeseries(np.ndarray[Any, np.dtype[_DT1]]):
    @overload
    def __new__(
        cls,
        data: Any,
        dtype: L[
            "datetime64",
            "datetime64[Y]",
            "datetime64[M]",
            "datetime64[W]",
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ] = ...,
    ) -> timeseries[datetime64]: ...
    @overload
    def __new__(
        cls,
        data: Any,
        dtype: np.dtype[_DT1] = ...,
    ) -> timeseries[_DT1]: ...
    def __new__(
        cls,
        data: Any,
        dtype: np.dtype[_DT1]
        | L[
            "datetime64",
            "datetime64[Y]",
            "datetime64[M]",
            "datetime64[W]",
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ]
        | None = None,
    ) -> timeseries[Any]:
        return np.array(data, dtype).view(timeseries)

    @overload
    def to(self, dtype: L["unix"]) -> timeseries[unix]: ...
    @overload
    def to(self, dtype: L["datetime64"]) -> timeseries[datetime64]: ...
    @overload
    def to(self, dtype: L["calendar"]) -> timeseries[calendar]: ...
    @overload
    def to(self, dtype: L["julian"]) -> timeseries[julian]: ...
    @overload
    def to(self, dtype: type[_DT2]) -> timeseries[_DT2]: ...
    def to(
        self, dtype: type[_DT2] | L["unix", "datetime64", "calendar", "julian"]
    ) -> timeseries[Any]:
        return cast_to(self, dtype).view(timeseries)

    def to_date(self) -> timeseries[calendar]:
        return self.to(calendar)

    def to_calendar(self) -> timeseries[calendar]:
        return self.to(calendar)

    def to_datetime64(self) -> timeseries[datetime64]:
        return self.to(datetime64)

    def to_unixtime(self) -> timeseries[unix]:
        return self.to(unix)

    def to_julian_day(self) -> timeseries[julian]:
        return self.to(julian)

    def delta_t(self) -> NDArray[np.float64]:
        date = self.to_date().view(np.recarray)

        return delta_t(date["year"], date["month"])
