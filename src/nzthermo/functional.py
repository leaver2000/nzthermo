from __future__ import annotations

import functools
from typing import NamedTuple, Generic
from typing import (
    Any,
    Callable,
    Concatenate,
    Iterable,
    Literal as L,
    ParamSpec,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
from numpy._typing._array_like import (
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
    _ArrayLikeTD64_co,
)
from numpy.typing import ArrayLike, NDArray

from .typing import N, Z, shape
from .utils import Profile, exactly_2d

_T = TypeVar("_T", bound=np.floating[Any])
_P = ParamSpec("_P")
_R = TypeVar("_R")

_T1 = TypeVar("_T1")


def map_partial(
    f: Callable[Concatenate[_T1, _P], _R], x: Iterable[_T1], *args: _P.args, **kwargs: _P.kwargs
) -> map[_R]:
    return map(functools.partial(f, *args, **kwargs), x)


def sort_nz(
    where: Callable[
        Concatenate[np.ndarray[shape[N, Z], np.dtype[Any]], _P],
        np.ndarray[shape[N, Z], np.dtype[np.bool_]],
    ]
    | np.ndarray[shape[N, Z], np.dtype[np.bool_]],
    pressure: np.ndarray[shape[N, Z], np.dtype[_T]],
    temperature: np.ndarray[shape[N, Z], np.dtype[_T]],
    dewpoint: np.ndarray[shape[N, Z], np.dtype[_T]],
    /,
    *args: _P.args,
    **kwargs: _P.kwargs,
):
    p_sort = np.argsort(pressure, axis=1, kind="quicksort")
    if pressure.shape != temperature.shape:
        sort = np.s_[:, p_sort]
        pressure = pressure[sort][:, ::-1]
        sort = np.arange(pressure.shape[0])[:, np.newaxis], p_sort
        temperature = temperature[sort][:, ::-1]
        dewpoint = dewpoint[sort][:, ::-1]
    else:
        sort = np.arange(pressure.shape[0])[:, np.newaxis], p_sort
        pressure = pressure[sort][:, ::-1]
        temperature = temperature[sort][:, ::-1]
        dewpoint = dewpoint[sort][:, ::-1]

    if callable(where):
        where = where(pressure, *args, **kwargs)

    pressure[where] = np.nan
    temperature[where] = np.nan
    dewpoint[where] = np.nan

    clip = max(np.argmax(np.isnan(pressure), axis=1)) + 1

    return pressure[:, :clip], temperature[:, :clip], dewpoint[:, :clip]


def nanwhere(
    mask: np.ndarray[shape[N, Z], np.dtype[np.bool_]],
    x: np.ndarray[shape[N, Z], np.dtype[_T]],
    *args: np.ndarray[shape[N, Z], np.dtype[_T]],
) -> tuple[np.ndarray[shape[N, Z], np.dtype[_T]], ...]:
    if x.shape == args[0].shape:
        return tuple(np.where(mask[np.newaxis, :, :], np.nan, [x, *args]))

    return (np.where(mask, np.nan, x),) + tuple(np.where(mask[np.newaxis, :, :], np.nan, args))


@overload
def nanroll_2d(__x: NDArray[_T]) -> np.ndarray[shape[N, Z], np.dtype[_T]]: ...
@overload
def nanroll_2d(
    *args: NDArray[_T],
) -> tuple[np.ndarray[shape[N, Z], np.dtype[_T]], ...]: ...
def nanroll_2d(
    *args: np.ndarray[Any, np.dtype[_T]],
) -> np.ndarray[shape[N, Z], np.dtype[_T]] | tuple[np.ndarray[shape[N, Z], np.dtype[_T]], ...]:
    args = tuple(np.where(np.isnan(x), np.roll(x, 1, axis=1), x) for x in exactly_2d(*args))

    if len(args) == 1:
        return args[0]

    return args


def nantrapz(
    y: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co | None = None,
    dx: float = 1.0,
    axis: SupportsIndex = -1,
    *,
    where: ArrayLike | None = None,
) -> NDArray[_T]:
    r"""
    This is a clone of the `numpy.trapz` function but with support for `nan` values.
    see the `numpy.lib.function_base.trapz` function for more information.

    Integrate along the given axis using the composite trapezoidal rule.

    If `x` is provided, the integration happens in sequence along its
    elements - they are not sorted.

    Integrate `y` (`x`) along each 1d slice on the given axis, compute
    :math:`\int y(x) dx`.
    When `x` is specified, this integrates along the parametric curve,
    computing :math:`\int_t y(t) dt =
    \int_t y(t) \left.\frac{dx}{dt}\right|_{x=x(t)} dt`.

    Parameters
    ----------
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    -------
    trapz : float or ndarray
        Definite integral of `y` = n-dimensional array as approximated along
        a single axis by the trapezoidal rule. If `y` is a 1-dimensional array,
        then the result is a float. If `n` is greater than 1, then the result
        is an `n`-1 dimensional array.

    Notes
    ------
    The try-except block was removed because it was not necessary, for the use case of this
    of this library.
    """
    if where is not None:
        y = np.where(where, y, np.nan)
        if x is not None:
            x = np.where(where, x, np.nan)

    else:
        y = np.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = np.asanyarray(x)
        if x.ndim == 1:
            d = np.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = np.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    return np.nansum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis=axis)


def intersect_nz(
    x: np.ndarray[shape[N, Z], np.dtype[_T]],
    a: np.ndarray[shape[N, Z], np.dtype[_T]],
    b: np.ndarray[shape[N, Z], np.dtype[_T]],
    direction: L["increasing", "decreasing"] = "increasing",
    log_x: bool = False,
) -> Profile[_T]:
    """
    NOTE: this function will leave a trailing `nan` value at the end of the array, and clip the
    array to the length of the longest column.

    >>> x = np.linspace(5, 30, 17)
    >>> y1 = 3 * x**2
    >>> y2 = 100 * x - 650
    >>> x = np.concatenate((x, -x), axis=0)
    >>> y1 = np.concatenate((y1, -y1), axis=0)
    >>> y2 = np.concatenate((y2, -y2), axis=0)
    >>> intersect = F.intersect_nz(x, y1, y2)
    >>> intersect
    x [[-8.88842282   24.44536424   nan]]
    y [[-238.84228188 1794.53642384 nan]]
    >>> intersect.bottom()
    x [24.44536424]
    y [1794.53642384]
    >>> intersect.top()
    x [-8.88842282]
    y [-238.84228188]
    """
    x, a, b = nanroll_2d(x, a, b)

    if log_x is True:
        x = np.log(x)

    x = np.broadcast_to(x.squeeze(), a.shape)
    nx, z0 = np.nonzero(np.diff(np.sign(a - b), axis=1))
    z1 = z0 + 1

    sign_change = np.sign(a[nx, z1] - b[nx, z1])

    x0, x1 = x[nx, z0], x[nx, z1]
    a0, a1 = a[nx, z0], a[nx, z1]
    b0, b1 = b[nx, z0], b[nx, z1]
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1

    with np.errstate(divide="ignore", invalid="ignore"):
        x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)  # type: ignore
        y = ((x - x0) / (x1 - x0)) * (a1 - a0) + a0
        if log_x:
            x = np.exp(x)

    x_full = np.full_like(a, fill_value=np.nan)
    y_full = np.full_like(a, fill_value=np.nan)

    if direction == "increasing":
        x[sign_change <= 0] = np.nan  # increasing
    else:
        x[sign_change >= 0] = np.nan

    x_full[nx, z0] = x
    y_full[nx, z0] = y

    sort = np.arange(x_full.shape[0])[:, np.newaxis], np.argsort(x_full, axis=1)
    x, y = x_full[sort], y_full[sort]

    clip = max(np.argmax(np.isnan(x), axis=1)) + 1

    return Profile(x[:, :clip], y[:, :clip])


def zero_crossings(
    X: np.ndarray[shape[N, Z], np.dtype[_T]] | np.ndarray[shape[N], np.dtype[_T]],
    Y: np.ndarray[shape[N, Z], np.dtype[_T]],
) -> Profile[_T]:
    """
    This function targets the `metpy.thermo._find_append_zero_crossings` function but with support
    for 2D arrays.


    >>> F.find_append_zero_crossings(x, x - y)
    """
    N, Z = Y.shape
    X = np.broadcast_to(X, (N, Z))

    x_full, y_full = np.full((2, N, Z), fill_value=np.nan, dtype=X.dtype)
    x, y = np.log(X[:, 1:]), Y[:, 1:]

    nx, z0 = np.nonzero(np.diff(np.sign(y), axis=1))
    z1 = z0 + 1

    x0, x1 = x[nx, z0], x[nx, z1]
    y0, y1 = y[nx, z0], y[nx, z1]

    with np.errstate(divide="ignore", invalid="ignore"):
        x_full[nx, z0] = np.exp(x := (y1 * x0 - y0 * x1) / (y1 - y0))
        y_full[nx, z0] = ((x - x0) / (x1 - x0)) * (y1 - y0) + y0

    x, y = np.concatenate([X, x_full], axis=1), np.concatenate([Y, y_full], axis=1)

    # sort all of the values with the lowest value first and all of the nan values last
    sort = np.arange(N)[:, np.newaxis], np.argsort(x, axis=1, kind="quicksort")
    x, y = x[sort], y[sort]
    # clip axis 1 to the last non nan value.
    clip = max(np.argmax(np.isnan(x), axis=1)) + 1

    return Profile(x[:, :clip], y[:, :clip])


class TopK(NamedTuple, Generic[_T]):
    values: NDArray[_T]
    indices: NDArray[np.intp]


def topk(
    values: NDArray[_T], k: int, axis: int = -1, absolute: bool = False, sort: bool = False
) -> TopK[_T]:
    values = np.asarray(values)
    arg = np.abs(values) if absolute else values
    arg[np.isnan(values)] = -np.inf
    indices = np.argpartition(arg, -k, axis=axis)[-k:]
    values = np.take_along_axis(values, indices, axis=axis)
    if sort:
        idx = np.flip(np.argsort(values, axis=axis), axis=axis)
        indices = np.take_along_axis(indices, idx, axis=axis)
        values = np.take_along_axis(values, idx, axis=axis)

    return TopK(values, indices)
