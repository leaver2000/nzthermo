from __future__ import annotations

from typing import (
    Callable,
    SupportsIndex,
    Any,
    overload,
    Literal as L,
    TypeVar,
    Generic,
    NamedTuple,
)

import numpy as np
from numpy._typing._array_like import _ArrayLikeComplex_co, _ArrayLikeObject_co, _ArrayLikeTD64_co
from numpy.typing import NDArray

from .typing import N, Z, shape

_T = TypeVar("_T")
float_ = TypeVar("float_", bound=np.float_)


def not_nan(x: NDArray[Any]) -> NDArray[np.bool_]:
    return ~np.isnan(x)


class ElementNd(NamedTuple, Generic[_T, float_]):
    x: np.ndarray[_T, np.dtype[float_]]
    y: np.ndarray[_T, np.dtype[float_]]


class Element1d(ElementNd[shape[N], float_]): ...


class Element2d(ElementNd[shape[N, Z], float_]):
    def pick(self, which: L["bottom", "top"] = "top") -> ElementNd[shape[N], float_]:
        x, y = self.x, self.y
        if which == "bottom":
            idx = np.s_[
                np.arange(x.shape[0]),
                np.argmin(~np.isnan(x), axis=1) - 1,  # the last non-nan value
            ]

        elif which == "top":
            idx = np.s_[:, 0]

        return ElementNd(x[idx], y[idx])

    def bottom(self) -> ElementNd[shape[N], float_]:
        return self.pick("bottom")

    def top(self) -> ElementNd[shape[N], float_]:
        return self.pick("top")


@overload
def exactly_2d(
    __x: np.ndarray[Any, np.dtype[np.float_]],
) -> np.ndarray[shape[N, Z], np.dtype[np.float_]]: ...
@overload
def exactly_2d(
    *args: np.ndarray[Any, np.dtype[np.float_]],
) -> tuple[np.ndarray[shape[N, Z], np.dtype[np.float_]]]: ...
def exactly_2d(
    *args: np.ndarray[Any, np.dtype[np.float_]],
) -> (
    np.ndarray[shape[N, Z], np.dtype[np.float_]]
    | tuple[np.ndarray[shape[N, Z], np.dtype[np.float_]], ...]
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


@overload
def nanroll_2d(
    __x: np.ndarray[Any, np.dtype[np.float_]],
) -> np.ndarray[shape[N, Z], np.dtype[np.float_]]: ...
@overload
def nanroll_2d(
    *args: np.ndarray[Any, np.dtype[np.float_]],
) -> tuple[np.ndarray[shape[N, Z], np.dtype[np.float_]], ...]: ...
def nanroll_2d(
    *args: np.ndarray[Any, np.dtype[np.float_]],
) -> (
    np.ndarray[shape[N, Z], np.dtype[np.float_]]
    | tuple[np.ndarray[shape[N, Z], np.dtype[np.float_]], ...]
):
    args = exactly_2d(*args)
    values = []
    for x in args:
        values.append(np.where(np.isnan(x), np.roll(x, 1, axis=1), x))

    if len(values) == 1:
        return values[0]

    return tuple(values)


def logical_or_close(
    op: Callable[
        [NDArray[np.float_] | float, NDArray[np.float_] | float], NDArray[np.bool_] | bool
    ],
    a: NDArray[np.float_] | float,
    b: NDArray[np.float_] | float,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
) -> NDArray[np.bool_]:
    return np.logical_or(op(a, b), np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def logical_and_close(
    op: Callable[
        [NDArray[np.float_] | float, NDArray[np.float_] | float], NDArray[np.bool_] | bool
    ],
    a: NDArray[np.float_] | float,
    b: NDArray[np.float_] | float,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
    equal_nan: bool = False,
) -> NDArray[np.bool_]:
    return np.logical_and(op(a, b), np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan))


def nantrapz(
    y: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co | None = None,
    dx: float = 1.0,
    axis: SupportsIndex = -1,
) -> NDArray[np.float_]:
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


def find_intersections(
    x: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    a: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    b: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    direction: L["increasing", "decreasing"] = "increasing",
    log_x: bool = False,
) -> Element2d[np.float_]:
    x, a, b = nanroll_2d(x, a, b)

    if log_x is True:
        x = np.log(x)

    x = np.broadcast_to(x.squeeze(), a.shape)
    ind, nearest_idx = np.nonzero(np.diff(np.sign(a - b), axis=1))
    next_idx = nearest_idx + 1
    sign_change = np.sign(a[ind, next_idx] - b[ind, next_idx])
    x0, x1 = x[ind, nearest_idx], x[ind, next_idx]
    a0, a1 = a[ind, nearest_idx], a[ind, next_idx]
    b0, b1 = b[ind, nearest_idx], b[ind, next_idx]
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1

    with np.errstate(divide="ignore", invalid="ignore"):
        x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)  # type: ignore
        y = ((x - x0) / (x1 - x0)) * (a1 - a0) + a0  # type: NDArray[float_] # type: ignore
        if log_x:
            x = np.exp(x)

    if direction == "increasing":
        x[sign_change <= 0] = np.nan  # increasing
    else:
        x[sign_change >= 0] = np.nan

    x_full = np.full_like(a, fill_value=np.nan)
    y_full = np.full_like(a, fill_value=np.nan)

    x_full[ind, nearest_idx] = x[...]
    y_full[ind, nearest_idx] = y[...]

    sort = np.arange(x_full.shape[0])[:, np.newaxis], np.argsort(x_full, axis=1)
    x, y = x_full[sort], y_full[sort]

    clip = max(np.argmax(np.isnan(x), axis=1)) + 1

    return Element2d(x[:, :clip], y[:, :clip])


def find_append_zero_crossings(
    X: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    Y: np.ndarray[shape[N, Z], np.dtype[np.float_]],
) -> Element2d[np.float_]:
    """
    This function targets the `metpy.thermo._find_append_zero_crossings` function but with support
    for 2D arrays.
    """
    x = np.log(X[:, 1:])
    a = Y[:, 1:]
    b = np.zeros_like(a)

    ind, nearest_idx = np.nonzero(np.diff(np.sign(a - b), axis=1))

    next_idx = np.clip(nearest_idx + 1, 0, X.shape[1] - 1)
    x0, x1 = x[ind, nearest_idx], x[ind, next_idx]
    a0, a1 = a[ind, nearest_idx], a[ind, next_idx]
    b0, b1 = b[ind, nearest_idx], b[ind, next_idx]
    delta_y0 = a0 - b0
    delta_y1 = a1 - b1

    with np.errstate(divide="ignore", invalid="ignore"):
        x = (delta_y1 * x0 - delta_y0 * x1) / (delta_y1 - delta_y0)  # type: ignore
        y = ((x - x0) / (x1 - x0)) * (a1 - a0) + a0  # type: NDArray[float_] # type: ignore
        x = np.exp(x)

    x_full = np.full_like(X, fill_value=np.nan)
    y_full = np.full_like(Y, fill_value=np.nan)
    x_full[ind, nearest_idx] = x
    y_full[ind, nearest_idx] = y

    x = np.column_stack([X, x_full])
    y = np.column_stack([Y, y_full])

    sort = np.arange(x.shape[0])[:, None], np.argsort(x, axis=1)
    x, y = x[sort], y[sort]

    clip = max(np.argmax(np.isnan(x), axis=1))

    return Element2d(x[:, :clip], y[:, :clip])
