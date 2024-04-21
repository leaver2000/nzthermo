from __future__ import annotations

from typing import Generic, Literal, NamedTuple, ParamSpec, Self, TypeVar, overload

import numpy as np
from numpy.typing import NDArray

from ._typing import N, Z, shape

P = ParamSpec("P")

float_ = TypeVar("float_", bound=np.float_)
newaxis = np.newaxis


# -------------------------------------------------------------------------------------------------
# interpolate_nz
# -------------------------------------------------------------------------------------------------
def linear_interpolate(
    x: NDArray[float_],
    x0: NDArray[float_],
    x1: NDArray[float_],
    y0: NDArray[float_],
    y1: NDArray[float_],
    /,
    *,
    log_x: bool = False,
    interp_nan: bool = False,
) -> NDArray[float_]:
    with np.errstate(divide="ignore", invalid="ignore"):
        if log_x is True:
            x = np.log(x)
        x = y0 + (x - x0) * (y1 - y0) / (x1 - x0)  # type: ignore

        if log_x is True:
            x = np.exp(x)

    if interp_nan is True:
        mask = np.isnan(x)
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])

    return x


def indices_nz(
    x: np.ndarray[shape[N], np.dtype[float_]],
    y: np.ndarray[shape[Z], np.dtype[float_]],
) -> tuple[np.ndarray[shape[N], np.dtype[np.intp]], np.ndarray[shape[Z], np.dtype[np.intp]]]:
    delta = x[:, newaxis] - y[newaxis, :]
    nx, zx = np.nonzero(np.diff(np.sign(delta), axis=1, append=0))
    _, index = np.unique(nx, return_index=True)
    return nx[index], zx[index]


@overload
def interpolate_nz(  # type: ignore[overload-overlap]
    x: np.ndarray[shape[N], np.dtype[float_]],
    xp: np.ndarray[shape[Z], np.dtype[float_]],
    fp: np.ndarray[shape[N, Z], np.dtype[float_]],
    /,
    *,
    mode: Literal["clip"] = ...,
    log_x: bool = False,
    interp_nan: bool = ...,
) -> np.ndarray[shape[N], np.dtype[float_]]: ...
@overload
def interpolate_nz(
    x: np.ndarray[shape[N], np.dtype[float_]],
    xp: np.ndarray[shape[Z], np.dtype[float_]],
    *args: np.ndarray[shape[N, Z], np.dtype[float_]],
    mode: Literal["clip"] = ...,
    log_x: bool = False,
    interp_nan: bool = ...,
) -> tuple[np.ndarray[shape[N], np.dtype[float_]], ...]: ...
def interpolate_nz(
    x: np.ndarray[shape[N], np.dtype[float_]],
    xp: np.ndarray[shape[Z], np.dtype[float_]],
    *args: np.ndarray[shape[N, Z], np.dtype[float_]],
    mode: Literal["clip"] = "clip",
    log_x: bool = False,
    interp_nan: bool = False,
) -> np.ndarray[shape[N], np.dtype[float_]] | tuple[np.ndarray[shape[N], np.dtype[float_]], ...]:
    """
    Interpolates values for multiple batches of data.

    Args:
        x: Input array of shape (N,) containing the values to be interpolated.
        xp: Input array of shape (Z,) containing the reference values.
        *args: Variable number of input arrays of shape (N, Z) containing additional data.

    Returns:
        np.ndarray or tuple of np.ndarray: Interpolated values for each batch.

    Raises:
        None

    Examples:
        >>> lcl_p = np.array([93290.11, 92921.01, 92891.83, 93356.17, 94216.14]) # (N,)
        >>> pressure_levels = np.array([101300., 100000.,  97500.,  95000.,  92500.,  90000.,  87500.,  85000.,  82500.,  80000.]) # (Z,)
        >>> temperature = np.array([
            [303.3 , 302.36, 300.16, 298.  , 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
            [303.58, 302.6 , 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6 ],
            [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
            [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1 ],
            [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47]]) # (N, Z)
        >>> dewpoint = np.array([
            [297.61, 297.36, 296.73, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
            [297.62, 297.36, 296.79, 296.18, 294.5 , 292.07, 287.74, 286.67, 285.15, 284.02],
            [297.76, 297.51, 296.91, 296.23, 295.05, 292.9 , 288.86, 287.12, 285.99, 283.98],
            [297.82, 297.56, 296.95, 296.23, 295.  , 292.47, 289.97, 288.45, 287.09, 285.17],
            [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31]]) # (N, Z)
        >>> F.interpolate_nz(lcl_p, pressure_levels, temperature, dewpoint) # temp & dwpt values interpolated at LCL pressure
        (array([296.69, 296.78, 296.68, 296.97, 297.44]), array([295.12, 294.78, 295.23, 295.42, 296.22]))

    """
    fp = np.array(args)  # (B, N, Z)
    assert fp.shape[1:] == (x.shape[0], xp.shape[0])
    if log_x is True:
        x = np.log(x)
        xp = np.log(xp)

    nx, zx = indices_nz(x, xp)

    x0, y0 = xp[zx], fp[:, nx, zx]
    assert mode == "clip"
    # TODO: add support for other modes
    zx = np.minimum(zx + 1, xp.size - 1)  # clip
    x1, y1 = xp[zx], fp[:, nx, zx]

    x = linear_interpolate(x, x0, x1, y0, y1, log_x=False, interp_nan=interp_nan)

    if x.shape[0] == 1:
        return x[0]

    return tuple(x)


# -------------------------------------------------------------------------------------------------
# insert_along_z
# -------------------------------------------------------------------------------------------------
def insert_along_z(
    arr: np.ndarray[shape[N, Z], np.dtype[float_]],
    values: np.ndarray[shape[N], np.dtype[float_]],
    z: np.ndarray[shape[Z], np.dtype[float_]],
    x: np.ndarray[shape[N] | shape[N, Z], np.dtype[float_]] | None = None,
) -> np.ndarray[shape[N, Z], np.dtype[float_]]:
    N, Z = arr.shape
    dtype = arr.dtype
    arr = np.pad(arr, ((0, 0), (0, 1)), mode="constant", constant_values=np.nan)
    Z += 1
    if x is None:
        x = values[:, newaxis]
    if x.ndim == 1:
        x = x[:, np.newaxis]
    elif not x.ndim == 2:
        raise ValueError("x must be a 1D or 2D array")

    indices = np.zeros((N, 1), dtype=int) + np.arange(Z)  # (N, Z)
    out = np.full((N, Z), np.nan, dtype=dtype)
    idx = np.argmin(np.abs(z[newaxis, :] - x), axis=1)
    # insert all of our new values
    out[np.arange(N), idx] = values

    # insert all the values below the new values
    mask = idx[:, newaxis] > indices
    out[mask] = arr[mask]

    # insert all the values above the new values
    mask |= np.isnan(arr)  # account for the additional padding
    nx, zx = np.nonzero(~mask)
    out[nx, np.minimum(zx + 1, Z)] = arr[nx, zx]

    return out


def mask_insert(
    mask: np.ndarray[shape[N, Z], np.dtype[np.bool_]],
    x: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    y: np.ndarray[shape[N, Z], np.dtype[np.float_]],
    *,
    kind: Literal["above", "below"] = "above",
    fill_value=np.nan,
    copy: bool = True,
) -> np.ma.MaskedArray[shape[N, Z], np.dtype[np.float_]]:
    if copy:
        mask = mask.copy()
    nx, zx = np.nonzero(mask)  # N, Z
    z = np.where(mask, x, y)
    zx += 1 if kind == "above" else -1
    mask[nx, zx] = True
    z[~mask] = fill_value
    return np.ma.masked_array(z, mask=~mask)


# -------------------------------------------------------------------------------------------------
# intersect_nz
# -------------------------------------------------------------------------------------------------
class Intersection(NamedTuple, Generic[float_]):
    x: np.ndarray[shape[N], np.dtype[float_]]
    y: np.ndarray[shape[N], np.dtype[float_]]
    indices: np.ndarray[shape[N], np.dtype[np.intp]]

    @property
    def shape(self) -> shape[N]:
        (N,) = self.x.shape
        return (N,)

    def lower(self) -> Self:
        cls = self.__class__
        x, y, indices = self
        _, idx = np.unique(indices, return_index=True)
        return cls(x[idx], y[idx], indices[idx])

    def upper(self) -> Self:
        cls = self.__class__
        (N,) = self.shape
        x, y, indices = self

        idx = np.flatnonzero(np.diff(indices, append=N))
        idx[np.isnan(x[idx])] -= 1
        idx = np.clip(idx, 0, N - 1)

        return cls(x[idx], y[idx], indices[idx])

    def to_numpy(self):
        return np.array([self.x, self.y])


def intersect_nz(
    x: np.ndarray[shape[Z] | shape[N, Z], np.dtype[float_]],
    a: np.ndarray[shape[N, Z], np.dtype[float_]],
    b: np.ndarray[shape[N, Z], np.dtype[float_]],
    *,
    log_x: bool = False,
) -> Intersection[float_]:
    """interpolate the points on `x` where `a` and `b` intersect.
    >>> x = np.array([1013, 1000,  975,  950])
    >>> a = np.array([[ 0, -1, -2, -3], [ 1, -1, -2, -3]])
    >>> b = np.array([[ 1. , -1.1, -1.2,  3. ], [ 2. , -1.2, -2. , -2. ]])
    >>> intersect = F.intersect_nz(x, a, b, log_x=True)
    >>> lower = intersect.lower()
    >>> lower
    Intersection(
      x=[1001.17 1002.16],
      y=[-0.91 -0.67],
      indices=[0 1]
    )
    >>> upper = intersect.upper()
    >>> upper
    Intersection(
      x=[997.19 975.  ],
      y=[-1.11 -2.  ],
      indices=[0 1]
    )
    """
    x = x.squeeze()
    Z = x.shape[-1]
    if not (a.shape[-1] == b.shape[-1] == Z):
        raise ValueError("a, b, and x must have the same number of elements")
    elif not (a.shape[0] == b.shape[0]):
        raise ValueError("a and b must have the same number of elements")

    if log_x is True:
        x = np.log(x)

    mask = np.diff(np.sign(a - b), axis=-1, append=1) != 0  # (N, Z)
    if np.sum(nah := np.all(~mask, axis=-1)):
        mask[nah] = True

    nx, z0 = np.nonzero(mask)
    z1 = np.clip(z0 + 1, 0, Z - 1)

    if x.ndim == 1:
        x0, x1 = x[z0], x[z1]
    else:
        x0, x1 = x[nx, z0], x[nx, z1]

    a0, a1 = a[nx, z0], a[nx, z1]
    b0, b1 = b[nx, z0], b[nx, z1]

    delta_y0 = a0 - b0
    delta_y1 = a1 - b1

    with np.errstate(divide="ignore", invalid="ignore"):
        x = ((delta_y1 * x0 - delta_y0 * x1)) / (delta_y1 - delta_y0)  # type: ignore
        y = ((x - x0) / (x1 - x0)) * (a1 - a0) + a0  # type: NDArray[float_] # type: ignore
        if log_x is True:
            x = np.exp(x)

    return Intersection(x, y, nx)  # type: ignore
