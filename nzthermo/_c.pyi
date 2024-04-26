from typing import Any, Literal as L, TypeVar, overload

import numpy as np

from ._typing import Kelvin, Pascal, N, Z, shape

_T = TypeVar("_T")
_dtype_T = TypeVar("_dtype_T", np.float64, np.float32)
_dtype = type[_dtype_T] | type[float] | L["float32", "float64"]
OPENMP_ENABLED: bool

@overload
def moist_lapse(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]] = ...,
    *,
    step: float = ...,
    dtype: _dtype[_dtype_T] | None = ...,
) -> Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]]: ...
@overload
def moist_lapse(
    pressure: Pascal[
        np.ndarray[shape[Z], np.dtype[_dtype_T]]
        | np.ndarray[shape[L[1], Z], np.dtype[_dtype_T]]
        | np.ndarray[shape[N, Z], np.dtype[_dtype_T]]
    ],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[_dtype_T]]] | None = ...,
    *,
    step: float = ...,
    dtype: _dtype[_dtype_T] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_dtype_T]]]: ...
def moist_lapse(
    pressure: Pascal[np.ndarray[Any, np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[Any, np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[Any, np.dtype[_dtype_T]]] | None = ...,
    *,
    step: float = ...,
    dtype: _dtype[_dtype_T] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_dtype_T]]]: ...
def lcl(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    *,
    max_iters: int = ...,
    tolerance: float = ...,
    dtype: _dtype[_dtype_T] | None = ...,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
]: ...
