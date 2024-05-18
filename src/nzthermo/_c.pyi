from typing import Any, Literal as L, TypeVar, overload

import numpy as np

from ._typing import Kelvin, N, Pascal, T, Z, shape

_Dtype_t = TypeVar("_Dtype_t", bound=np.float_)

OPENMP_ENABLED: bool

@overload
def moist_lapse[T: np.float_](  # type: ignore
    pressure: Pascal[np.ndarray[shape[N], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[shape[N], np.dtype[np.float_]]],
    *,
    step: float = ...,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N], np.dtype[T]]]: ...
@overload
def moist_lapse[T: np.float_](
    pressure: Pascal[
        np.ndarray[shape[Z], np.dtype[T]]
        | np.ndarray[shape[L[1], Z], np.dtype[T]]
        | np.ndarray[shape[N, Z], np.dtype[T]]
    ],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[np.float_]]] | None = ...,
    *,
    step: float = ...,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]]: ...
@overload
def moist_lapse[T: np.float_](
    pressure: Pascal[np.ndarray[Any, np.dtype[T]]],
    temperature: Kelvin[np.ndarray[Any, np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[Any, np.dtype[np.float_]]] | None = ...,
    *,
    step: float = ...,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[Any, np.dtype[T]]]: ...
def lcl[T: np.float_](
    pressure: Pascal[np.ndarray[shape[N], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    *,
    max_iters: int = ...,
    tolerance: float = ...,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[T]]],
    Kelvin[np.ndarray[shape[N], np.dtype[T]]],
]: ...
