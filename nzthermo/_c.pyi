from typing import Any, Literal, TypeVar, Union, overload

import numpy as np

from ._typing import Kelvin, N, Pascal, Z, shape

_T = TypeVar("_T")
_dtype_T = TypeVar("_dtype_T", np.float64, np.float32, np.float_)
_dtype = type[_dtype_T] | type[float] | Literal["float32", "float64"]
OPENMP_ENABLED: bool

@overload
def moist_lapse(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[_dtype_T]]] | None = None,
    *,
    step: float = 1000.0,
    dtype: _dtype[_dtype_T] | None = None,
) -> Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]]: ...
@overload
def moist_lapse(
    pressure: Pascal[
        Union[
            np.ndarray[shape[Z], np.dtype[_dtype_T]],
            np.ndarray[shape[Literal[1], Z], np.dtype[_dtype_T]],
            np.ndarray[shape[N, Z], np.dtype[_dtype_T]],
        ]
    ],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[_dtype_T]]] | None = None,
    *,
    step: float = 1000.0,
    dtype: _dtype[_dtype_T] | None = None,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_dtype_T]]]: ...
def moist_lapse(
    pressure: Pascal[np.ndarray[Any, np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[Any, np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[Any, np.dtype[_dtype_T]]] | None = None,
    *,
    step: float = 1000.0,
    dtype: _dtype[_dtype_T] | None = None,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_dtype_T]]]: ...
def lcl(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    *,
    max_iters: int = 50,
    tolerance: float = 0.1,
    dtype: _dtype[_dtype_T] | None = None,
) -> Pascal[np.ndarray[shape[Literal[2], N], np.dtype[_dtype_T]]]: ...
