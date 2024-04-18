from typing import TypeVar

import numpy as np

from ._typing import Kelvin, N, Pascal, Z, shape

_T = TypeVar("_T")
_dtype_T = TypeVar("_dtype_T", np.float64, np.float32, np.float_)

OPENMP_ENABLED: bool

def moist_lapse(
    pressure: Pascal[np.ndarray[shape[N, Z], np.dtype[_dtype_T]] | np.ndarray[shape[N], np.dtype[_dtype_T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[_dtype_T]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[_dtype_T]]] | None = None,
    *,
    step: float = 1000.0,
    dtype: type[_dtype_T] | None = None,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_dtype_T]]]: ...
