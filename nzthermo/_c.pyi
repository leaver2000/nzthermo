from typing import Any, Literal as L, TypeVar, overload

import numpy as np

from ._typing import Kelvin, Pascal, N, Z, shape

_T = TypeVar("_T")
_Dtype_t = TypeVar("_Dtype_t", bound=np.float_)

OPENMP_ENABLED: bool

@overload
def moist_lapse(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_Dtype_t]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[shape[N], np.dtype[np.float_]]],
    *,
    step: float = ...,
    dtype: type[_Dtype_t] | type[float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N], np.dtype[_Dtype_t]]]: ...
@overload
def moist_lapse(
    pressure: Pascal[
        np.ndarray[shape[Z], np.dtype[_Dtype_t]]
        | np.ndarray[shape[L[1], Z], np.dtype[_Dtype_t]]
        | np.ndarray[shape[N, Z], np.dtype[_Dtype_t]]
    ],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[np.float_]]] | None = ...,
    *,
    step: float = ...,
    dtype: type[_Dtype_t] | type[float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[_Dtype_t]]]: ...
@overload
def moist_lapse(
    pressure: Pascal[np.ndarray[Any, np.dtype[_Dtype_t]]],
    temperature: Kelvin[np.ndarray[Any, np.dtype[np.float_]]],
    reference_pressure: Pascal[np.ndarray[Any, np.dtype[np.float_]]] | None = ...,
    *,
    step: float = ...,
    dtype: type[_Dtype_t] | type[float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[Any, np.dtype[_Dtype_t]]]: ...
def lcl(
    pressure: Pascal[np.ndarray[shape[N], np.dtype[_Dtype_t]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[np.float_]]],
    *,
    max_iters: int = ...,
    tolerance: float = ...,
    dtype: type[_Dtype_t] | type[float] | L["float32", "float64"] | None = ...,
) -> tuple[
    Pascal[np.ndarray[shape[N], np.dtype[_Dtype_t]]],
    Kelvin[np.ndarray[shape[N], np.dtype[_Dtype_t]]],
]: ...
