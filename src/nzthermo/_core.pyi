from typing import (
    Any,
    Literal as L,
    ParamSpec,
    TypeVar,
    overload,
)

import numpy as np

from .typing import (
    Kelvin,
    N,
    NestedSequence,
    Pascal,
    SupportsArray,
    Z,
    shape,
)

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_In = TypeVar("_In")
_Out = TypeVar("_Out")
_Dtype_t = TypeVar("_Dtype_t", bound=np.generic)
_DualArrayLike = SupportsArray[_Dtype_t] | NestedSequence[SupportsArray[_Dtype_t]] | _T | NestedSequence[_T]
_float = TypeVar("_float", np.float32, np.float64)
OPENMP_ENABLED: bool

@overload
def moist_lapse[T: np.float_](
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
def parcel_profile(): ...
@overload
def interpolate_nz[T: np.float_](
    x: np.ndarray[shape[N], np.dtype[T]],
    xp: np.ndarray[shape[Z], np.dtype[T]],
    fp: np.ndarray[shape[N, Z], np.dtype[T]],
    /,
    *,
    log_x: bool = ...,
    interp_nan: bool = ...,
) -> np.ndarray[shape[N], np.dtype[T]]: ...
@overload
def interpolate_nz[T: np.float_](
    x: np.ndarray[shape[N], np.dtype[T]],
    xp: np.ndarray[shape[Z], np.dtype[T]],
    *args: np.ndarray[shape[N, Z], np.dtype[T]],
    log_x: bool = ...,
    interp_nan: bool = ...,
) -> tuple[np.ndarray[shape[N], np.dtype[T]], ...]: ...
