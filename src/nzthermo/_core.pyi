from typing import Any, Literal as L, ParamSpec, TypeVar, overload

import numpy as np

from .typing import Kelvin, N, NestedSequence, Pascal, SupportsArray, Z, shape

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_In = TypeVar("_In")
_Out = TypeVar("_Out")
_Dtype_t = TypeVar("_Dtype_t", bound=np.generic)
_DualArrayLike = (
    SupportsArray[_Dtype_t] | NestedSequence[SupportsArray[_Dtype_t]] | _T | NestedSequence[_T]
)
_float = TypeVar("_float", np.float32, np.float64)

OPENMP_ENABLED: bool = ...

T0: float = ...
E0: float = ...
Cp: float = ...
Rd: float = ...
Rv: float = ...
Lv: float = ...
P0: float = ...
Mw: float = ...
Md: float = ...
epsilon: float = ...
kappa: float = ...

@overload
def moist_lapse[T: np.floating[Any]](
    pressure: Pascal[np.ndarray[shape[N], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    reference_pressure: Pascal[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    *,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N], np.dtype[T]]]: ...
@overload
def moist_lapse[T: np.floating[Any]](
    pressure: Pascal[
        np.ndarray[shape[Z], np.dtype[T]]
        | np.ndarray[shape[L[1], Z], np.dtype[T]]
        | np.ndarray[shape[N, Z], np.dtype[T]]
    ],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    reference_pressure: Pascal[np.ndarray[N, np.dtype[np.floating[Any]]]] | None = ...,
    *,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]]: ...
@overload
def moist_lapse[T: np.floating[Any]](
    pressure: Pascal[np.ndarray[Any, np.dtype[T]]],
    temperature: Kelvin[np.ndarray[Any, np.dtype[np.floating[Any]]]],
    reference_pressure: Pascal[np.ndarray[Any, np.dtype[np.floating[Any]]]] | None = ...,
    *,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
) -> Kelvin[np.ndarray[Any, np.dtype[T]]]: ...
def parcel_profile[T: np.floating[Any]](
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[T]] | np.ndarray[shape[N, Z], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    /,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]]: ...
def parcel_profile_with_lcl[T: np.floating[Any]](
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[T]] | np.ndarray[shape[N, Z], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]],
    /,
) -> tuple[
    Pascal[np.ndarray[shape[N, Z], np.dtype[T]]],
    Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]],
    Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]],
    Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]],
]: ...
@overload
def interpolate_nz[T: np.floating[Any]](
    x: np.ndarray[shape[N], np.dtype[T]],
    xp: np.ndarray[shape[Z], np.dtype[T]],
    fp: np.ndarray[shape[N, Z], np.dtype[T]],
    /,
    *,
    log_x: bool = ...,
    interp_nan: bool = ...,
) -> np.ndarray[shape[N], np.dtype[T]]: ...
@overload
def interpolate_nz[T: np.floating[Any]](
    x: np.ndarray[shape[N], np.dtype[T]],
    xp: np.ndarray[shape[Z], np.dtype[T]],
    *args: np.ndarray[shape[N, Z], np.dtype[T]],
    log_x: bool = ...,
    interp_nan: bool = ...,
) -> tuple[np.ndarray[shape[N], np.dtype[T]], ...]: ...
def intersect[T: np.floating[Any]](
    x: np.ndarray[shape[N, Z], np.dtype[T]],
    a: np.ndarray[shape[N, Z], np.dtype[T]],
    b: np.ndarray[shape[N, Z], np.dtype[T]],
    log_x: bool = ...,
    direction: L["increasing", "decreasing"] = ...,
    increasing: bool = ...,
    bottom: bool = ...,
) -> tuple[np.ndarray[shape[N], np.dtype[T]], np.ndarray[shape[N], np.dtype[T]]]: ...
