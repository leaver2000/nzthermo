from typing import Any, Final, Literal as L, ParamSpec, TypeVar, overload

import numpy as np

from ._ufunc import pressure_vector
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

OPENMP_ENABLED: bool = ...

T0: Final[float] = ...
E0: Final[float] = ...
Cp: Final[float] = ...
Rd: Final[float] = ...
Rv: Final[float] = ...
Lv: Final[float] = ...
P0: Final[float] = ...
Mw: Final[float] = ...
Md: Final[float] = ...
epsilon: Final[float] = ...
kappa: Final[float] = ...

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
    /,
    *,
    dtype: type[T | float] | L["float32", "float64"] | None = ...,
    where: np.ndarray[shape[N], np.dtype[np.bool_]] | None = ...,
) -> Kelvin[np.ndarray[Any, np.dtype[T]]]: ...
def parcel_profile[T: np.floating[Any]](
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[T]] | np.ndarray[shape[N, Z], np.dtype[T]]],
    temperature: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    dewpoint: Kelvin[np.ndarray[shape[N], np.dtype[np.floating[Any]]]],
    /,
    *,
    where: np.ndarray[shape[N], np.dtype[np.bool_]] | None = ...,
) -> Kelvin[np.ndarray[shape[N, Z], np.dtype[T]]]: ...
def parcel_profile_with_lcl[T: np.floating[Any]](
    pressure: Pascal[
        pressure_vector[shape[Z], np.dtype[T]] | np.ndarray[shape[N, Z], np.dtype[T]]
    ],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[np.floating[Any]]]],
    /,
    *,
    where: np.ndarray[shape[N], np.dtype[np.bool_]] | None = ...,
) -> tuple[
    Pascal[pressure_vector[shape[N, Z], np.dtype[T]]],
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
