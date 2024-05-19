from typing import (
    Any,
    Callable,
    Generic,
    Literal as L,
    ParamSpec,
    Sequence,
    SupportsIndex,
    TypeVar,
    overload,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import (
    Kelvin,
    N,
    NestedSequence,
    Pascal,
    SupportsArray,
    Z,
    shape,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")
_In = TypeVar("_In")
_Out = TypeVar("_Out")
_Dtype_t = TypeVar("_Dtype_t", bound=np.float_)
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


class ufunc(Generic[_P, _R, _In, _Out]):
    def __init__(self, f: Callable[_P, _R]) -> None: ...
    @property
    def __name__(self) -> str: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def identity(self) -> None: ...
    @property
    def nin(self) -> _In: ...
    @property
    def nout(self) -> _Out: ...
    @property
    def nargs(self) -> L[2]: ...
    @property
    def signature(self) -> None: ...


class _ufunc3x1(ufunc[_P, _R, L[3], L[1]]):
    @overload
    def __call__(
        self,
        __x1: float,
        __x2: float,
        __x3: float,
        /,
        out: None = ...,
        *,
        where: SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> float: ...
    @overload
    def __call__(
        self,
        __x1: NestedSequence[float] | NDArray[np.float_] | float,
        __x2: NestedSequence[float] | NDArray[np.float_],
        __x3: NestedSequence[float] | NDArray[np.float_],
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[np.float_]: ...
    @overload
    def __call__(
        self,
        __x1: NestedSequence[float] | NDArray[np.float_],
        __x2: NestedSequence[float] | NDArray[np.float_] | float,
        __x3: NestedSequence[float] | NDArray[np.float_],
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[np.float_]: ...
    def at(
        self,
        a: NDArray[Any],
        indices: SupportsArray[np.int_] | NestedSequence[SupportsArray[np.int_]] | int | NestedSequence[int],
        b: ArrayLike,
        /,
    ) -> None: ...
    def reduce(
        self,
        array: NestedSequence[float] | NDArray[np.float_],
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        out: None | NDArray[Any] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: NestedSequence[bool] | NDArray[np.bool_] = ...,
    ) -> Any: ...
    def accumulate(
        self,
        array: ArrayLike,
        axis: SupportsIndex = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...
    def reduceat(
        self,
        array: ArrayLike,
        indices: SupportsArray[np.int_] | NestedSequence[SupportsArray[np.int_]] | int | NestedSequence[int],
        axis: SupportsIndex = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        out: None | NDArray[Any] = ...,
    ) -> NDArray[Any]: ...

    # Expand `**kwargs` into explicit keyword-only arguments
    @overload
    def outer(
        self,
        A: float,
        B: float,
        C: float,
        /,
        *,
        out: None = ...,
        where: NestedSequence[bool] | NDArray[np.bool_] | None = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> Any: ...
    @overload
    def outer(  # type: ignore[misc]
        self,
        A: NestedSequence[float] | NDArray[np.float_],
        B: NestedSequence[float] | NDArray[np.float_],
        C: NestedSequence[float] | NDArray[np.float_],
        /,
        *,
        out: NDArray[Any] | tuple[NDArray[Any]] | None = ...,
        where: NestedSequence[bool] | NDArray[np.bool_] | None = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[np.float_]: ...

@_ufunc3x1
def wet_bulb_temperature(pressure:Pascal[float], temperature: Kelvin[float], dewpoint:Kelvin[float]) -> Kelvin[float]: ...
