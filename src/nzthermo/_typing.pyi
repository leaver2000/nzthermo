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
from .typing import NestedSequence, SupportsArray

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_In = TypeVar("_In")
_Out = TypeVar("_Out")
_Dtype_t = TypeVar("_Dtype_t", bound=np.generic)
_DualArrayLike = SupportsArray[_Dtype_t] | NestedSequence[SupportsArray[_Dtype_t]] | _T | NestedSequence[_T]

class _ufunc(np.ufunc, Generic[_P, _R, _In, _Out]):  # type: ignore[misc]
    @overload
    def __init__(self, f: Callable[_P, _R]) -> None: ...
    @overload
    def __init__[*Ts](self, f: Callable[[*Ts], _R]) -> None: ...
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

class _ufunc1x1(_ufunc[_P, _R, L[2], L[1]]):
    @overload
    def __call__(
        self,
        __x1: float,
        /,
        out: None = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> float: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        /,
        out: None | NDArray[Any] = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[Any]: ...

class _ufunc2x1(_ufunc[_P, _R, L[2], L[1]]):
    @overload
    def __call__(
        self,
        __x1: float,
        __x2: float,
        out: None = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> _R: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[Any]: ...
    def at(
        self,
        a: NDArray[Any],
        indices: _DualArrayLike[np.int_, int],
        b: ArrayLike,
        /,
    ) -> None: ...
    def reduce(
        self,
        array: ArrayLike,
        axis: SupportsIndex | Sequence[SupportsIndex] | None = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        out: None | NDArray[Any] = ...,
        keepdims: bool = ...,
        initial: Any = ...,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
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
        indices: _DualArrayLike[np.int_, int],
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
        /,
        *,
        out: None = ...,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> float: ...
    @overload
    def outer(  # type: ignore[misc]
        self,
        A: _DualArrayLike[np.float_, float],
        B: _DualArrayLike[np.float_, float],
        /,
        *,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[np.float_] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> NDArray[Any]: ...

class _ufunc2x2(_ufunc[_P, _R, L[2], L[2]]):  # type: ignore[misc]
    @overload
    def __call__(
        self,
        __x1: float,
        __x2: float,
        out: None = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> _R: ...
    @overload
    def __call__(
        self,
        __x1: ArrayLike,
        __x2: ArrayLike,
        __out1: None | NDArray[Any] = ...,
        __out2: None | NDArray[Any] = ...,
        *,
        out: tuple[NDArray[Any], NDArray[Any]] = ...,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
        casting: L["no", "equiv", "safe", "same_kind", "unsafe"] = ...,
        order: L[None, "K", "A", "C", "F"] = ...,
        dtype: np.dtype[np.float_] | type[float] | L["float", "float32", "float64"] | None = ...,
        subok: bool = ...,
        signature: str | tuple[None | str, None | str, None | str, None | str] = ...,
        extobj: list[Any] = ...,
    ) -> tuple[NDArray[Any], NDArray[Any]]: ...

class _ufunc3x1(_ufunc[_P, _R, L[3], L[1]]):
    @overload
    def __call__(
        self,
        __x1: float,
        __x2: float,
        __x3: float,
        /,
        out: None = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
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
        __x1: _DualArrayLike[np.float_, float],
        __x2: _DualArrayLike[np.float_, float],
        __x3: _DualArrayLike[np.float_, float],
        /,
        out: None | NDArray[Any] | tuple[NDArray[Any]] = ...,
        *,
        where: (
            SupportsArray[np.bool_] | NestedSequence[SupportsArray[np.bool_]] | bool | NestedSequence[bool] | None
        ) = ...,
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
