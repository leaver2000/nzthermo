from __future__ import annotations

import sys
from typing import (
    Annotated,
    Any,
    Literal as L,
    NewType,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import numpy as np

_T = TypeVar("_T")
_DType_co = TypeVar("_DType_co", bound=np.generic, covariant=True)

Kelvin = Annotated[_T, "Kelvin"]
Pascal = Annotated[_T, "Pascal"]
Kilogram = Annotated[_T, "kg"]
Percent = Annotated[_T, "%"]
Ratio = Annotated[_T, "ratio"]

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple

    Ts = TypeVarTuple("Ts")
    shape = Annotated[tuple[*Ts], "shape"]
else:
    shape = tuple

N = NewType("N", int)
T = NewType("T", int)
Z = NewType("Z", int)
Y = NewType("Y", int)
X = NewType("X", int)

NZArray = Annotated[np.ndarray[shape[N, Z], np.dtype[np.float_]], "NZArray"]
NestedSequence: TypeAlias = "Sequence[_T | NestedSequence[_T]]"


@runtime_checkable
class SupportsArray(Protocol[_DType_co]):
    def __array__(self) -> np.ndarray[Any, np.dtype[_DType_co]]: ...


class SupportsArrayUFunc(Protocol):
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...
