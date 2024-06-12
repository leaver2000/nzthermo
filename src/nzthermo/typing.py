from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
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
Dimensionless = Annotated[_T, "dimensionless"]
Kilogram = Annotated[_T, "kg"]
Percent = Annotated[_T, "%"]
Ratio = Annotated[_T, "ratio"]


if sys.version_info >= (3, 11):
    from typing import Self  # noqa
else:
    if TYPE_CHECKING:
        from typing_extensions import Self
    else:
        Self = Any


if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack

    Ts = TypeVarTuple("Ts")
    shape = Annotated[tuple[Unpack[Ts]], "shape"]
else:
    shape = tuple

N = NewType("N", int)
T = NewType("T", int)
Z = NewType("Z", int)
Y = NewType("Y", int)
X = NewType("X", int)

NZArray = np.ndarray[shape[N, Z], np.dtype[_DType_co]]
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
