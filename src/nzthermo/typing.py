from __future__ import annotations

import sys
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    NewType,
    Protocol,
    Sequence,
    TypeAlias,
    TypeVar,
)
from typing import (
    Literal as L,
)

import numpy as np

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", bound=np.generic, covariant=True)

Kelvin = Annotated[_T, "Kelvin"]
Pascal = Annotated[_T, "Pascal"]
Meter = Annotated[_T, "meters"]
Dimensionless = Annotated[_T, "dimensionless"]


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


NestedSequence: TypeAlias = "Sequence[_T | NestedSequence[_T]]"


class SupportsArray(Protocol[_T_co]):
    def __array__(self) -> np.ndarray[Any, np.dtype[_T_co]]: ...


class SupportsArrayUFunc(Protocol):
    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...


class SupportsDType(Protocol[_T_co]):
    @property
    def dtype(self) -> np.dtype[_T_co]: ...
