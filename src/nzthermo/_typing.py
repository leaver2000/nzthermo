from typing import Annotated, NewType, TypeVar, TypeVarTuple

import numpy as np

_T = TypeVar("_T")
Kelvin = Annotated[_T, "Kelvin"]
Pascal = Annotated[_T, "Pascal"]
Kilogram = Annotated[_T, "kg"]
Percent = Annotated[_T, "%"]
Ratio = Annotated[_T, "ratio"]

Ts = TypeVarTuple("Ts")
shape = Annotated[tuple[*Ts], "shape"]

N = NewType("N", int)
T = NewType("T", int)
Z = NewType("Z", int)
Y = NewType("Y", int)
X = NewType("X", int)

NZArray = Annotated[np.ndarray[shape[N, Z], np.dtype[np.float_]], "NZArray"]
