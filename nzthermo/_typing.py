from typing import Annotated, TypeVar, TypeVarTuple, NewType

import numpy as np

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
N = NewType("N", int)
Z = NewType("Z", int)
shape = Annotated[tuple[*Ts], "shape"]
Kelvin = Annotated[T, "Kelvin"]
Pascal = Annotated[T, "Pascal"]
Kilogram = Annotated[T, "kg"]
Percent = Annotated[T, "%"]
Ratio = Annotated[T, "ratio"]

NZArray = Annotated[np.ndarray[shape[N, Z], np.dtype[np.float_]], "NZArray"]
