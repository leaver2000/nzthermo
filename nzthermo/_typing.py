from typing import Annotated, TypeVar, TypeVarTuple

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
N = Annotated[int, "N"]
Z = Annotated[int, "Z"]
shape = Annotated[tuple[*Ts], "shape"]
Kelvin = Annotated[T, "Kelvin"]
Pascal = Annotated[T, "Pascal"]
Kilogram = Annotated[T, "kg"]
Percent = Annotated[T, "%"]
Ratio = Annotated[T, "ratio"]
