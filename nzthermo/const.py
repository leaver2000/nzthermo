__all__ = [
    "g",
    "G",
    "Re",
    "R",
    "Rd",
    "Rv",
    "Cpi",
    "Cpv",
    "Cpl",
    "Cpd",
    "Cvv",
    "Cvd",
    "Lf",
    "Lv",
    "epsilon",
    "P0",
    "T0",
    "E0",
]
from typing import Annotated, Final

# [earth]
g: Final[Annotated[float, "(m / s^2)"]] = 9.80665
"""standard acceleration due to gravity"""
G: Final[Annotated[float, ...]] = 6.6743e-11
"""gravitational constant"""
Re: Final[Annotated[float, "m"]] = 6371008.7714
"""Earth's radius meters"""


# [mass]

# [gas]
R: Final[Annotated[float, " (J mol^-1 K^-1)", "R*"]] = 8.314462618
"""The molar gas constant (also known as the gas constant, universal gas constant, or ideal gas 
constant) is denoted by the symbol R or R"""

# Specific gas constant
Rd: Final[Annotated[float, "(J/kg*K)"]] = 287.04749097718457
"""constant (Dry Air) (J/kg*K)"""
Rv: Final[Annotated[float, ...]] = 461.52311572606084
"""constant (Water Vapor) (J/kg*K)"""

# [temperature]
T0: Final[Annotated[float, "K"]] = 273.15
"""freezing point in kelvin"""
E0: Final[Annotated[float, "Pa"]] = 611.2
"""saturation pressure at freezing in Pa"""

# [specific heat]
Cpi: Final[Annotated[float, ...]] = 2090.0  # (J kg^-1 K^-1) specific heat of ice
Cpv: Final[Annotated[float, ...]] = 1860.078011865639  # (J kg^-1 K^-1) specific heat of water vapor
Cpl: Final[Annotated[float, ...]] = 4219.400000000001  # (J kg^-1 K^-1) specific heat of liquid water
Cpd: Final[Annotated[float, ...]] = 1004.6662184201462  # (J kg^-1 K^-1) specific heat of dry air

Cvv: Final[Annotated[float, ...]] = 1398.554896139578  # (J kg^-1 K^-1) specific heat of water vapor
Cvd: Final[Annotated[float, ...]] = 717.6187274429616  # (J kg^-1 K^-1) dry air specific heat volume

# [latent heat]
Lf: Final[Annotated[float, ...]] = 3.34e5  # (J kg^-1) latent heat of fusion
Lv: Final[Annotated[float, ...]] = 2.50084e6  # (J kg^-1) latent heat of vaporization
"""Latent heat of vaporization (J/kg)"""

# [other]
epsilon: Final[Annotated[float, "Rd / Rv"]] = Rd / Rv
P0: Final[Annotated[float, ...]] = 1e5
