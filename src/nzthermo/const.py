from typing import Annotated, Final

# [earth]
g: Final[Annotated[float, "(m / s^2)"]] = 9.80665
"""standard acceleration due to gravity"""
G: Final[Annotated[float, ...]] = 6.6743e-11
"""gravitational constant"""
Re: Final[Annotated[float, "m"]] = 6371008.7714
"""Earth's radius meters"""
# [gas]
R: Final[Annotated[float, " (J mol^-1 K^-1)", "R*"]] = 8.314462618
"""the molar gas constant (also known as the gas constant, universal gas constant, or ideal gas 
constant) is denoted by the symbol R or R"""
# Specific gas constant
Rd: Final[Annotated[float, "(J/kg*K)"]] = 287.04749097718457
"""constant (Dry Air) (J/kg*K)"""
Rv: Final[Annotated[float, ...]] = 461.52311572606084
"""constant (Water Vapor) (J/kg*K)"""
# [specific heat]
Cpd: Final[Annotated[float, ...]] = 1004.6662184201462
Cp = Cpd
# [latent heat]
Lv: Final[Annotated[float, ...]] = 2.50084e6  # (J kg^-1) latent heat of vaporization
"""Latent heat of vaporization (J/kg)"""
P0: Final[Annotated[float, ...]] = 1e5
T0: Final[Annotated[float, "K"]] = 273.15
"""freezing point in kelvin"""
E0: Final[Annotated[float, "Pa"]] = 611.21
"""saturation pressure at freezing in Pa"""
epsilon: Final[Annotated[float, "Rd / Rv"]] = Rd / Rv
"""`Rd / Rv`"""
