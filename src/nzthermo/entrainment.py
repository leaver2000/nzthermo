import numpy as np

from . import _ufunc as uf


# @check_units("[pressure]", "[length]", "[temperature]", "[mass]/[mass]")
def moist_static_energy(
    pressure: np.ndarray,
    height_msl: np.ndarray,
    temperature: np.ndarray,
    specific_humidity: np.ndarray,
):
    """Calculate the moist static energy terms of interest.

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        specific_humidity:
            Specific humidity

    Returns:
        moist_static_energy_bar:
            Mean moist static energy from the surface to a layer
        moist_static_energy_star:
            Saturated moist static energy
    """
    N, Z = temperature.shape
    # calculate MSE_bar
    e = uf.moist_static_energy(height_msl / 100, temperature, specific_humidity)  # / 1000

    bar = np.cumsum(e, axis=1) / (np.arange(1, Z + 1).reshape(1, Z))
    # moist_static_energy_bar = moist_static_energy_bar.to("J/kg")

    # calculate MSE*

    star = uf.moist_static_energy(
        height_msl, temperature, uf.saturation_mixing_ratio(pressure, temperature)
    )
    # moist_static_energy_star = moist_static_energy_star.to("J/kg")

    return bar, star
