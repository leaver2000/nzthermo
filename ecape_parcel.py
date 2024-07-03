"""Calculate the entraining CAPE (ECAPE) of a parcel"""

import math
import sys
from typing import Callable, Tuple

import metpy.calc as mpcalc
import numpy as np
import pint
from metpy.constants import dry_air_spec_heat_press, earth_gravity
from metpy.units import check_units, units
from numpy.typing import NDArray as PintList


@check_units("[pressure]", "[temperature]", "[temperature]")
def _get_parcel_profile(
    pressure: PintList,
    temperature: PintList,
    dew_point_temperature: PintList,
    parcel_func: Callable = None,
) -> PintList:
    """
    Retrieve a parcel's temperature profile.

    Args:
        pressure:
            Total atmospheric pressure
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        parcel_profile

    """

    # if surface-based, skip this process, None is default for lfc, el MetPy calcs
    if parcel_func:
        # calculate the parcel's starting temperature, then parcel temperature profile
        parcel_p, parcel_t, parcel_td, *parcel_i = parcel_func(
            pressure, temperature, dew_point_temperature
        )
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_t, parcel_td)
    else:
        parcel_profile = None

    return parcel_profile


@check_units("[pressure]", "[length]", "[temperature]", "[temperature]")
def calc_lfc_height(
    pressure: PintList,
    height_msl: PintList,
    temperature: PintList,
    dew_point_temperature: PintList,
    parcel_func: Callable,
) -> Tuple[int, pint.Quantity]:
    """
    Retrieve a parcel's level of free convection (lfc).

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        lfc:
            index of the last instance of negative buoyancy below the lfc
        lfc_z:
            height of the last instance of negative buoyancy below the lfc

    """

    # calculate the parcel's temperature profile
    parcel_profile = _get_parcel_profile(pressure, temperature, dew_point_temperature, parcel_func)

    # print("profile:", parcel_func)
    # for i in range(len(temperature)):
    #     print(i, temperature[i], parcel_profile[i].to('degC'))

    # calculate the lfc, select the appropriate index & associated height
    lfc_p, lfc_t = mpcalc.lfc(
        pressure, temperature, dew_point_temperature, parcel_temperature_profile=parcel_profile
    )

    if math.isnan(lfc_p.m):
        return None, None

    lfc_idx = (pressure - lfc_p > 0).nonzero()[0][-1]
    lfc_z = height_msl[lfc_idx]

    return lfc_idx, lfc_z


@check_units("[pressure]", "[length]", "[temperature]", "[temperature]")
def calc_el_height(
    pressure: PintList,
    height_msl: PintList,
    temperature: PintList,
    dew_point_temperature: PintList,
    parcel_func: Callable,
) -> Tuple[int, pint.Quantity]:
    """
    Retrieve a parcel's equilibrium level (el).

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        el_idx:
            index of the last instance of positive buoyancy below the el
        el_z:
            height of the last instance of positive buoyancy below the el

    """

    # calculate the parcel's temperature profile
    parcel_profile = _get_parcel_profile(pressure, temperature, dew_point_temperature, parcel_func)

    # calculate the el, select the appropriate index & associated height
    el_p, el_t = mpcalc.el(
        pressure, temperature, dew_point_temperature, parcel_temperature_profile=parcel_profile
    )

    if math.isnan(el_p.m):
        return None, None

    el_idx = (pressure - el_p > 0).nonzero()[0][-1]
    el_z = height_msl[el_idx]

    return el_idx, el_z


@check_units("[pressure]", "[speed]", "[speed]", "[length]")
def calc_sr_wind(
    pressure: PintList,
    u_wind: PintList,
    v_wind: PintList,
    height_msl: PintList,
    infl_bottom: pint.Quantity = 0 * units("m"),
    infl_top: pint.Quantity = 1000 * units("m"),
    storm_motion_type: str = "right_moving",
    sm_u: pint.Quantity = None,
    sm_v: pint.Quantity = None,
) -> pint.Quantity:
    """
    Calculate the mean storm relative (as compared to Bunkers right motion) wind magnitude in the 0-1 km AGL layer

    Modified by Amelia Urquhart to allow for custom inflow layers as well as a choice between Bunkers right, Bunkers left, and Mean Wind

    Args:
        pressure:
            Total atmospheric pressure
        u_wind:
            X component of the wind
        v_wind
            Y component of the wind
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.

    Returns:
        sr_wind:
            0-1 km AGL average storm relative wind magnitude

    """
    height_agl = height_msl - height_msl[0]
    bunkers_right, bunkers_left, mean_wind = mpcalc.bunkers_storm_motion(
        pressure, u_wind, v_wind, height_agl
    )  # right, left, mean

    u_sr = None
    v_sr = None

    if "left_moving" == storm_motion_type:
        u_sr = u_wind - bunkers_left[0]  # u-component
        v_sr = v_wind - bunkers_left[1]  # v-component
    elif "mean_wind" == storm_motion_type:
        u_sr = u_wind - mean_wind[0]  # u-component
        v_sr = v_wind - mean_wind[1]  # v-component
    elif "user_defined" == storm_motion_type and sm_u != None and sm_v != None:
        u_sr = u_wind - sm_u  # u-component
        v_sr = v_wind - sm_v  # v-component
    else:
        u_sr = u_wind - bunkers_right[0]  # u-component
        v_sr = v_wind - bunkers_right[1]  # v-component

    u_sr_1km = u_sr[np.nonzero((height_agl >= infl_bottom) & (height_agl <= infl_top))]
    v_sr_1km = v_sr[np.nonzero((height_agl >= infl_bottom) & (height_agl <= infl_top))]

    sr_wind = np.mean(mpcalc.wind_speed(u_sr_1km, v_sr_1km))

    return sr_wind


@check_units("[pressure]", "[length]", "[temperature]", "[mass]/[mass]")
def calc_mse(
    pressure: PintList, height_msl: PintList, temperature: PintList, specific_humidity: PintList
) -> Tuple[PintList, PintList]:
    """
    Calculate the moist static energy terms of interest.

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

    # calculate MSE_bar
    moist_static_energy = mpcalc.moist_static_energy(height_msl, temperature, specific_humidity)
    moist_static_energy_bar = np.cumsum(moist_static_energy) / np.arange(
        1, len(moist_static_energy) + 1
    )
    moist_static_energy_bar = moist_static_energy_bar.to("J/kg")

    # calculate MSE*
    saturation_mixing_ratio = mpcalc.saturation_mixing_ratio(pressure, temperature)
    moist_static_energy_star = mpcalc.moist_static_energy(
        height_msl, temperature, saturation_mixing_ratio
    )
    moist_static_energy_star = moist_static_energy_star.to("J/kg")

    return moist_static_energy_bar, moist_static_energy_star


@check_units("[energy]/[mass]", "[energy]/[mass]", "[temperature]")
def calc_integral_arg(moist_static_energy_bar, moist_static_energy_star, temperature) -> PintList:
    """
    Calculate the contents of the integral defined in the NCAPE equation (54).

    Args:
        moist_static_energy_bar:
            Mean moist static energy from the surface to a layer
        moist_static_energy_star:
            Saturated moist static energy
        temperature:
            Air temperature

    Returns:
        integral_arg:
            Contents of integral defined in NCAPE eqn. 54

    """

    # NCAPE eqn 54 integrand, see compute_NCAPE.m L32
    integral_arg = -(earth_gravity / (dry_air_spec_heat_press * temperature)) * (
        moist_static_energy_bar - moist_static_energy_star
    )

    return integral_arg


@check_units("[length]/[time]**2", "[length]", "[dimensionless]", "[dimensionless]")
def calc_ncape(
    integral_arg: PintList, height_msl: PintList, lfc_idx: int, el_idx: int
) -> pint.Quantity:
    """
    Calculate the buoyancy dilution potential (NCAPE)

    Args:
        integral_arg:
            Contents of integral defined in NCAPE eqn. 54
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        lfc_idx:
            Index of the last instance of negative buoyancy below the lfc
        el_idx:
            Index of the last instance of positive buoyancy below the el

    Returns:
        ncape:
            Buoyancy dilution potential of the free troposphere (eqn. 54)
    """

    # see compute_NCAPE.m L41
    ncape = np.sum(
        (0.5 * integral_arg[lfc_idx:el_idx] + 0.5 * integral_arg[lfc_idx + 1 : el_idx + 1])
        * (height_msl[lfc_idx + 1 : el_idx + 1] - height_msl[lfc_idx:el_idx])
    )

    return ncape


# Borrowed directly from ECAPE_FUNCTIONS
# ==============================================================================
# descriminator function between liquid and ice (i.e., omega defined in the
# beginning of section 2e in Peters et al. 2022)
def omega(T, T1, T2):
    return ((T - T1) / (T2 - T1)) * np.heaviside((T - T1) / (T2 - T1), 1) * np.heaviside(
        (1 - (T - T1) / (T2 - T1)), 1
    ) + np.heaviside(-(1 - (T - T1) / (T2 - T1)), 1)


def domega(T, T1, T2):
    return (np.heaviside(T1 - T, 1) - np.heaviside(T2 - T, 1)) / (T2 - T1)


# ==============================================================================


# Borrowed directly from ECAPE_FUNCTIONS
# ==============================================================================
# FUNCTION THAT CALCULATES THE SATURATION MIXING RATIO
def compute_rsat(T, p, iceflag, T1, T2):

    # THIS FUNCTION COMPUTES THE SATURATION MIXING RATIO, USING THE INTEGRATED
    # CLAUSIUS CLAPEYRON EQUATION (eq. 7-12 in Peters et al. 2022).
    # https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1

    # input arguments
    # T temperature (in K)
    # p pressure (in Pa)
    # iceflag (give mixing ratio with respect to liquid (0), combo liquid and
    # ice (2), or ice (3)
    # T1 warmest mixed-phase temperature
    # T2 coldest mixed-phase temperature

    # NOTE: most of my scripts and functions that use this function need
    # saturation mass fraction qs, not saturation mixing ratio rs.  To get
    # qs from rs, use the formula qs = (1 - qt)*rs, where qt is the total
    # water mass fraction

    # CONSTANTS
    Rd = 287.04  # %dry gas constant
    Rv = 461.5  # water vapor gas constant
    epsilon = Rd / Rv
    cp = 1005  # specific heat of dry air at constant pressure
    g = 9.81  # gravitational acceleration
    xlv = 2501000  # reference latent heat of vaporization at the triple point temperature
    xls = 2834000  # reference latent heat of sublimation at the triple point temperature
    cpv = 1870  # specific heat of water vapor at constant pressure
    cpl = 4190  # specific heat of liquid water
    cpi = 2106  # specific heat of ice
    ttrip = 273.15  # triple point temperature
    eref = 611.2  # reference pressure at the triple point temperature

    omeg = omega(T, T1, T2)
    if iceflag == 0:
        term1 = (cpv - cpl) / Rv
        term2 = (xlv - ttrip * (cpv - cpl)) / Rv
        esl = np.exp((T - ttrip) * term2 / (T * ttrip)) * eref * (T / ttrip) ** (term1)
        qsat = epsilon * esl / (p - esl)
    elif (
        iceflag == 1
    ):  # give linear combination of mixing ratio with respect to liquid and ice (eq. 20 in Peters et al. 2022)
        term1 = (cpv - cpl) / Rv
        term2 = (xlv - ttrip * (cpv - cpl)) / Rv
        esl_l = np.exp((T - ttrip) * term2 / (T * ttrip)) * eref * (T / ttrip) ** (term1)
        qsat_l = epsilon * esl_l / (p - esl_l)
        term1 = (cpv - cpi) / Rv
        term2 = (xls - ttrip * (cpv - cpi)) / Rv
        esl_i = np.exp((T - ttrip) * term2 / (T * ttrip)) * eref * (T / ttrip) ** (term1)
        qsat_i = epsilon * esl_i / (p - esl_i)
        qsat = (1 - omeg) * qsat_l + (omeg) * qsat_i
    elif iceflag == 2:  # only give mixing ratio with respect to ice
        term1 = (cpv - cpi) / Rv
        term2 = (xls - ttrip * (cpv - cpi)) / Rv
        esl = np.exp((T - ttrip) * term2 / (T * ttrip)) * eref * (T / ttrip) ** (term1)
        esl = min(esl, p * 0.5)
        qsat = epsilon * esl / (p - esl)
    return qsat


# ==============================================================================


# Borrowed directly from ECAPE_FUNCTIONS
# ==============================================================================
# FUNCTION THAT COMPUTES NCAPE
def compute_NCAPE(T0, p0, q0, z0, T1, T2, LFC, EL):

    Rd = 287.04  # %DRY GAS CONSTANT
    Rv = 461.5  # %GAS CONSTANT FOR WATEEER VAPRR
    epsilon = Rd / Rv  # %RATO OF THE TWO
    cp = 1005  # HEAT CAPACITY OF DRY AIR AT CONSTANT PRESSUREE
    gamma = Rd / cp  # POTENTIAL TEMPERATURE EXPONENT
    g = 9.81  # GRAVITATIONAL CONSTANT
    Gamma_d = g / cp  # DRY ADIABATIC LAPSE RATE
    xlv = 2501000  # LATENT HEAT OF VAPORIZATION AT TRIPLE POINT TEMPERATURE
    xls = 2834000  # LATENT HEAT OF SUBLIMATION AT TRIPLE POINT TEMPERATURE
    cpv = 1870  # HEAT CAPACITY OF WATER VAPOR AT CONSTANT PRESSURE
    cpl = 4190  # HEAT CAPACITY OF LIQUID WATER
    cpi = 2106  # HEAT CAPACITY OF ICE
    pref = 611.65  # REFERENCE VAPOR PRESSURE OF WATER VAPOR AT TRIPLE POINT TEMPERATURE
    ttrip = 273.15  # TRIPLE POINT TEMPERATURE

    # COMPUTE THE MOIST STATIC ENERGY
    MSE0 = cp * T0 + xlv * q0 + g * z0

    # COMPUTE THE SATURATED MOIST STATIC ENERGY
    rsat = compute_rsat(T0, p0, 0, T1, T2)
    qsat = (1 - rsat) * rsat
    MSE0_star = cp * T0 + xlv * qsat + g * z0

    # COMPUTE MSE0_BAR
    MSE0bar = np.zeros(MSE0.shape)
    # for iz in np.arange(0,MSE0bar.shape[0],1):
    #   MSE0bar[iz]=np.mean(MSE0[1:iz])

    MSE0bar[0] = MSE0[0]
    for iz in np.arange(1, MSE0bar.shape[0], 1):
        MSE0bar[iz] = (
            0.5
            * np.sum((MSE0[0:iz] + MSE0[1 : iz + 1]) * (z0[1 : iz + 1] - z0[0:iz]))
            / (z0[iz] - z0[0])
        )

    int_arg = -(g / (cp * T0)) * (MSE0bar - MSE0_star)
    ddiff = abs(z0 - LFC)
    mn = np.min(ddiff)
    ind_LFC = np.where(ddiff == mn)[0][0]
    ddiff = abs(z0 - EL)
    mn = np.min(ddiff)
    ind_EL = np.where(ddiff == mn)[0][0]
    # ind_LFC=max(ind_LFC);
    # ind_EL=max(ind_EL);

    NCAPE = np.maximum(
        np.nansum(
            (0.5 * int_arg[ind_LFC : ind_EL - 1] + 0.5 * int_arg[ind_LFC + 1 : ind_EL])
            * (z0[ind_LFC + 1 : ind_EL] - z0[ind_LFC : ind_EL - 1])
        ),
        0,
    )
    return NCAPE, MSE0_star, MSE0bar


# ==============================================================================


@check_units("[speed]", "[dimensionless]", "[length]**2/[time]**2", "[energy]/[mass]")
def calc_ecape_a(
    sr_wind: PintList, psi: pint.Quantity, ncape: pint.Quantity, cape: pint.Quantity
) -> pint.Quantity:
    """
    Calculate the entraining cape of a parcel

    Args:
        sr_wind:
            0-1 km AGL average storm relative wind magnitude
        psi:
            Parameter defined in eqn. 52, constant for a given equilibrium level
        ncape:
            Buoyancy dilution potential of the free troposphere (eqn. 54)
        cape:
            Convective available potential energy (CAPE, user-defined type)
    Returns:
        ecape:
            Entraining CAPE (eqn. 55)
    """

    # broken into terms for readability
    term_a = sr_wind**2 / 2.0
    term_b = (-1 - psi - (2 * psi / sr_wind**2) * ncape) / (4 * psi / sr_wind**2)
    term_c = (
        np.sqrt(
            (1 + psi + (2 * psi / sr_wind**2) * ncape) ** 2
            + 8 * (psi / sr_wind**2) * (cape - (psi * ncape))
        )
    ) / (4 * psi / sr_wind**2)

    ecape_a = term_a + term_b + term_c

    # set to 0 if negative
    return ecape_a.to("J/kg") if ecape_a >= 0 else 0


@check_units("[length]")
def calc_psi(el_z: pint.Quantity) -> pint.Quantity:
    """
    Calculate the constant psi as denoted in eqn. 52

    Args:
        el_z:
            height of the last instance of positive buoyancy below the el

    Returns:
        psi:
            Parameter defined in eqn. 52, constant for a given equilibrium level, see COMPUTE_ECAPE.m L88 (pitchfork)
    """

    # additional constants as denoted in section 4 step 1.
    sigma = 1.1 * units("dimensionless")
    alpha = 0.8 * units("dimensionless")
    l_mix = 120.0 * units("m")
    pr = (1.0 / 3.0) * units("dimensionless")  # prandtl number
    ksq = 0.18 * units("dimensionless")  # von karman constant

    psi = (ksq * alpha**2 * np.pi**2 * l_mix) / (4 * pr * sigma**2 * el_z)

    return psi


@check_units("[length]", "[pressure]", "[temperature]", "[mass]/[mass]", "[speed]", "[speed]")
def calc_ecape(
    height_msl: PintList,
    pressure: PintList,
    temperature: PintList,
    specific_humidity: PintList,
    u_wind: PintList,
    v_wind: PintList,
    cape_type: str = "most_unstable",
    undiluted_cape: pint.Quantity = None,
    inflow_bottom: pint.Quantity = 0 * units("m"),
    inflow_top: pint.Quantity = 1000 * units("m"),
    storm_motion: str = "right_moving",
    lfc: pint.Quantity = None,
    el: pint.Quantity = None,
    u_sm: pint.Quantity = None,
    v_sm: pint.Quantity = None,
) -> pint.Quantity:
    """
    Calculate the entraining CAPE (ECAPE) of a parcel

    Parameters:
    ------------
        height_msl: np.ndarray[pint.Quantity]
            Atmospheric heights at the levels given by 'pressure' (MSL)
        pressure: np.ndarray[pint.Quantity]
            Total atmospheric pressure
        temperature: np.ndarray[pint.Quantity]
            Air temperature
        specific humidity: np.ndarray[pint.Quantity]
            Specific humidity
        u_wind: np.ndarray[pint.Quantity]
            X component of the wind
        v_wind np.ndarray[pint.Quantity]
            Y component of the wind
        cape_type: str
            Variation of CAPE desired. 'most_unstable' (default), 'surface_based', or 'mixed_layer'
        undiluted_cape: pint.Quantity
            User-provided undiluted CAPE value

    Returns:
    ----------
        ecape : 'pint.Quantity'
            Entraining CAPE
    """

    cape_func = {
        "most_unstable": mpcalc.most_unstable_cape_cin,
        "surface_based": mpcalc.surface_based_cape_cin,
        "mixed_layer": mpcalc.mixed_layer_cape_cin,
    }

    parcel_func = {
        "most_unstable": mpcalc.most_unstable_parcel,
        "surface_based": None,
        "mixed_layer": mpcalc.mixed_parcel,
    }

    # calculate cape
    dew_point_temperature = mpcalc.dewpoint_from_specific_humidity(
        pressure, temperature, specific_humidity
    )

    # whether the user has not / has overidden the cape calculations
    if not undiluted_cape:
        cape, _ = cape_func[cape_type](pressure, temperature, dew_point_temperature)
    else:
        cape = undiluted_cape

    # lfc_idx = None
    lfc_z = None
    # el_idx = None
    el_z = None

    # print("cape_type:", cape_type)
    # print("parcel_func:", parcel_func[cape_type])

    if lfc is None:
        # print("doing lfc_idx as calc lfc height")
        # calculate the level of free convection (lfc) and equilibrium level (el) indexes
        _, lfc_z = calc_lfc_height(
            pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type]
        )
        _, el_z = calc_el_height(
            pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type]
        )
    else:
        # print("doing lfc_idx as np where")
        # lfc_idx = np.where(height_msl > lfc)[0][0]
        # el_idx = np.where(height_msl > el)[0][0]
        #     print(i, temperature[i], parcel_profile[i].to('degC'))
        el_z = el
        lfc_z = lfc

    # calculate the buoyancy dilution potential (ncape)
    # moist_static_energy_bar, moist_static_energy_star = calc_mse(pressure, height_msl, temperature, specific_humidity)
    # integral_arg = calc_integral_arg(moist_static_energy_bar, moist_static_energy_star, temperature)
    # ncape = calc_ncape(integral_arg, height_msl, lfc_idx, el_idx)

    ncape = compute_NCAPE(
        temperature.to("degK").magnitude,
        pressure.to("Pa").magnitude,
        specific_humidity.to("kg/kg").magnitude,
        height_msl.to("m").magnitude,
        273.15,
        253.15,
        lfc_z.to("m").magnitude,
        el_z.to("m").magnitude,
    )
    ncape = ncape[0] * units("J/kg")

    # calculate the storm relative (sr) wind
    sr_wind = calc_sr_wind(
        pressure,
        u_wind,
        v_wind,
        height_msl,
        infl_bottom=inflow_bottom,
        infl_top=inflow_top,
        storm_motion_type=storm_motion,
        sm_u=u_sm,
        sm_v=v_sm,
    )

    # calculate the entraining cape (ecape)
    psi = calc_psi(el_z)
    ecape_a = calc_ecape_a(sr_wind, psi, ncape, cape)

    return ecape_a


@check_units("[length]", "[pressure]", "[temperature]", "[mass]/[mass]", "[speed]", "[speed]")
def calc_ecape_ncape(
    height_msl: PintList,
    pressure: PintList,
    temperature: PintList,
    specific_humidity: PintList,
    u_wind: PintList,
    v_wind: PintList,
    cape_type: str = "most_unstable",
    undiluted_cape: pint.Quantity = None,
    inflow_bottom: pint.Quantity = 0 * units("m"),
    inflow_top: pint.Quantity = 1000 * units("m"),
    storm_motion: str = "right_moving",
    lfc: pint.Quantity = None,
    el: pint.Quantity = None,
    u_sm: pint.Quantity = None,
    v_sm: pint.Quantity = None,
) -> pint.Quantity:
    """
    Calculate the entraining CAPE (ECAPE) of a parcel

    Parameters:
    ------------
        height_msl: np.ndarray[pint.Quantity]
            Atmospheric heights at the levels given by 'pressure' (MSL)
        pressure: np.ndarray[pint.Quantity]
            Total atmospheric pressure
        temperature: np.ndarray[pint.Quantity]
            Air temperature
        specific humidity: np.ndarray[pint.Quantity]
            Specific humidity
        u_wind: np.ndarray[pint.Quantity]
            X component of the wind
        v_wind np.ndarray[pint.Quantity]
            Y component of the wind
        cape_type: str
            Variation of CAPE desired. 'most_unstable' (default), 'surface_based', or 'mixed_layer'
        undiluted_cape: pint.Quantity
            User-provided undiluted CAPE value

    Returns:
    ----------
        ecape : 'pint.Quantity'
            Entraining CAPE
    """

    cape_func = {
        "most_unstable": mpcalc.most_unstable_cape_cin,
        "surface_based": mpcalc.surface_based_cape_cin,
        "mixed_layer": mpcalc.mixed_layer_cape_cin,
    }

    parcel_func = {
        "most_unstable": mpcalc.most_unstable_parcel,
        "surface_based": None,
        "mixed_layer": mpcalc.mixed_parcel,
    }

    # calculate cape
    dew_point_temperature = mpcalc.dewpoint_from_specific_humidity(
        pressure, temperature, specific_humidity
    )

    # whether the user has not / has overidden the cape calculations
    if not undiluted_cape:
        cape, _ = cape_func[cape_type](pressure, temperature, dew_point_temperature)
    else:
        cape = undiluted_cape

    lfc_idx = None
    lfc_z = None
    el_idx = None
    el_z = None

    # print("cape_type:", cape_type)
    # print("parcel_func:", parcel_func[cape_type])

    if lfc == None:
        # print("doing lfc_idx as calc lfc height")
        # calculate the level of free convection (lfc) and equilibrium level (el) indexes
        lfc_idx, lfc_z = calc_lfc_height(
            pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type]
        )
        el_idx, el_z = calc_el_height(
            pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type]
        )
    else:
        # print("doing lfc_idx as np where")
        lfc_idx = np.where(height_msl > lfc)[0][0]
        el_idx = np.where(height_msl > el)[0][0]
        #     print(i, temperature[i], parcel_profile[i].to('degC'))
        el_z = el
        lfc_z = lfc

    # calculate the buoyancy dilution potential (ncape)
    # moist_static_energy_bar, moist_static_energy_star = calc_mse(pressure, height_msl, temperature, specific_humidity)
    # integral_arg = calc_integral_arg(moist_static_energy_bar, moist_static_energy_star, temperature)
    # ncape = calc_ncape(integral_arg, height_msl, lfc_idx, el_idx)

    ncape = compute_NCAPE(
        temperature.to("degK").magnitude,
        pressure.to("Pa").magnitude,
        specific_humidity.to("kg/kg").magnitude,
        height_msl.to("m").magnitude,
        273.15,
        253.15,
        lfc_z.to("m").magnitude,
        el_z.to("m").magnitude,
    )
    ncape = ncape[0] * units("J/kg")

    # calculate the storm relative (sr) wind
    sr_wind = calc_sr_wind(
        pressure,
        u_wind,
        v_wind,
        height_msl,
        infl_bottom=inflow_bottom,
        infl_top=inflow_top,
        storm_motion_type=storm_motion,
        sm_u=u_sm,
        sm_v=v_sm,
    )

    # calculate the entraining cape (ecape)
    psi = calc_psi(el_z)
    ecape_a = calc_ecape_a(sr_wind, psi, ncape, cape)

    return ecape_a, ncape


#
# AUTHOR: Amelia Urquhart (https://github.com/a-urq)
# VERSION: 1.2.2
# DATE: March 25, 2024
#


# relevant ECAPE constants
sigma = 1.6
alpha = 0.8
k2 = 0.18
L_mix = 120
Pr = 1 / 3


# @param updraftRadius              Units: Meters
# @return entrainment_rate:         Units: m^-1
def entrainment_rate(
    cape: float, ecape: float, ncape: float, vsr: float, storm_column_height: float
) -> float:
    E_A_tilde = ecape / cape
    N_tilde = ncape / cape
    vsr_tilde = vsr / np.sqrt(2 * cape)

    E_tilde = E_A_tilde - vsr_tilde**2

    entrainment_rate = (2 * (1 - E_tilde) / (E_tilde + N_tilde)) / (storm_column_height)

    return entrainment_rate


def updraft_radius(entrainment_rate: float) -> float:
    updraft_radius = np.sqrt(2 * k2 * L_mix / (Pr * entrainment_rate))

    return updraft_radius


ECAPE_PARCEL_DZ: pint.Quantity = 20 * units.meter


# Unlike the Java version, this expects arrays sorted in order of increasing height, decreasing pressure
# This is to keep in line with MetPy conventions
# Returns Tuple of { parcel_pressure, parcel_height, parcel_temperature, parcel_qv, parcel_qt }
@check_units("[pressure]", "[length]", "[temperature]", "[temperature]", "[speed]", "[speed]")
def calc_ecape_parcel(
    pressure: PintList,
    height: PintList,
    temperature: PintList,
    dewpoint: PintList,
    u_wind: PintList,
    v_wind: PintList,
    align_to_input_pressure_values: bool,
    entrainment_switch: bool = True,
    pseudoadiabatic_switch: bool = True,
    cape_type: str = "most_unstable",
    mixed_layer_depth_pressure: pint.Quantity = 100 * units("hPa"),
    mixed_layer_depth_height: pint.Quantity = None,
    storm_motion_type: str = "right_moving",
    inflow_layer_bottom: pint.Quantity = 0 * units.kilometer,
    inflow_layer_top: pint.Quantity = 1 * units.kilometer,
    cape: pint.Quantity = None,
    lfc: pint.Quantity = None,
    el: pint.Quantity = None,
    storm_motion_u: pint.Quantity = None,
    storm_motion_v: pint.Quantity = None,
    origin_pressure: pint.Quantity = None,
    origin_height: pint.Quantity = None,
    origin_temperature: pint.Quantity = None,
    origin_dewpoint: pint.Quantity = None,
) -> Tuple[pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity]:

    if cape_type not in ["most_unstable", "mixed_layer", "surface_based", "user_defined"]:
        sys.exit(
            "Invalid 'cape_type' kwarg. Valid cape_types include ['most_unstable', 'mixed_layer', 'surface_based', 'user_defined']"
        )

    if storm_motion_type not in ["right_moving", "left_moving", "mean_wind", "user_defined"]:
        sys.exit(
            "Invalid 'storm_motion_type' kwarg. Valid storm_motion_types include ['right_moving', 'left_moving', 'mean_wind', 'user_defined']"
        )

    specific_humidity = []

    for i in range(len(pressure)):
        pressure_0 = pressure[i]
        dewpoint_0 = dewpoint[i]

        q_0 = mpcalc.specific_humidity_from_dewpoint(pressure_0, dewpoint_0).magnitude

        specific_humidity.append(q_0)

        # print("amelia q0: ", pressure_0, q_0)

    specific_humidity *= units("dimensionless")

    # print(specific_humidity)

    # moist_static_energy = mpcalc.moist_static_energy(height, temperature, specific_humidity).to("J/kg")

    parcel_pressure = -1024
    parcel_height = -1024
    parcel_temperature = -1024
    parcel_dewpoint = -1024

    # have a "user_defined" switch option
    if "user_defined" == cape_type:
        if origin_pressure != None:
            parcel_pressure = origin_pressure
        else:
            parcel_pressure = pressure[0]

        if origin_height != None:
            parcel_height = origin_height
        else:
            parcel_height = height[0]

        if origin_temperature != None:
            parcel_temperature = origin_temperature
        else:
            parcel_temperature = temperature[0]

        if origin_dewpoint != None:
            parcel_dewpoint = origin_dewpoint
        else:
            parcel_dewpoint = dewpoint[0]
    elif "most_unstable" == cape_type:
        parcel_pressure, parcel_temperature, parcel_dewpoint, mu_idx = mpcalc.most_unstable_parcel(
            pressure, temperature, dewpoint
        )
        parcel_height = height[mu_idx]
    elif "mixed_layer" == cape_type:
        env_potential_temperature = mpcalc.potential_temperature(pressure, temperature)
        env_specific_humidity = mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)

        env_idxs_to_include_in_average = None

        if mixed_layer_depth_pressure != None:
            mixed_layer_top_pressure = pressure[0] - mixed_layer_depth_pressure
            env_idxs_to_include_in_average = np.where(pressure >= mixed_layer_top_pressure)[0]
        elif mixed_layer_depth_height != None:
            mixed_layer_top_height = height[0] + mixed_layer_depth_height
            env_idxs_to_include_in_average = np.where(height <= mixed_layer_top_height)[0]
            pass
        else:
            mixed_layer_depth_pressure = 100 * units("hPa")
            mixed_layer_top_pressure = pressure[0] - mixed_layer_depth_pressure
            env_idxs_to_include_in_average = np.where(pressure >= mixed_layer_top_pressure)[0]

        avg_potential_temperature_sum = 0.0
        avg_specific_humidity_sum = 0.0
        for i in range(len(env_idxs_to_include_in_average)):
            avg_potential_temperature_sum += env_potential_temperature[
                env_idxs_to_include_in_average[i]
            ]
            avg_specific_humidity_sum += env_specific_humidity[env_idxs_to_include_in_average[i]]

        avg_potential_temperature = avg_potential_temperature_sum / len(
            env_idxs_to_include_in_average
        )
        avg_specific_humidity = avg_specific_humidity_sum / len(env_idxs_to_include_in_average)

        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = mpcalc.temperature_from_potential_temperature(
            parcel_pressure, avg_potential_temperature
        )
        parcel_dewpoint = mpcalc.dewpoint_from_specific_humidity(
            parcel_pressure, parcel_temperature, avg_specific_humidity
        )
    elif "surface_based" == cape_type:
        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = temperature[0]
        parcel_dewpoint = dewpoint[0]
    else:
        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = temperature[0]
        parcel_dewpoint = dewpoint[0]

    # print("in house cape/el calc:", cape, el, entrainment_switch)
    if (cape == None or lfc == None or el == None) and entrainment_switch == True:
        # print("-- using in-house cape --")
        undiluted_parcel_profile = calc_ecape_parcel(
            pressure,
            height,
            temperature,
            dewpoint,
            u_wind,
            v_wind,
            align_to_input_pressure_values,
            False,
            pseudoadiabatic_switch,
            cape_type,
            mixed_layer_depth_pressure,
            mixed_layer_depth_height,
            storm_motion_type,
            inflow_layer_bottom,
            inflow_layer_top,
            origin_pressure=origin_pressure,
            origin_height=origin_height,
            origin_temperature=origin_temperature,
            origin_dewpoint=origin_dewpoint,
        )

        undiluted_parcel_profile_z = undiluted_parcel_profile[1]
        undiluted_parcel_profile_T = undiluted_parcel_profile[2]
        undiluted_parcel_profile_qv = undiluted_parcel_profile[3]
        undiluted_parcel_profile_qt = undiluted_parcel_profile[4]

        undil_cape, _, undil_lfc, undil_el = custom_cape_cin_lfc_el(
            undiluted_parcel_profile_z,
            undiluted_parcel_profile_T,
            undiluted_parcel_profile_qv,
            undiluted_parcel_profile_qt,
            height,
            temperature,
            specific_humidity,
        )
        # cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile)
        # el = calc_el_height(pressure, height, temperature, dewpoint, parcel_func[cape_type])[1]

        if cape == None:
            cape = undil_cape

        if lfc == None:
            lfc = undil_lfc

        if el == None:
            el = undil_el

    # print("el: ", el)
    # print("parcel_height: ", parcel_height)

    # print(height)
    # print(pressure)
    # print(temperature)
    # print(specific_humidity)
    # print(u_wind)
    # print(v_wind)
    # print(cape_type)
    # print("post in-house cape:", cape)

    if entrainment_switch == True:
        if cape <= 0:
            if align_to_input_pressure_values:
                pressure_raw = [None] * len(pressure)
                height_raw = [None] * len(pressure)
                temperature_raw = [None] * len(pressure)
                qv_raw = [None] * len(pressure)
                qt_raw = [None] * len(pressure)

                return (pressure_raw, height_raw, temperature_raw, qv_raw, qt_raw)
            else:
                pressure_raw = [None]
                height_raw = [None]
                temperature_raw = [None]
                qv_raw = [None]
                qt_raw = [None]

                return (pressure_raw, height_raw, temperature_raw, qv_raw, qt_raw)

    entr_rate = None

    if entrainment_switch:
        # print("amelia calc ecape")
        # print("lfc:", lfc)
        # print("el:", el)
        ecape, ncape = calc_ecape_ncape(
            height,
            pressure,
            temperature,
            specific_humidity,
            u_wind,
            v_wind,
            cape_type,
            cape,
            inflow_bottom=inflow_layer_bottom,
            inflow_top=inflow_layer_top,
            storm_motion=storm_motion_type,
            lfc=lfc,
            el=el,
            u_sm=storm_motion_u,
            v_sm=storm_motion_v,
        )
        vsr = calc_sr_wind(
            pressure,
            u_wind,
            v_wind,
            height,
            inflow_layer_bottom,
            inflow_layer_top,
            storm_motion_type,
            sm_u=storm_motion_u,
            sm_v=storm_motion_v,
        )
        storm_column_height = el - parcel_height

        cape = cape.to("joule / kilogram")
        ecape = ecape.to("joule / kilogram")
        ncape = ncape.to("joule / kilogram")
        vsr = vsr.to("meter / second")
        storm_column_height = storm_column_height.to("meter")

        # print("amelia ecape env profile")
        # for i in range(len(height)):
        #     print(height[i], pressure[i], temperature[i].to('degK'), specific_humidity[i], u_wind[i], v_wind[i])

        # print("amelia cape: ", cape)
        # print("amelia eil0: ", inflow_layer_bottom)
        # print("amelia eil1: ", inflow_layer_top)
        # print("amelia psi: ", calc_psi(storm_column_height))
        # print("amelia ecape: ", ecape)
        # print("amelia vsr: ", vsr)
        # print("amelia storm_column_height: ", storm_column_height)

        epsilon = entrainment_rate(
            cape.magnitude,
            ecape.magnitude,
            ncape.magnitude,
            vsr.magnitude,
            storm_column_height.magnitude,
        )

        # print("amelia ur inputs:", cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)
        # print("amelia ur:", r)
        # print("amelia eps:", epsilon)

        entr_rate = epsilon / units.meter
    else:
        entr_rate = 0 / units.meter

    # print("updr: ", r, " m")
    # print("entr: ", entr_rate)

    parcel_temperature = parcel_temperature.to("degK")
    parcel_dewpoint = parcel_dewpoint.to("degK")

    parcel_qv = mpcalc.specific_humidity_from_dewpoint(parcel_pressure, parcel_dewpoint)
    parcel_qt = parcel_qv

    pressure_raw = []
    height_raw = []
    temperature_raw = []
    qv_raw = []
    qt_raw = []

    pressure_raw.append(parcel_pressure)
    height_raw.append(parcel_height)
    temperature_raw.append(parcel_temperature)
    qv_raw.append(parcel_qv)
    qt_raw.append(parcel_qt)

    # print("parcel z/q/q0:", parcel_height, parcel_qv, specific_humidity[0])
    # print("specific humidity: ", specific_humidity)
    # print("amelia entr rate:", entr_rate)

    prate = 1 / ECAPE_PARCEL_DZ
    if not pseudoadiabatic_switch:
        prate *= 0

    dqt_dz = 0 / ECAPE_PARCEL_DZ

    while parcel_pressure >= pressure[-1]:
        env_temperature = linear_interp(height, temperature, parcel_height)
        # parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
        # parcel_height += ECAPE_PARCEL_DZ

        parcel_saturation_qv = (1 - parcel_qt) * r_sat(parcel_temperature, parcel_pressure, 1)

        if parcel_saturation_qv > parcel_qv:
            parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
            parcel_height += ECAPE_PARCEL_DZ

            env_temperature = linear_interp(height, temperature, parcel_height)
            env_qv = linear_interp(height, specific_humidity, parcel_height)

            dT_dz = unsaturated_adiabatic_lapse_rate(
                parcel_temperature, parcel_qv, env_temperature, env_qv, entr_rate
            )
            dqv_dz = -entr_rate * (parcel_qv - env_qv)

            # q_sat = specific_humidity_from_dewpoint(parcel_pressure, parcel_temperature)

            # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, env_temperature.m, env_qv.m, parcel_pressure.m, entr_rate.m, "dqv_dz", dqv_dz.m, (-entr_rate * (parcel_qv - env_qv)).m, parcel_qv.m, env_qv.m)
            # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, env_temperature.m, env_qv.m, parcel_pressure.m, parcel_height.m, entr_rate.m, "q_sat", q_sat)

            parcel_temperature += dT_dz * ECAPE_PARCEL_DZ
            parcel_qv += dqv_dz * ECAPE_PARCEL_DZ
            # parcel_qt += dqt_dz * ECAPE_PARCEL_DZ
            parcel_qt = parcel_qv

            # print("amelia qv:", parcel_qv)

            parcel_dewpoint = dewpoint_from_specific_humidity(parcel_pressure, parcel_qv)
        else:
            parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
            parcel_height += ECAPE_PARCEL_DZ

            env_temperature = linear_interp(height, temperature, parcel_height)
            env_qv = linear_interp(height, specific_humidity, parcel_height)

            dT_dz = None

            if pseudoadiabatic_switch:
                dT_dz = saturated_adiabatic_lapse_rate(
                    parcel_temperature,
                    parcel_qt,
                    parcel_pressure,
                    env_temperature,
                    env_qv,
                    entr_rate,
                    prate,
                    qt_entrainment=dqt_dz,
                )
            else:
                dT_dz = saturated_adiabatic_lapse_rate(
                    parcel_temperature,
                    parcel_qt,
                    parcel_pressure,
                    env_temperature,
                    env_qv,
                    entr_rate,
                    prate,
                )

            new_parcel_qv = (1 - parcel_qt) * r_sat(parcel_temperature, parcel_pressure, 1).to(
                "kg/kg"
            )

            if pseudoadiabatic_switch:
                dqt_dz = (new_parcel_qv - parcel_qv) / ECAPE_PARCEL_DZ
            else:
                dqt_dz = -entr_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv)

            if parcel_pressure < 40000 * units("Pa") and parcel_pressure > 20000 * units("Pa"):
                pass
                # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, parcel_qt.m, parcel_pressure.m, entr_rate.m, prate.m, "dqt_dz", dqt_dz.m, (-entr_rate * (parcel_qt - env_qv)).m, parcel_qt.m, env_qv.m)
                # print("amelia new_parcel_qv:", new_parcel_qv, parcel_qt, )
                # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, parcel_qt.m, parcel_pressure.m, entr_rate.m, prate.m, "dqt_dz", dqt_dz.m, -entr_rate * (parcel_qt - env_qv), parcel_qt, env_qv)
                # print("amelia dT/dz:", dT_dz.m, parcel_qt.m, parcel_pressure.m, "dqt_dz", dqt_dz.m, (-entr_rate * (parcel_qt - env_qv)).m,  (-prate * (parcel_qt - parcel_qv)).m, parcel_qt.m, env_qv.m, prate.m)
                # print("amelia dqt_dz:", dqt_dz, parcel_pressure)
                # print("amelia qv qt qv0:", parcel_qv, parcel_qt, env_qv)
            # print("dqt_dz - 1:", -entr_rate * (parcel_qv - env_qv))
            # print("dqt_dz - 2:", entr_rate)
            # print("dqt_dz - 3:", (parcel_qv))
            # print("dqt_dz - 4:", (env_qv))
            # print("dqt_dz - 5:", (parcel_qv - env_qv))
            # print("dqt_dz - 6:", - prate * (parcel_qt - parcel_qv))
            # print("dqt_dz - 7:", -prate)
            # print("dqt_dz - 8:", (parcel_qt - parcel_qv))

            parcel_temperature += dT_dz * ECAPE_PARCEL_DZ
            parcel_qv = new_parcel_qv

            if pseudoadiabatic_switch:
                parcel_qt = parcel_qv
            else:
                dqt_dz = -entr_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv)
                parcel_qt += dqt_dz * ECAPE_PARCEL_DZ

            # print("qv:", parcel_qv)
            # print("qt & dqt:", parcel_qt, dqt_dz)

            if parcel_qt < parcel_qv:
                parcel_qv = parcel_qt

        pressure_raw.append(parcel_pressure)
        height_raw.append(parcel_height)
        temperature_raw.append(parcel_temperature)
        qv_raw.append(parcel_qv)
        qt_raw.append(parcel_qt)

        # print(parcel_pressure, parcel_height, parcel_temperature, parcel_qv, parcel_qt)

    pressure_units = pressure_raw[-1].units
    height_units = height_raw[-1].units
    temperature_units = temperature_raw[-1].units
    qv_units = qv_raw[-1].units
    qt_units = qt_raw[-1].units

    # print(pressure_units)
    # print(height_units)
    # print(temperature_units)
    # print(dewpoint_units)

    # print(pressure_raw[0:3])
    # print(height_raw[0:3])
    # print("input_height", height[0:30])
    # print("height_raw", height_raw[0:30])
    # print("temperature_raw", temperature_raw[0:30])
    # print("dewpoint_raw", dewpoint_raw[0:30])

    pressure_nondim = [None] * len(pressure_raw)
    height_nondim = [None] * len(height_raw)
    temperature_nondim = [None] * len(temperature_raw)
    qv_nondim = [None] * len(qv_raw)
    qt_nondim = [None] * len(qt_raw)

    for i in range(len(height_raw)):
        pressure_nondim[i] = pressure_raw[i].magnitude
        height_nondim[i] = height_raw[i].magnitude
        temperature_nondim[i] = temperature_raw[i].magnitude
        qv_nondim[i] = qv_raw[i].magnitude
        qt_nondim[i] = qt_raw[i].magnitude

    # makes it work ok with sounderpy
    if align_to_input_pressure_values:
        pressure_nondim_aligned = []
        height_nondim_aligned = []
        temperature_nondim_aligned = []
        qv_nondim_aligned = []
        qt_nondim_aligned = []

        # print(temperature_nondim)
        # print(dewpoint_nondim)
        # print(linear_interp(height, temperature_nondim, 1000 * units.meter))
        # print("above is debugging lin interp")

        for i in range(len(height)):
            input_pressure = pressure[i]
            input_height = height[i]

            # print("searching for interp_t at height", input_height)

            if input_height >= height_raw[0]:
                new_t = rev_linear_interp(pressure_raw, temperature_nondim, input_pressure)
                new_qv = rev_linear_interp(pressure_raw, qv_nondim, input_pressure)
                new_qt = rev_linear_interp(pressure_raw, qt_nondim, input_pressure)

                pressure_nondim_aligned.append(input_pressure.magnitude)
                height_nondim_aligned.append(input_height.magnitude)
                temperature_nondim_aligned.append(new_t)
                qv_nondim_aligned.append(new_qv)
                qt_nondim_aligned.append(new_qt)
            else:
                pressure_nondim_aligned.append(input_pressure.magnitude)
                height_nondim_aligned.append(input_height.magnitude)
                temperature_nondim_aligned.append(temperature[i].to("degK").magnitude)
                qv_nondim_aligned.append(specific_humidity[i].magnitude)
                qt_nondim_aligned.append(specific_humidity[i].magnitude)

        pressure_nondim = pressure_nondim_aligned
        height_nondim = height_nondim_aligned
        temperature_nondim = temperature_nondim_aligned
        qv_nondim = qv_nondim_aligned
        qt_nondim = qt_nondim_aligned

    pressure_qty: pint.Quantity = pressure_nondim * pressure_units
    height_qty: pint.Quantity = height_nondim * height_units
    temperature_qty: pint.Quantity = temperature_nondim * temperature_units
    qv_qty: pint.Quantity = qv_nondim * qv_units
    qt_qty: pint.Quantity = qt_nondim * qt_units

    # print(pressure_qty[0], height_qty[0], temperature_qty[0], qv_qty[0], qt_qty[0])

    return (pressure_qty, height_qty, temperature_qty, qv_qty, qt_qty)


molar_gas_constant = 8.314 * units.joule / units.kelvin / units.mole
avg_molar_mass = 0.029 * units.kilogram / units.mole
g = 9.81 * units.meter / units.second / units.second

c_pd = 1005 * units("J/kg") / units("K")
c_pv = 1875 * units("J/kg") / units("K")


def specific_heat_capacity_of_moist_air(specific_humidity: pint.Quantity):
    c_p = specific_humidity * c_pv + (1 - specific_humidity) * c_pd

    return c_p


def pressure_at_height(
    ref_pressure: pint.Quantity,
    height_above_ref_pressure: pint.Quantity,
    temperature: pint.Quantity,
) -> pint.Quantity:
    temperature = temperature.to("degK")
    height_above_ref_pressure = height_above_ref_pressure.to("m")

    scale_height = (molar_gas_constant * temperature) / (avg_molar_mass * g)

    scale_height = scale_height.magnitude * units.meter

    # print(ref_pressure)
    # print(height_above_ref_pressure)
    # print(scale_height)
    # print(-height_above_ref_pressure.magnitude / scale_height.magnitude)
    # print(math.exp(-height_above_ref_pressure.magnitude / scale_height.magnitude))

    return ref_pressure * math.exp(-height_above_ref_pressure.magnitude / scale_height.magnitude)


def linear_interp(
    input_arr: PintList, output_arr: PintList, input: pint.Quantity, debug: bool = False
) -> pint.Quantity:
    if input < input_arr[0]:
        if debug:
            print("tree 1")
        return output_arr[0]
    elif input >= input_arr[-1]:
        if debug:
            print("tree 2")
        return output_arr[-1]
    else:
        for i in range(len(input_arr) - 1):
            input_1 = input_arr[i]
            input_2 = input_arr[i + 1]

            if input == input_1:
                if debug:
                    print("tree 3 - 1")
                    print(input)
                    print(input_1)

                return output_arr[i]
            elif input < input_2:
                if debug:
                    print("tree 3 - 2")
                output_1 = output_arr[i]
                output_2 = output_arr[i + 1]

                weight_1 = (input_2 - input) / (input_2 - input_1)
                weight_2 = (input - input_1) / (input_2 - input_1)

                return output_1 * weight_1 + output_2 * weight_2
            else:
                continue

    return None  # case should not be reached


def rev_linear_interp(
    input_arr: PintList, output_arr: PintList, input: pint.Quantity, debug: bool = False
) -> pint.Quantity:
    if input > input_arr[0]:
        if debug:
            print("tree 1")
        return output_arr[0]
    elif input <= input_arr[-1]:
        if debug:
            print("tree 2")
        return output_arr[-1]
    else:
        for i in range(len(input_arr) - 1):
            input_1 = input_arr[i]
            input_2 = input_arr[i + 1]

            if input == input_1:
                if debug:
                    print("tree 3 - 1")
                    print(input)
                    print(input_1)

                return output_arr[i]
            elif input > input_2:
                if debug:
                    print("tree 3 - 2")
                output_1 = output_arr[i]
                output_2 = output_arr[i + 1]

                weight_1 = (input_2 - input) / (input_2 - input_1)
                weight_2 = (input - input_1) / (input_2 - input_1)

                return output_1 * weight_1 + output_2 * weight_2
            else:
                continue

    return None  # case should not be reached


c_p = 1005 * units.joule / units.kilogram / units.kelvin


# allows for true adiabatic CAPE calculation and also bypasses metpy weirdness even for pseudoadiabatic
# only intended for undiluted
def custom_cape_cin_lfc_el(
    parcel_height: pint.Quantity,
    parcel_temperature: pint.Quantity,
    parcel_qv: pint.Quantity,
    parcel_qt: pint.Quantity,
    env_height: pint.Quantity,
    env_temperature: pint.Quantity,
    env_qv: pint.Quantity,
    integration_bound_lower: pint.Quantity = None,
    integration_bound_upper: pint.Quantity = None,
) -> pint.Quantity:
    parcel_density_temperature = density_temperature(parcel_temperature, parcel_qv, parcel_qt)
    env_density_temperature = density_temperature(env_temperature, env_qv, env_qv)

    integrated_positive_buoyancy = 0 * units("J/kg")
    integrated_negative_buoyancy = 0 * units("J/kg")
    lfc = None
    el = None

    env_mse = mpcalc.moist_static_energy(env_height, env_temperature, env_qv)

    # for i in range(len(env_mse)):
    #     print(i, env_height[i], env_temperature[i], env_mse[i])

    height_min_mse_idx = np.where(env_mse == np.min(env_mse))[0][
        0
    ]  # FIND THE INDEX OF THE HEIGHT OF MINIMUM MSE
    height_min_mse = env_height[height_min_mse_idx]

    for i in range(len(parcel_height) - 1, 0, -1):
        z0 = parcel_height[i]
        dz = parcel_height[i] - parcel_height[i - 1]

        if integration_bound_lower != None:
            if z0 < integration_bound_lower:
                continue

        if integration_bound_upper != None:
            if z0 > integration_bound_upper:
                continue

        T_rho_0 = linear_interp(env_height, env_density_temperature, z0)

        T_rho = parcel_density_temperature[i]

        buoyancy = g * (T_rho - T_rho_0) / T_rho_0

        # print(z0, buoyancy)

        ### MARK FIRST POSITIVE BUOYANCY HEIGHT AS EL
        if buoyancy > 0 * g.units and el == None:
            el = z0

        ### IF LFC IS NOT YET REACHED, INTEGRATE ALL POSITIVE BUOYANCY
        if buoyancy > 0 * g.units and lfc == None:
            integrated_positive_buoyancy += buoyancy * dz

        # print("z0 < height_min_mse:", z0, height_min_mse)
        ### MARK FIRST NEGATIVE BUOYANCY HEIGHT BELOW MIN_MSE AS LFC
        if z0 < height_min_mse and buoyancy < 0 * g.units:
            integrated_negative_buoyancy += buoyancy * dz

            if lfc == None:
                lfc = z0

    if lfc == None:
        lfc = env_height[0]

    return integrated_positive_buoyancy, integrated_negative_buoyancy, lfc, el


def specific_humidity_from_dewpoint(pressure, dewpoint):
    vapor_pressure_ = vapor_pressure(dewpoint)

    return specific_humidity(pressure, vapor_pressure_) * units("dimensionless")


def vapor_pressure(dewpoint):
    dewpoint_nondim = dewpoint.to("K").magnitude

    e0 = 611 * units("Pa")
    t0 = 273.15

    return e0 * math.exp(
        latent_heat_of_vaporization / water_vapor_gas_constant * (1 / t0 - 1 / dewpoint_nondim)
    )


def specific_humidity(pressure: pint.Quantity, vapor_pressure: pint.Quantity) -> pint.Quantity:
    pressure_nondim = pressure.to("Pa").magnitude
    vapor_pressure_nondim = vapor_pressure.to("Pa").magnitude

    water_vapor_density = absolute_humidity(vapor_pressure_nondim, 280)  # kg m^-3
    air_density = dry_air_density(pressure_nondim - vapor_pressure_nondim, 280)  # kg m^-3

    # print("d_wv:", water_vapor_density)
    # print("d_da:", air_density)

    return water_vapor_density / (water_vapor_density + air_density)


dry_air_gas_constant = 287
water_vapor_gas_constant = 461.5
latent_heat_of_vaporization = 2500000


def absolute_humidity(vapor_pressure, temperature):
    water_vapor_density = vapor_pressure / (water_vapor_gas_constant * temperature)

    return water_vapor_density


def dry_air_density(dry_air_pressure, temperature):
    dry_air_density = dry_air_pressure / (dry_air_gas_constant * temperature)

    return dry_air_density


def dewpoint_from_specific_humidity(pressure, specific_humidity):
    vapor_pressure = vapor_pressure_from_specific_humidity(
        pressure.to("Pa").magnitude, specific_humidity
    )
    dewpoint = dewpoint_from_vapor_pressure(vapor_pressure)
    return dewpoint


def vapor_pressure_from_specific_humidity(pressure, specific_humidity):
    water_vapor_gas_constant = 461.5  # J/(kg·K)
    dry_air_gas_constant = 287  # J/(kg·K)

    numerator = specific_humidity * pressure
    denominator_term = (
        1 / water_vapor_gas_constant
        + specific_humidity / dry_air_gas_constant
        - specific_humidity / water_vapor_gas_constant
    )

    vapor_pressure = numerator / (dry_air_gas_constant * denominator_term)

    return vapor_pressure


def dewpoint_from_vapor_pressure(vapor_pressure):
    e0 = 611  # Pascals
    t0 = 273.15  # Kelvins
    latent_heat_of_vaporization = 2.5e6  # J/kg

    vapor_pres_nondim = vapor_pressure

    # print(1 / t0)
    # print((461.5 / latent_heat_of_vaporization))
    # print(vapor_pressure)
    # print(e0)
    # print(math.log(vapor_pres_nondim / e0))

    dewpoint_reciprocal = 1 / t0 - (461.5 / latent_heat_of_vaporization) * math.log(
        vapor_pres_nondim / e0
    )

    return (1 / dewpoint_reciprocal) * units("K")


g = 9.81 * units("m") / units("s") / units("s")
c_pd = 1005 * units("J/kg") / units("K")
c_pv = 1870 * units("J/kg") / units("K")
c_pl = 4190 * units("J/kg") / units("K")
c_pi = 2106 * units("J/kg") / units("K")
R_d = 287.04 * units("J/kg") / units("K")
R_v = 461.5 * units("J/kg") / units("K")
L_v_trip = 2501000 * units("J/kg")
L_i_trip = 333000 * units("J/kg")
T_trip = 273.15 * units("K")

phi = R_d / R_v


@check_units("[temperature]", "[dimensionless]", "[dimensionless]")
def density_temperature(temperature, qv, qt) -> pint.Quantity:
    if isinstance(temperature, list):
        if temperature[0] == None:
            len_t = len(temperature)

            return [None] * len_t

    t_rho = temperature * (1 - qt + qv / phi)

    return t_rho


# Equation 19 in Peters et. al. 2022 (https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml)
@check_units("[temperature]", "[dimensionless]", "[temperature]", "[dimensionless]")
def unsaturated_adiabatic_lapse_rate(
    temperature_parcel: pint.Quantity,
    qv_parcel: pint.Quantity,
    temperature_env: pint.Quantity,
    qv_env: pint.Quantity,
    entrainment_rate: pint.Quantity,
) -> pint.Quantity:
    temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env)

    density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qv_parcel)
    density_temperature_env = density_temperature(temperature_env, qv_env, qv_env)

    buoyancy = g * (density_temperature_parcel - density_temperature_env) / density_temperature_env

    c_pmv = (1 - qv_parcel) * c_pd + qv_parcel * c_pv

    # print("amelia cpmv:", c_pmv)
    # print("amelia B:", buoyancy)
    # print("amelia eps:", temperature_entrainment)

    term_1 = -g / c_pd
    term_2 = 1 + (buoyancy / g)
    term_3 = c_pmv / c_pd

    dTdz = term_1 * (term_2 / term_3) + temperature_entrainment

    return dTdz


@check_units("[temperature]", "[temperature]", "[temperature]")
def ice_fraction(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp):
    if temperature >= warmest_mixed_phase_temp:
        return 0
    elif temperature <= coldest_mixed_phase_temp:
        return 1
    else:
        return (1 / (coldest_mixed_phase_temp - warmest_mixed_phase_temp)) * (
            temperature - warmest_mixed_phase_temp
        )


@check_units("[temperature]", "[temperature]", "[temperature]")
def ice_fraction_deriv(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp):
    if temperature >= warmest_mixed_phase_temp:
        return 0 / units("K")
    elif temperature <= coldest_mixed_phase_temp:
        return 0 / units("K")
    else:
        return 1 / (coldest_mixed_phase_temp - warmest_mixed_phase_temp)


vapor_pres_ref = 611.2 * units("Pa")


# borrowed and adapted from ECAPE_FUNCTIONS
def r_sat(
    temperature,
    pressure,
    ice_flag: int,
    warmest_mixed_phase_temp: pint.Quantity = 273.15 * units("K"),
    coldest_mixed_phase_temp: pint.Quantity = 253.15 * units("K"),
):
    if ice_flag == 2:
        term_1 = (c_pv - c_pi) / R_v
        term_2 = (L_v_trip - T_trip * (c_pv - c_pi)) / R_v
        esi = (
            np.exp((temperature - T_trip) * term_2 / (temperature * T_trip))
            * vapor_pres_ref
            * (temperature / T_trip) ** (term_1)
        )
        q_sat = phi * esi / (pressure - esi)

        return q_sat
    elif ice_flag == 1:
        omega = ice_fraction(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp)

        qsat_l = r_sat(temperature, pressure, 0)
        qsat_i = r_sat(temperature, pressure, 2)

        q_sat = (1 - omega) * qsat_l + (omega) * qsat_i

        return q_sat
    else:
        term_1 = (c_pv - c_pl) / R_v
        term_2 = (L_v_trip - T_trip * (c_pv - c_pl)) / R_v
        esi = (
            np.exp((temperature - T_trip) * term_2 / (temperature * T_trip))
            * vapor_pres_ref
            * (temperature / T_trip) ** (term_1)
        )
        q_sat = phi * esi / (pressure - esi)

        return q_sat


# Equation 24 in Peters et. al. 2022 (https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml)
# @check_units('[temperature]', '[dimensionless]',  '[dimensionless]', '[temperature]', '[dimensionless]', '[dimensionless]')
def saturated_adiabatic_lapse_rate(
    temperature_parcel: pint.Quantity,
    qt_parcel: pint.Quantity,
    pressure_parcel: pint.Quantity,
    temperature_env: pint.Quantity,
    qv_env: pint.Quantity,
    entrainment_rate: pint.Quantity,
    prate: pint.Quantity,
    warmest_mixed_phase_temp: pint.Quantity = 273.15 * units("K"),
    coldest_mixed_phase_temp: pint.Quantity = 253.15 * units("K"),
    qt_entrainment: pint.Quantity = None,
) -> pint.Quantity:
    omega = ice_fraction(temperature_parcel, warmest_mixed_phase_temp, coldest_mixed_phase_temp)
    d_omega = ice_fraction_deriv(
        temperature_parcel, warmest_mixed_phase_temp, coldest_mixed_phase_temp
    )

    q_vsl = (1 - qt_parcel) * r_sat(temperature_parcel, pressure_parcel, 0)
    q_vsi = (1 - qt_parcel) * r_sat(temperature_parcel, pressure_parcel, 2)

    qv_parcel = (1 - omega) * q_vsl + omega * q_vsi

    temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env)
    qv_entrainment = -entrainment_rate * (qv_parcel - qv_env)

    if qt_entrainment == None:
        qt_entrainment = -entrainment_rate * (qt_parcel - qv_env) - prate * (qt_parcel - qv_parcel)

    q_condensate = qt_parcel - qv_parcel
    ql_parcel = q_condensate * (1 - omega)
    qi_parcel = q_condensate * omega

    c_pm = (1 - qt_parcel) * c_pd + qv_parcel * c_pv + ql_parcel * c_pl + qi_parcel * c_pi

    density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qt_parcel)
    density_temperature_env = density_temperature(temperature_env, qv_env, qv_env)

    buoyancy = g * (density_temperature_parcel - density_temperature_env) / density_temperature_env

    L_v = L_v_trip + (temperature_parcel - T_trip) * (c_pv - c_pl)
    L_i = L_i_trip + (temperature_parcel - T_trip) * (c_pl - c_pi)

    L_s = L_v + omega * L_i

    Q_vsl = q_vsl / (phi - phi * qt_parcel + qv_parcel)
    Q_vsi = q_vsi / (phi - phi * qt_parcel + qv_parcel)

    Q_M = (1 - omega) * (q_vsl) / (1 - Q_vsl) + omega * (q_vsi) / (1 - Q_vsi)
    L_M = (1 - omega) * L_v * (q_vsl) / (1 - Q_vsl) + omega * (L_v + L_i) * (q_vsi) / (1 - Q_vsi)
    R_m0 = (1 - qv_env) * R_d + qv_env * R_v

    term_1 = buoyancy
    term_2 = g
    term_3 = ((L_s * Q_M) / (R_m0 * temperature_env)) * g
    term_4 = (c_pm - L_i * (qt_parcel - qv_parcel) * d_omega) * temperature_entrainment
    term_5 = L_s * (
        qv_entrainment + qv_parcel / (1 - qt_parcel) * qt_entrainment
    )  # - (q_vsi - q_vsl) * d_omega) # peters left this end bit out

    term_6 = c_pm
    term_7 = (L_i * (qt_parcel - qv_parcel) - L_s * (q_vsi - q_vsl)) * d_omega
    term_8 = (L_s * L_M) / (R_v * temperature_parcel * temperature_parcel)

    return -(term_1 + term_2 + term_3 - term_4 - term_5) / (term_6 - term_7 + term_8)
