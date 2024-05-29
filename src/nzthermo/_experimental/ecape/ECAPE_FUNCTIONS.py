from __future__ import annotations

from typing import Final

import numpy as np
from scipy.special import lambertw, sici

Rd: Final = 287.04  #           dry gas constant
Rv: Final = 461.5  #            water vapor gas constant
epsilon: Final = Rd / Rv  #     %RATO OF THE TWO
g: Final = 9.81  #              gravitational acceleration
c_d: Final = 0.2  #             DRAG COEFFICIENT ON A SPHERE
Lambda: Final = 0.6  #          RATIO OF ASCENT RATE OF THERMAL TO ITS MAX W
alpha: Final = 0.8  #           ASSUMED RATIO OF HORIZONTALLY AVERAGED W TO HORIZONTAL MAX OF W AT A GIVEN LEVEL
cp: Final = 1005  #             specific heat of dry air at constant pressure
xlv: Final = 2501000  #         reference latent heat of vaporization at the triple point temperature
xls: Final = 2834000  #         reference latent heat of sublimation at the triple point temperature
cpv: Final = 1870.0  #            specific heat of water vapor at constant pressure
cpl: Final = 4190.0  #            specific heat of liquid water
cpi: Final = 2106.0  #            specific heat of ice
ttrip: Final = 273.15  #        triple point temperature
eref: Final = 611.2  #          reference pressure at the triple point temperature


def comp_cdwave(F):
    gamma_em = np.euler_gamma
    return 4 * (F * np.sin(2 / F) - (F**2) * (np.sin(1 / F) ** 2) - cosint(2 / F) + np.log(2 / F) - 1 + gamma_em)


def cosint(x):
    si, ci = sici(x)
    return ci


def omega(T: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """
    discriminator function between liquid and ice (i.e., omega defined in the beginning of section
    2e in Peters et al. 2022
    )"""
    return ((T - T1) / (T2 - T1)) * np.heaviside((T - T1) / (T2 - T1), 1) * np.heaviside(
        (1 - (T - T1) / (T2 - T1)), 1
    ) + np.heaviside(-(1 - (T - T1) / (T2 - T1)), 1)


def domega(T: np.ndarray, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    return (np.heaviside(T1 - T, 1) - np.heaviside(T2 - T, 1)) / (T2 - T1)


# -------------------------------------------------------------------------------------------------
def compute_rsat(T, p, iceflag, T1, T2):
    # FUNCTION THAT CALCULATES THE SATURATION MIXING RATIO
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


# -------------------------------------------------------------------------------------------------
def drylift(T, qv, T0, qv0, fracent):
    # LAPSE RATE FOR AN UNSATURATED PARCEL
    # CONSTANTS
    # Rd = 287.04  # dry gas constant
    # Rv = 461.5  # water vapor gas constant
    # cp = 1005  # specific heat of dry air at constant pressure
    # g = 9.81  # gravitational acceleration
    # cpv = 1870  # specific heat of water vapor at constant pressure

    cpmv = (1 - qv) * cp + qv * cpv
    B = g * ((T - T0) / T0 + (Rv / Rd - 1) * (qv - qv0))
    eps = -fracent * (T - T0)
    gamma_d = -(g + B) / cpmv + eps
    return gamma_d


# -------------------------------------------------------------------------------------------------
def compute_LCL(T, qv, p):
    # LIFTED CONDENSATION LEVEL USING THE ROMPS 2017 FORMULA
    # # CONSTANTS
    # Rd = 287.04  # %dry gas constant
    # Rv = 461.5  # water vapor gas constant
    # # epsilon = Rd / Rv
    # cp = 1005  # specific heat of dry air at constant pressure
    # g = 9.81  # gravitational acceleration
    # xlv = 2501000  # reference latent heat of vaporization at the triple point temperature
    # # xls = 2834000  # reference latent heat of sublimation at the triple point temperature
    # cpv = 1870  # specific heat of water vapor at constant pressure
    # cpl = 4190  # specific heat of liquid water
    # # cpi = 2106  # specific heat of ice
    # ttrip = 273.15  # triple point temperature
    # eref = 611.2  # reference pressure at the triple point temperature
    # cv = cp - Rd
    # cvv = cpv - Rv

    cpm = (1 - qv) * cp + qv * cpv
    Rm = (1 - qv) * Rd + qv * Rv

    a = cpm / Rm + (cpl - cpv) / Rv
    b = -(xlv - (cpv - cpl) * ttrip) / (Rv * T)
    c = b / a

    r_sat = compute_rsat(T, p, 0, 273.15, 253.15)
    q_sat = r_sat / (1 + r_sat)
    RH = qv / q_sat
    arg1 = RH ** (1 / a)
    arg2 = c * np.exp(1) ** c
    arg3 = lambertw(arg1 * arg2, k=-1)
    T_LCL = c * T / arg3
    # P_LCL = p * (T_LCL / T) ** (cpm / Rm)
    Z_LCL = (cpm / g) * (T - T_LCL)

    return Z_LCL


# -------------------------------------------------------------------------------------------------
def compute_LCL_NUMERICAL(T, qv, p, dz):
    # LIFTED CONDENSATION LEVEL USING NUMERICAL INTEGRATION
    # CONSTANTS
    # NOTE, WE ARE ASSUMING ZERO BUOYANCY (I.E., WELL MIXED PBL)
    # Rd = 287.04  # %dry gas constant
    # Rv = 461.5  # water vapor gas constant
    # # epsilon = Rd / Rv
    # # cp = 1005  # specific heat of dry air at constant pressure
    # g = 9.81  # gravitational acceleration
    # # xlv = 2501000  # reference latent heat of vaporization at the triple point temperature
    # # xls = 2834000  # reference latent heat of sublimation at the triple point temperature
    # # cpv = 1870  # specific heat of water vapor at constant pressure
    # # cpl = 4190  # specific heat of liquid water
    # # cpi = 2106  # specific heat of ice
    # # ttrip = 273.15  # triple point temperature
    # # eref = 611.2  # reference pressure at the triple point temperature
    # # cv = cp - Rd
    # cvv = cpv - Rv

    nfound_LCL = True

    # zon = 0
    ind_hgt = 0
    Ton = T
    Qon = qv
    Pon = p
    while nfound_LCL:
        ind_hgt = ind_hgt + 1
        Ton = Ton + dz * drylift(Ton, Qon, Ton, Qon, 0)
        Pon = Pon - dz * (Pon * g) / (Rd * (1 + (Rv / Rd - 1) * Qon) * Ton)
        rsat = compute_rsat(Ton, Pon, 0, 273.15, 253.15)
        qsat = rsat / (1 + rsat)
        if Qon >= qsat:
            nfound_LCL = False
    Z_LCL = ind_hgt * dz

    return Z_LCL


# -------------------------------------------------------------------------------------------------
def moislif(T, qv, qvv, qvi, p0, T0, q0, qt, fracent, prate, T1, T2):
    # LAPSE RATE FOR A SATURATED PARCEL
    # CONSTANTS
    # Rd = 287.04  # dry gas constant
    # Rv = 461.5  # water vapor gas constant
    # epsilon = Rd / Rv
    # cp = 1005  # specific heat of dry air at constant pressure
    # g = 9.81  # gravitational acceleration
    # xlv = 2501000  # reference latent heat of vaporization at the triple point temperature
    # xls = 2834000  # reference latent heat of sublimation at the triple point temperature
    # cpv = 1870  # specific heat of water vapor at constant pressure
    # cpl = 4190  # specific heat of liquid water
    # cpi = 2106  # specific heat of ice
    # ttrip = 273.15  # triple point temperature

    qt = max(qt, 0.0)
    qv = max(qv, 0.0)

    OMEGA = omega(T, T1, T2)
    dOMEGA = domega(T, T1, T2)

    cpm = (1 - qt) * cp + qv * cpv + (1 - OMEGA) * (qt - qv) * cpl + OMEGA * (qt - qv) * cpi
    Lv = xlv + (T - ttrip) * (cpv - cpl)
    Li = (xls - xlv) + (T - ttrip) * (cpl - cpi)
    Rm0 = (1 - q0) * Rd + q0 * Rv

    T_rho = T * (1 - qt + qv / epsilon)
    T_rho0 = T0 * (1 - q0 + q0 / epsilon)
    B = g * (T_rho - T_rho0) / (T_rho0)

    Qvsl = qvv / (epsilon - epsilon * qt + qv)
    Qvsi = qvi / (epsilon - epsilon * qt + qv)
    Q_M = (1 - OMEGA) * qvv / (1 - Qvsl) + OMEGA * qvi / (1 - Qvsi)
    L_M = Lv * (1 - OMEGA) * qvv / (1 - Qvsl) + (Lv + Li) * OMEGA * qvi / (1 - Qvsi)

    eps_T = -fracent * (T - T0)
    eps_qv = -fracent * (qv - q0)
    eps_qt = -fracent * (qt - q0) - prate * (qt - qv)
    term1 = -B

    term2 = -Q_M * (Lv + Li * OMEGA) * g / (Rm0 * T0)

    term3 = -g
    term4 = (cpm - Li * (qt - qv) * dOMEGA) * eps_T
    term5 = (Lv + Li * OMEGA) * (eps_qv + (qv / (1 - qt)) * eps_qt)

    term6 = cpm
    term7 = -Li * (qt - qv) * dOMEGA
    term8 = (Lv + Li * OMEGA) * (-dOMEGA * (qvv - qvi) + (1 / (Rv * (T**2))) * (L_M))
    gamma_m = (term1 + term2 + term3 + term4 + term5) / (term6 + term7 + term8)
    return gamma_m


# -------------------------------------------------------------------------------------------------
def lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2):
    # FUNCTION THAT LIFTS A PARCEL
    # [T_lif,Qv_lif,Qt_lif,B_lif]

    # this function computes lifted parcel properties using the unsaturated
    # and saturated lapse rate formulas from (Peters et al. 2022)
    # https://doi-org.ezaccess.libraries.psu.edu/10.1175/JAS-D-21-0118.1

    # input arguments
    # T0: sounding profile of temperature (in K)
    # p0: sounding profile of pressure (in Pa)
    # q0: sounding profile of water vapor mass fraction (in kg/kg)
    # start_loc: index of the parcel starting location (set to 1 for the
    # lowest: level in the sounding)
    # fracent: fractional entrainment rate (in m^-1)

    # output arguments
    # T_lif: lifted parcel temperature
    # Qv_lif: lifted parcel water vapor mass fraction
    # Qt_lif: lifted parcel total water mass fraction
    # B_lif: Lifted parcel buoyancy, computed using Eq. B6 in (Peters et al.
    # 2022) (accounts for virtual temperature and loading effects)

    # prate: precipitation rate (in m^-1) large values make parcel more
    # pseudoadiabatic, small values make parcel more adiabatic.  I usually
    # just set it to 0 to get an adiabatic parce

    # z0: sounding profile of height above ground level (first height should
    # be 0 m)
    # T1 warmest mixed-phase temperature
    # T2 coldest mixed-phase temperature

    # CONSTANTS
    Rd = 287.04  # dry gas constant
    Rv = 461.5  # water vapor gas constant
    # epsilon = Rd / Rv
    g = 9.81  # gravitational acceleration
    cp = 1005
    xlv = 2501000  # reference latent heat of vaporization at the triple point temperature

    # ESTIMATE THE MOIST STATIC ENERGY (MSE)
    MSE = cp * T0 + xlv * q0 + g * z0
    mn_hgt = np.where(MSE == np.min(MSE))  # FIND THE INDEX OF THE HEIGHT OF MINIMUM MSE

    # discriminator function between liquid and ice (i.e., omega defined in the  beginning of
    # section 2e in Peters et al. 2022)
    T_lif = np.zeros(T0.shape) * np.nan  # temperature of the lifted parcel
    Qv_lif = np.zeros(T0.shape) * np.nan  # water vapor mass fraction of the lifted parcel
    Qt_lif = np.ones(T0.shape) * np.nan  # total water mass fraction of the lifted parcel

    if start_loc > 0:
        T_lif[0 : start_loc + 1] = T0[0 : start_loc + 1]  # set initial values to that of the environment
        Qv_lif[0 : start_loc + 1] = q0[0 : start_loc + 1]  # set initial values to that of the environment
        Qt_lif[0 : start_loc + 1] = Qv_lif[0 : start_loc + 1]  # set initial values to that of the environment
    else:
        T_lif[0] = T0[0]  # set initial values to that of the environment
        Qv_lif[0] = q0[0]  # set initial values to that of the environment
        Qt_lif[0] = Qv_lif[0]  # set initial values to that of the environment

    q_sat_prev = 0
    B_run = 0
    iz = start_loc
    #
    # for iz in np.arange(start_loc+1,z0.shape[0]):
    #
    #
    # I REVISED THIS A BIT.  TO MAKE THE CODE FASTER, I HAVE THE CALCULATION CUT OUT WHEN THE INTEGRATED NEGATIVE BUOYANCY ("BRUN")
    # BECOMES MORE NEGATIVE THAN THAN THE TOTAL INTEGRATED POSITIVE BUOYANCY.  I RESTRICT THIS TO ONLY HAPPEN AFTER WE HAVE PASSED
    # THE HEIGHT OF MINIMUM MSE.  UNCOMMENT THE FOR LOOP ABOVE AND COMMENT OUT THE WHILE LOOP IF YOU JUST WANT TO INTEGRATE TO THE TOP OF THE SOUNDING.
    # THE +25 PART IN THE WHILE STATEMENT IS A PAD ON B_RUN (THE NEGATIVE CAPE HAS TO BE 25 J/KG LESS THAN THE POSITIVE CAPE TO KILL THE LOOP)
    # while iz<(z0.shape[0])-1 and (z0[iz]<z0[mn_hgt] or (B_run+25)>0):
    while iz < (z0.shape[0]) - 1 and (z0[iz] < z0[mn_hgt] or (B_run + 250) > 0):
        iz = iz + 1
        q_sat = (1 - Qt_lif[iz - 1]) * compute_rsat(T_lif[iz - 1], p0[iz - 1], 1, T1, T2)
        # if we are unsaturated, go up at the unsaturated adiabatic lapse rate (eq. 19 in Peters et al. 2022)
        if Qv_lif[iz - 1] < q_sat:
            T_lif[iz] = T_lif[iz - 1] + (z0[iz] - z0[iz - 1]) * drylift(
                T_lif[iz - 1], Qv_lif[iz - 1], T0[iz - 1], q0[iz - 1], fracent
            )
            Qv_lif[iz] = Qv_lif[iz - 1] - (z0[iz] - z0[iz - 1]) * fracent * (Qv_lif[iz - 1] - q0[iz - 1])
            Qt_lif[iz] = Qv_lif[iz]
            q_sat = (1 - Qt_lif[iz]) * compute_rsat(T_lif[iz], p0[iz], 1, T1, T2)

            # if we hit saturation, split the vertical step into two stages.  The first stage advances at the saturated lapse rate to the saturation point, and the second stage completes the grid step at the moist lapse rate
            if Qv_lif[iz] >= q_sat:
                OMEGA = omega(T_lif[iz - 1], T1, T2)
                dOMEGA = domega(T_lif[iz - 1], T1, T2)
                satrat = (Qv_lif[iz] - q_sat_prev) / (q_sat - q_sat_prev)
                dz_dry = satrat * (z0[iz] - z0[iz - 1])
                dz_wet = (1 - satrat) * (z0[iz] - z0[iz - 1])

                T_halfstep = T_lif[iz - 1] + dz_dry * drylift(
                    T_lif[iz - 1], Qv_lif[iz - 1], T0[iz - 1], q0[iz - 1], fracent
                )
                Qv_halfstep = Qv_lif[iz - 1] - dz_dry * fracent * (Qv_lif[iz - 1] - q0[iz - 1])
                Qt_halfstep = Qv_lif[iz]
                p_halfstep = p0[iz - 1] * satrat + p0[iz] * (1 - satrat)
                T0_halfstep = T0[iz - 1] * satrat + T0[iz] * (1 - satrat)
                Q0_halfstep = q0[iz - 1] * satrat + q0[iz] * (1 - satrat)

                T_lif[iz] = T_halfstep + dz_wet * moislif(
                    T_halfstep,
                    Qv_halfstep,
                    (1 - Qt_halfstep) * compute_rsat(T_halfstep, p_halfstep, 0, T1, T2),
                    (1 - Qt_halfstep) * compute_rsat(T_halfstep, p_halfstep, 2, T1, T2),
                    p_halfstep,
                    T0_halfstep,
                    Q0_halfstep,
                    Qt_halfstep,
                    fracent,
                    prate,
                    T1,
                    T2,
                )

                Qt_lif[iz] = Qt_lif[iz - 1] - (z0[iz] - z0[iz - 1]) * fracent * (Qt_halfstep - Q0_halfstep)
                Qv_lif[iz] = (1 - Qt_lif[iz]) * compute_rsat(T_lif[iz], p0[iz], 1, T1, T2)

                if Qt_lif[iz] < Qv_lif[iz]:
                    Qv_lif[iz] = Qt_lif[iz]

            q_sat_prev = q_sat

        else:  # if we are already at saturation, just advance upward using the saturated lapse rate (eq. 24 in Peters et al. 2022)
            OMEGA = omega(T_lif[iz - 1], T1, T2)
            dOMEGA = domega(T_lif[iz - 1], T1, T2)

            T_lif[iz] = T_lif[iz - 1] + (z0[iz] - z0[iz - 1]) * moislif(
                T_lif[iz - 1],
                Qv_lif[iz - 1],
                (1 - Qt_lif[iz - 1]) * compute_rsat(T_lif[iz - 1], p0[iz - 1], 0, T1, T2),
                (1 - Qt_lif[iz - 1]) * compute_rsat(T_lif[iz - 1], p0[iz - 1], 2, T1, T2),
                p0[iz - 1],
                T0[iz - 1],
                q0[iz - 1],
                Qt_lif[iz - 1],
                fracent,
                prate,
                T1,
                T2,
            )

            Qt_lif[iz] = Qt_lif[iz - 1] - (z0[iz] - z0[iz - 1]) * (
                fracent * (Qt_lif[iz - 1] - q0[iz - 1]) + prate * (Qt_lif[iz - 1] - Qv_lif[iz - 1])
            )
            Qv_lif[iz] = (1 - Qt_lif[iz]) * compute_rsat(T_lif[iz], p0[iz], 1, T1, T2)

            if Qt_lif[iz] < Qv_lif[iz]:
                Qv_lif[iz] = Qt_lif[iz]

        B_run = B_run + (
            g * T_lif[iz] * (1 + (Rv / Rd) * Qv_lif[iz] - Qt_lif[iz]) / (T0[iz] * (1 + (Rv / Rd) * q0[iz] - q0[iz]))
            - g
        ) * (z0[iz] - z0[iz - 1])

    T_rho_lif = T_lif * (1 + (Rv / Rd) * Qv_lif - Qt_lif)
    T_0_lif = T0 * (1 + (Rv / Rd - 1) * q0)
    # T_rho_lif=T_lif*(1 - Qt_lif + Qv_lif)/( 1 + (epsilon - 1)/( ( epsilon*(1 - Qt_lif)/Qv_lif - 1) ) )
    # T_0_lif=T0/( 1 + (epsilon - 1)/( ( epsilon*(1 - q0)/q0 - 1) ) )

    B_lif = g * (T_rho_lif - T_0_lif) / T_0_lif

    return T_lif, Qv_lif, Qt_lif, B_lif


# ==============================================================================
# FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPE_AND_CIN(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2):
    # [CAPE,CIN,LFC,EL]

    # this function computes CAPE and CIN

    # input arguments
    # T0: sounding profile of temperature (in K)
    # p0: sounding profile of pressure (in Pa)
    # q0: sounding profile of water vapor mass fraction (in kg/kg)
    # start_loc: index of the parcel starting location (set to 1 for the
    # lowest: level in the sounding)
    # fracent: fractional entrainment rate (in m^-1)

    # CONSTANTS
    Rd = 287.04  # dry gas constant
    Rv = 461.5  # water vapor gas constant
    epsilon = Rd / Rv
    g = 9.81  # gravitational acceleration

    # compute lifted parcel buoyancy
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2)

    if np.nanmax(B_lif) > 0:
        # CAPE will be the total integrated positive buoyancy
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]
        B_pos[np.where(B_pos < 0)] = 0
        dz = z0[1 : z0.shape[0]] - z0[0 : z0.shape[0] - 1]
        CAPE = np.nansum(0.5 * B_pos[0 : z0.shape[0] - 1] * dz + 0.5 * B_pos[1 : z0.shape[0]] * dz)

        # CIN will be the total negative buoyancy below the height of maximum
        # buoyancy
        B_neg = np.zeros(B_lif.shape)
        B_neg[:] = B_lif[:]
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif == mx)
        imx = imx[0][0]
        B_neg[0:imx] = np.minimum(B_neg[0:imx], 0)
        B_neg[imx : z0.shape[0]] = 0
        CIN = np.nansum(0.5 * B_neg[0 : z0.shape[0] - 1] * dz + 0.5 * B_neg[1 : z0.shape[0]] * dz)

        # LFC will be the last instance of negative buoyancy before the
        # continuous interval that contains the maximum in buoyancy
        fneg = np.where(B_lif < 0)
        fneg = fneg[0]
        inn = np.where(fneg < imx)
        inn = inn[0]
        fneg = fneg[inn]
        if len(fneg) > 0:
            LFC = 0.5 * z0[np.max(fneg)] + 0.5 * z0[np.max(fneg) + 1]
        else:
            LFC = z0[start_loc]

        # EL will be last instance of positive buoyancy
        fpos = np.where(B_lif > 0)
        fpos = fpos[0]
        EL = 0.5 * z0[np.max(fpos)] + 0.5 * z0[np.max(fpos) + 1]
    else:
        CAPE = 0
        CIN = 0
        LFC = np.nan
        EL = np.nan

    return CAPE, CIN, LFC, EL


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
        MSE0bar[iz] = 0.5 * np.sum((MSE0[0:iz] + MSE0[1 : iz + 1]) * (z0[1 : iz + 1] - z0[0:iz])) / (z0[iz] - z0[0])

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
# FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_VSR(z0, u0, v0):
    # compute 0-1 km storm-relative flow (V_SR) using the storm motion
    # estimate of Bunkers et al. (2000)
    # https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2

    f6000 = np.where(z0 <= 6000)[0]
    meanx = np.mean(u0[f6000])
    meany = np.mean(v0[f6000])

    f0500 = np.where(z0 <= 500)[0]
    lowx = np.mean(u0[f0500])
    lowy = np.mean(v0[f0500])

    f560 = np.where(np.logical_and(z0 <= 6000, z0 >= 5500))[0]
    highx = np.mean(u0[f560])
    highy = np.mean(v0[f560])
    BK_SHRx = highx - lowx
    BK_SHRy = highy - lowy
    BK_mag = np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx = BK_SHRx / BK_mag
    BK_diry = BK_SHRy / BK_mag
    BK_orthx = BK_diry * 7.5
    BK_orthy = -BK_dirx * 7.5

    SR_mean_u = u0 - meanx
    SR_mean_v = v0 - meany
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(v0.shape)
    dudz[1 : dudz.shape[0] - 1] = (u0[2 : dudz.shape[0]] - u0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dudz[0] = 2 * dudz[1] - dudz[2]
    dvdz[1 : dudz.shape[0] - 1] = (v0[2 : dudz.shape[0]] - v0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dvdz[0] = 2 * dvdz[1] - dvdz[2]
    f1000 = np.where(z0 <= 1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000] * dvdz[f1000] + SR_mean_v[f1000] * dudz[f1000]) * 1000.0)

    propfac = min(SRH_mean / 150, 1)
    propfac = 1

    C_x = meanx + propfac * BK_orthx
    C_y = meany + propfac * BK_orthy

    u_sr = u0 - C_x
    v_sr = v0 - C_y

    f1000 = np.where(z0 <= 1000)[0]
    V_SR = np.nanmean(np.sqrt(u_sr[f1000] ** 2 + v_sr[f1000] ** 2))
    return V_SR, C_x, C_y


# ==============================================================================
def compute_ETILDE(CAPE, NCAPE, V_SR, EL, L):
    # THESE ARE A BUNCH OF CONSTANT PARAMTERS SET FOR THE ECAPE CALCULATION
    H = EL
    l = L / H
    sigma = 1.1
    alpha = 0.8
    Pr = 1 / 3  # PRANDTL NUMBER
    ksq = 0.18  # VON KARMAN CONSTANT
    pitchfork = ksq * (alpha**2) * (np.pi**2) * L / (4 * Pr * (sigma**2) * H)
    vsr_tilde = V_SR / np.sqrt(2 * CAPE)
    N_tilde = NCAPE / CAPE

    # EQUATION SOLVES FOR THE NONDIMENSIONAL ECAPE (E_TILDE_A IN THE PAPER)
    E_tilde = vsr_tilde**2 + (
        -1
        - pitchfork
        - (pitchfork / (vsr_tilde**2)) * N_tilde
        + np.sqrt(
            (1 + pitchfork + (pitchfork / (vsr_tilde**2)) * N_tilde) ** 2
            + (4 * (pitchfork / (vsr_tilde**2)) * (1 - pitchfork * N_tilde))
        )
    ) / (2 * pitchfork / (vsr_tilde**2))

    E_tilde_ = E_tilde - vsr_tilde**2

    varepsilon = 2 * ((1 - E_tilde_) / (E_tilde_ + N_tilde)) / (EL)

    # eps = 2*ksq*L/(EL*Pr)

    # Rm2 = ( (alpha*np.pi/(sigma) )**2 )*( E_tilde/vsr_tilde + 1)
    # Radius =  EL*Rm2**(-1/2)
    # varepsilon = 2*ksq*L/(Pr*Radius**2 )

    # Radius=Radius/2

    # varepsilon = 0.65*eps*(alpha**2)*(np.pi**2)*E_tilde/(4*(sigma**2)*EL*(vsr_tilde**2 ) ) #THIS IS THE FRACTIONAL ENTRAINMENT RATE
    Radius = np.sqrt(2 * ksq * L / (Pr * varepsilon))

    return E_tilde, varepsilon, Radius


# ==============================================================================
def CI_model(T0, p0, q0, z0, u0, v0, T1, T2, radrng, itmax, L, prate_global):
    # THIS FUNCTION EXECUTES THE "PROGRESSIVE ROOTING" TOY MODEL DESCRIBED BY PETERS ET AL. 2022A
    # https://journals.ametsoc.org/view/journals/atsc/79/6/JAS-D-21-0145.1.xml

    # NOTE, A VAREITY OF THINGS HAVE CHANGED SINCE THAT PUBLICATION.  I WILL TRY TO
    # POINT SPECIFIC EQUATIONS HERE TO EQUATION NUMBERS IN THE PUBLCIATION.  I WILL PROBABLY
    # CREATE A TECHNICAL DOCUMENT TO DESCRIBE THESE CHANGES SOMETIME SOON.  STAY TUNED...

    # THE FUNCTION TAKES AS INPUT:
    # T0, profile of temperature (K)
    # p0, profile of pressure (Pa)
    # q0, profile of specific humidity (kg/kg)
    # z0, profile of height above ground level (m)
    # u0, profile of u wind (m/s)
    # v0, profile of v wind (m/s)
    # T1, temperature at which freezing begins in parcel calculations (I usually set to 273.15 K)
    # T2, temperature at which freezing ends in the parcel calculation (K).  This will control the temperature
    # range over which mixed-phase occurs.  I usually set to 253.15 k
    # radrng, a vector containing the initial radii we are going to test.  A reasonable
    # choice here is a range from 100 m to 6000 m at intervals of 100 m (np.arange(100,6000,100))
    # itmax, the number of iterations (I usually set to 20)
    # L, the mixing length (I usually set to 250 m)
    # prate_global, the precipitation loss inverse length scale (km^(-1)).  Larger values make the
    # parcel more pseudoadiabatic, smaller values make it more adiabatic.

    # STANDARD THERMODYNAMIC CONSTANTS
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

    # PARAMTERS UNIQUE TO THE CI MODEL
    alpha = 0.8  # ASSUMED RATIO OF HORIZONTALLY AVERAGED W TO HORIZONTAL MAX OF W AT A GIVEN LEVEL
    Pr = 1 / 3  # PRANDTL NUMBER
    ksq = 0.18  # VON KARMAN CONSTANT
    start_loc = 0  # STARTING HEIGHT OF THE AIR PARCEL WE ARE LIFTING
    sig = 0.5  # RATIO OF THE HEIGHT OF WMAX TO EQUILBIRIUM LEVEL HEIGHT (SHOULD PROBABLY SET THIS TO 1)
    rfac = 1 / 4  # RELAXATION FACTOR FOR MODEL INTEGRATION.  SMALLER VALUE GIVES A SMOOTHER SOLUTION

    # WE WILL NEED THE DENSITY PROFILE TO COMPUTE THE STORM-RELATIVE WIND LATER
    rho0 = p0 / (Rd * T0 * (1 + (Rv / Rd - 1) * q0))

    # TIME SERIES OF QUANTITIES OUTPUTTED FROM THE CI MODEL
    R_TS = np.zeros((radrng.shape[0], itmax))  # RADIUS OF THE UPDRAFT
    H_TS = np.zeros((radrng.shape[0], itmax))  # EL HEIGHT
    W_TS = np.zeros((radrng.shape[0], itmax))  # MAX VERTICAL VELOCITY
    VSR_TS = np.zeros((radrng.shape[0], itmax))  # STORM-RELATIVE FLOW

    # INITIAL CONDITION ON RADIUS: SET TO R0
    R_TS[:, 0] = radrng

    # dudz=np.zeros(u0.shape)
    # dvdz=np.zeros(v0.shape)
    # dudz[1:dudz.shape[0]-1]= ( u0[2:dudz.shape[0]]-u0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    # dudz[0]=2*dudz[1]-dudz[2]
    # dvdz[1:dudz.shape[0]-1]= ( v0[2:dudz.shape[0]]-v0[0:dudz.shape[0]-2] )/( z0[2:dudz.shape[0]]-z0[0:dudz.shape[0]-2] )
    # dvdz[0]=2*dvdz[1]-dvdz[2]
    # SHR_mag = np.sqrt(dudz**2 + dvdz**2)

    # IN THE FUTRE, WE'LL PROBABLY WANT TO COMPUTE THE DENSITY WEIGHTED STORM-RELATIVE FLOW, LIKE IN THE ECAPE THEORY
    # UDCAPE,UDCIN,UDLFC,UDEL=compute_CAPE_AND_CIN(T0,p0,q0,start_loc,0,prate_global,z0,T1,T2)

    # PARAMETERS FOR CI MODEL
    for it in np.arange(0, itmax - 1, 1):  # LOOP THROUGH THE SPECIFIED NUMBER OF ITERATIONS
        for ir in np.arange(0, radrng.shape[0], 1):  # LOOP THROUGH EACH OF THE STARTING RADII
            R_on = R_TS[ir, it]  # STORE THE RADIUS (IN M)
            #
            fracent = (
                2 * ksq * L / (Pr * (R_on**2))
            )  # USE RADIUS TO COMPUTE FRACTION ENTRAINMENT RATE WITH EQ. XX IN XX

            # WHEN COMPUTING THE VERTICAL PROFILE OF KINETIC ENERGY, THE LOWER BOUNDARY CONDITION IS THAT A PARCEL
            # BEGINS WITH THE KINETIC ENERGY OF THE INFLOW.  THIS MEANS WE HAVE TO GIVE THE VERTICAl VELOCITY
            # FUNCTION THE STORM RELATIVE WIND.
            if it == 0:
                # AT THE FIRST TIME STEP, WE WONT HAVE THE STORM RELATIVE WIND YET SO WE'LL MAKE AN AD-HOC ESTIMATE
                # V_SR = 15*R_on/5000 #5.0
                V_SR = 20 * R_on / 5000  # 5.0
            else:
                # AT LATER TIMES, WE JUST USE THE STORM RELATIVE FLOW FROM THE PREVIOUS TIME STEP
                V_SR = VSR_TS[ir, it - 1]

            # GET THE MAXIMUM VERTICAL VELOCITY PROFILE FOR A RISING CLOUD THERMAL
            CAPE, LFC, EL, B_pos = compute_w(
                T0, p0, q0, start_loc, fracent, prate_global, z0, T1, T2, R_on, u0, v0, V_SR
            )
            # NOW THE VERTICAL VELOCITY AT THE BASE OF THE THERMAL, WHICH WILL EXPERIENCE A HIGHER ENTRAINMENT RATE
            CAPE2, LFC2, EL2, null = compute_w(
                T0, p0, q0, start_loc, fracent * 9 / 4, prate_global, z0, T1, T2, R_on, u0, v0, V_SR
            )

            # WE WILL NEED THE PROFILE OF POSITIVE BUOYANCY TO ESTIMATE STORM MOTION LATER.
            # ZERO OUT THE NEGATIVE BUOYANCY
            B_pos = np.maximum(B_pos, 0)

            # IF WE ACTUALLY HAVE ANY POSITIVE BUOYANCY, WE'LL ADVANCE THE MODEL
            if ~np.isnan(EL):
                # GET THE 0-1 KM STORM-RELATIVE FLOW
                V_SR = compute_VSR_DIFF(z0, u0, v0, rho0, EL, B_pos)
                # V_SR = compute_VSR(z0,u0,v0)

                # ADVANCE TO THE NEXT RADIUS USING EQ XX IN XX
                R_next = 1.7 * ((EL2 / EL) ** 2) * 2 * V_SR * (EL - LFC) * sig / (np.pi * alpha * np.sqrt(2 * CAPE))
                R_next = (rfac) * R_next + (1 - rfac) * R_on  # RELAXATION PROCEEDURE

            else:  # OTHERWISE SET THE RADIUS AT THE NEXT TIME TO ZERO
                R_next = 0

            if EL < LFC:  # THIS HAPPENS SOMETIMES.  SET TO ZERO IF EL IS LESS THAN LFC
                R_next = 0

            # STORE TIME SERIES'
            R_TS[ir, it + 1] = R_next
            H_TS[ir, it] = EL
            W_TS[ir, it] = np.sqrt(2 * CAPE)
            VSR_TS[ir, it] = V_SR
        R_TS[np.where(np.isnan(R_TS))] = 0

    return R_TS, H_TS, W_TS, VSR_TS


# ==============================================================================
# FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_w(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2, Radius, u0, v0, V_SR):
    # [CAPE,CIN,LFC,EL]

    # this function computes CAPE and CIN

    # input arguments
    # T0: sounding profile of temperature (in K)
    # p0: sounding profile of pressure (in Pa)
    # q0: sounding profile of water vapor mass fraction (in kg/kg)
    # start_loc: index of the parcel starting location (set to 1 for the
    # lowest: level in the sounding)
    # fracent: fractional entrainment rate (in m^-1)

    # CONSTANTS
    Rd = 287.04  # dry gas constant
    Rv = 461.5  # water vapor gas constant
    epsilon = Rd / Rv  # %RATO OF THE TWO
    g = 9.81  # gravitational acceleration
    c_d = 0.2  # DRAG COEFICIENT ON A SPHERE
    Lambda = 0.6  # RATIO OF ASCENT RATE OF THERMAL TO ITS MAX W
    alpha = 0.8  # ASSUMED RATIO OF HORIZONTALLY AVERAGED W TO HORIZONTAL MAX OF W AT A GIVEN LEVEL

    # COMPUTE A VERTICAL PROFILE OF THE MAGNITUDE OF VERTICAL WIND SHEAR
    dz = np.zeros(u0.shape)
    dz[0 : u0.shape[0] - 1] = z0[1 : u0.shape[0]] - z0[0 : u0.shape[0] - 1]
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(u0.shape)
    dudz[0 : dudz.shape[0] - 1] = (u0[1 : dudz.shape[0]] - u0[0 : dudz.shape[0] - 1]) / dz[0 : dudz.shape[0] - 1]
    dvdz[0 : dudz.shape[0] - 1] = (v0[1 : dudz.shape[0]] - v0[0 : dudz.shape[0] - 1]) / dz[0 : dudz.shape[0] - 1]
    S = np.sqrt(dudz**2 + dvdz**2)

    # COMPUTE THE LIFTED PARCEL BUOYANCY
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2)

    # CALCULATE THE LIFTED CONDENSATION LEVEL
    qdiff = abs(
        Qt_lif - Qv_lif
    )  # FIGURE OUT THE FIRST HEIGHT WHERE QV STARTS DEVIATING FROM QT, IMPLYING CONDENSATION
    if np.logical_and(~np.isnan(qdiff[1]), np.nanmax(qdiff) > 0):
        lcl_ind = np.where(qdiff > 0)[0][0]
        LCL = z0[lcl_ind]
    else:
        LCL = 1000
        lcl_ind = np.where(abs(LCL - z0) == np.amin(abs(LCL - z0)))[0][0]

    # IF WE HAVE SOME POSITIVE BUOYANCY, PROCEED
    if np.nanmax(B_lif) > 0:
        # MAKE A NEW MATRIX THAT WILL ONLY CONTAIN THE POSITIVE PART OF BUOYANCY
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]

        # GET RID OF ALL NEGATIVE BUOYANCY BELOW THE LCL
        B_pos[0:lcl_ind] = 0
        wpos = np.where(B_pos > 0)[0]
        if len(wpos) > 0:
            wpos = wpos[0]  # WPOS CONTAINS INDEX OF LCL.  SET TO 0 IF THERE IS NO POSTIVE BUOYANCY
        else:
            wpos = lcl_ind
        B_pos[0:wpos] = 0
        dz = z0[1 : z0.shape[0]] - z0[0 : z0.shape[0] - 1]

        # LFC WILL BE THE LAST INSTANCE OF NEGATIVE BUOYANCY BEFORE THE PARCEL REACHES ITS CONTINUOUS INTERVAL OF POSITIVE BUOY
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif == mx)
        imx = imx[0][0]

        fneg = np.where(B_lif < 0)
        fneg = fneg[0]
        inn = np.where(fneg < imx)

        inn = inn[0]
        fneg = fneg[inn]
        if len(inn) > 0:
            LFC = 0.5 * z0[np.max(fneg)] + 0.5 * z0[np.max(fneg) + 1]
        else:
            LFC = z0[start_loc]

        # EL WILL BE THE LAST INSTANCE OF POSITIVE BUOYANCY
        fpos = np.where(B_lif > 0)
        fpos = fpos[0]
        EL = 0.5 * z0[np.max(fpos)] + 0.5 * z0[np.max(fpos) + 1]

        # INTIALIZE PROFILE OF SQUARED VERTICAL VELOCITY (I.E., VERTICAL KINETIC ENERGY)
        WSQ_prof = np.zeros(B_pos.shape[0])
        WSQ_prof[start_loc] = (V_SR**2) / 2  # LOWER BOUNADRY CONDITION ON VERTICAL KE IS THE KE OF INFLOW
        uprime_prof = np.zeros(B_pos.shape[0])  # INITIALIZE UPRIME PROFILE
        for iz in np.arange(0, WSQ_prof.shape[0] - 1, 1):  # VERTICALLY INTEGRATE
            B_on = B_pos[iz]  # STORE THE CURRENT BUOYANCY
            ebuoy_fac = 1 / (
                1 + 2 * (alpha**2) * (Radius**2) / ((EL - LFC) ** 2)
            )  # SCALE FACTOR THAT ACCOUNTS FOR EFFECITVE BUOYANCY
            # ns_drag = -2.5*c_d*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG
            ns_drag = -c_d * (3 / 8) / Radius  # COEFICIENT ON THE NON-SHEARED PART OF DRAG
            s_drag = -(c_d / Radius) * (1 - Lambda) / (Lambda**2)  # COEFICIENT ON THE SHEARED PART OF DRAG
            sh_drag = (1 / (0.5 * np.sqrt(2 * WSQ_prof[iz - 1]))) * (3 * c_d / (8 * Radius))  # SHEARED DRAG TERM

            if (
                np.sqrt(2 * WSQ_prof[iz - 1]) < 1
            ):  # IF WE HAVE VERY SMALL VERTICAL VELOICTY (LESS THAN 1 M/S, WE NEED TO ZERO OUT THE SHEAR DRAG TERM OR THINGS BLOW UP)
                sh_drag = 0

            # NOW VERTICALLY INTEGRATE THE UPRIME AND WSQ EQUATIONS TOGETHER, FOLLOWING EQ. XX AND XX IN XX RESPECTIVELY
            uprime_prof[iz + 1] = uprime_prof[iz - 1] + (z0[iz + 1] - z0[iz]) * (
                -sh_drag * uprime_prof[iz] ** 2 + S[iz]
            )
            WSQ_prof[iz + 1] = WSQ_prof[iz] + (z0[iz + 1] - z0[iz]) * (
                ebuoy_fac * B_on + ns_drag * WSQ_prof[iz] + s_drag * uprime_prof[iz] * np.sqrt(2 * WSQ_prof[iz])
            )

        # WE WILL OUTPUT THE MAXIMUM KE AS THE "CAPE" ARGUMENT
        CAPE = np.nanmax(WSQ_prof)

        # SET THE EL TO THE HEIGHT OF MAXIMUM VERTICAL VELOCITY
        mxval = np.nanmax(WSQ_prof)
        fnval = np.where(WSQ_prof == mxval)
        LFC = LCL
        if fnval[0].shape[0] > 0:
            EL = z0[fnval[0][0]]
        else:
            EL = np.nan
    else:
        # IF WE HAVE NO POSITIVE BUOYANCY, SET EVERYTHING TO 0S AND NANS
        CAPE = 0
        LFC = np.nan
        EL = np.nan
        B_pos = np.zeros(T0.shape)

    return CAPE, LFC, EL, B_pos


# ==============================================================================
# FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_VSR_DIFF(z0, u0, v0, rho0, EL, B_pos):
    # compute 0-1 km storm-relative flow (V_SR) using the storm motion
    # estimate of Bunkers et al. (2000)
    # https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2

    zdiff = (z0 - EL) ** 2
    ind_top = np.where(zdiff == np.min(zdiff))[0][0]
    inds_avg = np.arange(0, ind_top, 1)

    meanx = np.nanmean(B_pos[inds_avg] * rho0[inds_avg] * u0[inds_avg]) / np.nanmean(B_pos[inds_avg] * rho0[inds_avg])
    meany = np.nanmean(B_pos[inds_avg] * rho0[inds_avg] * v0[inds_avg]) / np.nanmean(B_pos[inds_avg] * rho0[inds_avg])

    # meanx = np.nanmean(rho0[inds_avg]*u0[inds_avg])/np.nanmean(rho0[inds_avg])
    # meany = np.nanmean(rho0[inds_avg]*v0[inds_avg])/np.nanmean(rho0[inds_avg])

    f6000 = np.where(z0 <= 6000)[0]
    # meanx=np.mean(u0[f6000])
    # meany=np.mean(v0[f6000])

    f0500 = np.where(z0 <= 500)[0]
    lowx = np.mean(u0[f0500])
    lowy = np.mean(v0[f0500])

    f560 = np.where(np.logical_and(z0 <= 6000, z0 >= 5500))[0]
    highx = np.mean(u0[f560])
    highy = np.mean(v0[f560])
    BK_SHRx = highx - lowx
    BK_SHRy = highy - lowy
    BK_mag = np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx = BK_SHRx / BK_mag
    BK_diry = BK_SHRy / BK_mag
    BK_orthx = BK_diry * 7.5
    BK_orthy = -BK_dirx * 7.5

    SR_mean_u = u0 - meanx
    SR_mean_v = v0 - meany
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(v0.shape)
    dudz[1 : dudz.shape[0] - 1] = (u0[2 : dudz.shape[0]] - u0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dudz[0] = 2 * dudz[1] - dudz[2]
    dvdz[1 : dudz.shape[0] - 1] = (v0[2 : dudz.shape[0]] - v0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dvdz[0] = 2 * dvdz[1] - dvdz[2]
    f1000 = np.where(z0 <= 1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000] * dvdz[f1000] + SR_mean_v[f1000] * dudz[f1000]) * 1000.0)

    propfac = min(SRH_mean / 150, 1)

    C_x = meanx + propfac * BK_orthx
    C_y = meany + propfac * BK_orthy

    u_sr = u0 - C_x
    v_sr = v0 - C_y

    f1000 = np.where(z0 <= 1000)[0]
    V_SR = np.nanmean(np.sqrt(u_sr[f1000] ** 2 + v_sr[f1000] ** 2))
    return V_SR


# ==============================================================================
# FUNCTION THAT COMPUTES BUNKERS SR MOTION
def compute_OMEGA_AND_SRH(z0, u0, v0, C_x, C_y, rho0, EL):
    # compute 0-1 km storm-relative flow (V_SR) using the storm motion
    # estimate of Bunkers et al. (2000)
    # https://doi.org/10.1175/1520-0434(2000)015<0061:PSMUAN>2.0.CO;2

    zdiff = (z0 - EL) ** 2
    ind_top = np.where(zdiff == np.min(zdiff))[0][0]
    inds_avg = np.arange(0, ind_top, 1)

    meanx = np.nanmean(rho0[inds_avg] * u0[inds_avg]) / np.nanmean(rho0[inds_avg])
    meany = np.nanmean(rho0[inds_avg] * v0[inds_avg]) / np.nanmean(rho0[inds_avg])

    f6000 = np.where(z0 <= 6000)[0]
    # meanx=np.mean(u0[f6000])
    # meany=np.mean(v0[f6000])

    f0500 = np.where(z0 <= 500)[0]
    lowx = np.mean(u0[f0500])
    lowy = np.mean(v0[f0500])

    f560 = np.where(np.logical_and(z0 <= 6000, z0 >= 5500))[0]
    highx = np.mean(u0[f560])
    highy = np.mean(v0[f560])
    BK_SHRx = highx - lowx
    BK_SHRy = highy - lowy
    BK_mag = np.sqrt(BK_SHRx**2 + BK_SHRy**2)
    BK_dirx = BK_SHRx / BK_mag
    BK_diry = BK_SHRy / BK_mag
    BK_orthx = BK_diry * 7.5
    BK_orthy = -BK_dirx * 7.5

    SR_mean_u = u0 - meanx
    SR_mean_v = v0 - meany
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(v0.shape)
    dudz[1 : dudz.shape[0] - 1] = (u0[2 : dudz.shape[0]] - u0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dudz[0] = 2 * dudz[1] - dudz[2]
    dvdz[1 : dudz.shape[0] - 1] = (v0[2 : dudz.shape[0]] - v0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dvdz[0] = 2 * dvdz[1] - dvdz[2]
    f1000 = np.where(z0 <= 1000)[0]
    SRH_mean = abs(np.mean(-SR_mean_u[f1000] * dvdz[f1000] + SR_mean_v[f1000] * dudz[f1000]) * 1000.0)

    # propfac=min(SRH_mean/150,2)
    propfac = min(SRH_mean / 250, 2)
    # propfac=1

    if C_x < 0:
        C_x = meanx + propfac * BK_orthx
        C_y = meany + propfac * BK_orthy

    u_sr = u0 - C_x
    v_sr = v0 - C_y
    sr_mag = np.sqrt(u_sr**2 + v_sr**2)
    dudz = np.zeros(u0.shape)
    dvdz = np.zeros(v0.shape)
    dudz[1 : dudz.shape[0] - 1] = (u0[2 : dudz.shape[0]] - u0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dudz[0] = 2 * dudz[1] - dudz[2]
    dvdz[1 : dudz.shape[0] - 1] = (v0[2 : dudz.shape[0]] - v0[0 : dudz.shape[0] - 2]) / (
        z0[2 : dudz.shape[0]] - z0[0 : dudz.shape[0] - 2]
    )
    dvdz[0] = 2 * dvdz[1] - dvdz[2]
    f1000 = np.where(z0 <= 1000)[0]
    SRH = abs(np.mean(-u_sr[f1000] * dvdz[f1000] + v_sr[f1000] * dudz[f1000]) * 1000.0)
    OMEGA = np.mean((-u_sr[f1000] * dvdz[f1000] + v_sr[f1000] * dudz[f1000]) / sr_mag[f1000])

    f1000 = np.where(z0 <= 1000)[0]
    V_SR = np.nanmean(np.sqrt(u_sr[f1000] ** 2 + v_sr[f1000] ** 2))
    return V_SR, C_x, C_y, SRH, OMEGA


# ==============================================================================
# FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPE_CONTS(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2):
    # [CAPE,CIN,LFC,EL]

    # this function computes CAPE and CIN

    # input arguments
    # T0: sounding profile of temperature (in K)
    # p0: sounding profile of pressure (in Pa)
    # q0: sounding profile of water vapor mass fraction (in kg/kg)
    # start_loc: index of the parcel starting location (set to 1 for the
    # lowest: level in the sounding)
    # fracent: fractional entrainment rate (in m^-1)

    # CONSTANTS
    Rd = 287.04  # dry gas constant
    Rv = 461.5  # water vapor gas constant
    g = 9.81  # gravitational acceleration
    # epsilon = Rd / Rv

    # compute lifted parcel buoyancy
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2)

    # compute lifted parcel buoyancy
    T_lif_p, Qv_lif_p, Qt_lif_p, B_lif_p = lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, 0.01, z0, T1, T2)

    # compute thermal buoyancy
    B_therm = g * (T_lif - T0) / T0

    B_therm_p = g * (T_lif_p - T0) / T0
    B_cond = B_therm - B_therm_p

    # water vapor buoyancy contribution
    B_vap = g * (Rv / Rd - 1) * (Qv_lif - q0)

    # condensate loading contribution
    B_load = -g * (Qt_lif - Qv_lif)

    if np.nanmax(B_lif) > 0:
        # CAPE will be the total integrated positive buoyancy
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]

        B_therm[np.where(B_pos < 0)] = 0
        B_cond[np.where(B_pos < 0)] = 0
        B_vap[np.where(B_pos < 0)] = 0
        B_load[np.where(B_pos < 0)] = 0

        B_pos[np.where(B_pos < 0)] = 0
        dz = z0[1 : z0.shape[0]] - z0[0 : z0.shape[0] - 1]
        CAPE = np.nansum(0.5 * B_pos[0 : z0.shape[0] - 1] * dz + 0.5 * B_pos[1 : z0.shape[0]] * dz)
        CAPE_therm = np.nansum(0.5 * B_therm[0 : z0.shape[0] - 1] * dz + 0.5 * B_therm[1 : z0.shape[0]] * dz)
        CAPE_cond = np.nansum(0.5 * B_cond[0 : z0.shape[0] - 1] * dz + 0.5 * B_cond[1 : z0.shape[0]] * dz)
        CAPE_vap = np.nansum(0.5 * B_vap[0 : z0.shape[0] - 1] * dz + 0.5 * B_vap[1 : z0.shape[0]] * dz)
        CAPE_load = np.nansum(0.5 * B_load[0 : z0.shape[0] - 1] * dz + 0.5 * B_load[1 : z0.shape[0]] * dz)

        # CIN will be the total negative buoyancy below the height of maximum
        # buoyancy
        B_neg = np.zeros(B_lif.shape)
        B_neg[:] = B_lif[:]
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif == mx)
        imx = imx[0][0]
        B_neg[0:imx] = np.minimum(B_neg[0:imx], 0)
        B_neg[imx : z0.shape[0]] = 0
        CIN = np.nansum(0.5 * B_neg[0 : z0.shape[0] - 1] * dz + 0.5 * B_neg[1 : z0.shape[0]] * dz)

        # LFC will be the last instance of negative buoyancy before the
        # continuous interval that contains the maximum in buoyancy
        fneg = np.where(B_lif < 0)
        fneg = fneg[0]
        inn = np.where(fneg < imx)
        inn = inn[0]
        fneg = fneg[inn]
        if len(fneg) > 0:
            LFC = 0.5 * z0[np.max(fneg)] + 0.5 * z0[np.max(fneg) + 1]
        else:
            LFC = z0[start_loc]

        # EL will be last instance of positive buoyancy
        fpos = np.where(B_lif > 0)
        fpos = fpos[0]
        EL = 0.5 * z0[np.max(fpos)] + 0.5 * z0[np.max(fpos) + 1]
    else:
        CAPE = 0
        CIN = 0
        LFC = np.nan
        EL = np.nan
        CAPE_therm = 0
        CAPE_cond = 0
        CAPE_vap = 0
        CAPE_load = 0

    return CAPE, CAPE_therm, CAPE_cond, CAPE_vap, CAPE_load


# ==============================================================================
# FUNCTION THAT COMPUTES CAPE, CIN, EL, LFC
def compute_CAPES_DRAG(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2, Radius, V_SR):
    # [CAPE,CIN,LFC,EL]

    # this function computes CAPE and CIN

    # input arguments
    # T0: sounding profile of temperature (in K)
    # p0: sounding profile of pressure (in Pa)
    # q0: sounding profile of water vapor mass fraction (in kg/kg)
    # start_loc: index of the parcel starting location (set to 1 for the
    # lowest: level in the sounding)
    # fracent: fractional entrainment rate (in m^-1)

    # CONSTANTS
    Rd = 287.04  # dry gas constant
    Rv = 461.5  # water vapor gas constant
    epsilon = Rd / Rv
    g = 9.81  # gravitational acceleration
    cp = 1005  # HEAT CAPACITY OF DRY AIR AT CONSTANT PRESSUREE
    alpha = 0.8
    # c_d = 0.2
    c_d = 0.2

    th0 = T0 * (1000 * 100 / p0) ** (Rd / cp)

    # compute lifted parcel buoyancy
    T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T0, p0, q0, start_loc, fracent, prate, z0, T1, T2)

    # CALCULATE THE LIFTED CONDENSATION LEVEL
    qdiff = abs(
        Qt_lif - Qv_lif
    )  # FIGURE OUT THE FIRST HEIGHT WHERE QV STARTS DEVIATING FROM QT, IMPLYING CONDENSATION
    if np.logical_and(~np.isnan(qdiff[1]), np.nanmax(qdiff) > 0):
        lcl_ind = np.where(qdiff > 0)[0][0]
        LCL = z0[lcl_ind]
    else:
        LCL = 1000
        lcl_ind = np.where(abs(LCL - z0) == np.amin(abs(LCL - z0)))[0][0]

    # IF WE HAVE SOME POSITIVE BUOYANCY, PROCEED
    if np.nanmax(B_lif) > 0:
        # MAKE A NEW MATRIX THAT WILL ONLY CONTAIN THE POSITIVE PART OF BUOYANCY
        B_pos = np.zeros(B_lif.shape)
        B_pos[:] = B_lif[:]

        # GET RID OF ALL NEGATIVE BUOYANCY BELOW THE LCL
        B_pos[0:lcl_ind] = 0
        wpos = np.where(B_pos > 0)[0]
        if len(wpos) > 0:
            wpos = wpos[0]  # WPOS CONTAINS INDEX OF LCL.  SET TO 0 IF THERE IS NO POSTIVE BUOYANCY
        else:
            wpos = lcl_ind
        B_pos[0:wpos] = 0
        dz = z0[1 : z0.shape[0]] - z0[0 : z0.shape[0] - 1]

        # LFC WILL BE THE LAST INSTANCE OF NEGATIVE BUOYANCY BEFORE THE PARCEL REACHES ITS CONTINUOUS INTERVAL OF POSITIVE BUOY
        mx = np.nanmax(B_lif)
        imx = np.where(B_lif == mx)
        imx = imx[0][0]

        fneg = np.where(B_lif < 0)
        fneg = fneg[0]
        inn = np.where(fneg < imx)

        inn = inn[0]
        fneg = fneg[inn]
        if len(inn) > 0:
            LFC = 0.5 * z0[np.max(fneg)] + 0.5 * z0[np.max(fneg) + 1]
        else:
            LFC = z0[start_loc]

        # EL WILL BE THE LAST INSTANCE OF POSITIVE BUOYANCY
        fpos = np.where(B_lif > 0)
        fpos = fpos[0]
        EL = 0.5 * z0[np.max(fpos)] + 0.5 * z0[np.max(fpos) + 1]

        # INTIALIZE PROFILE OF SQUARED VERTICAL VELOCITY (I.E., VERTICAL KINETIC ENERGY)
        WSQ_prof = np.zeros(B_pos.shape[0])
        WSQ_prof[start_loc] = (V_SR**2) / 2  # LOWER BOUNADRY CONDITION ON VERTICAL KE IS THE KE OF INFLOW
        for iz in np.arange(0, WSQ_prof.shape[0] - 1, 1):  # VERTICALLY INTEGRATE
            B_on = B_pos[iz]  # STORE THE CURRENT BUOYANCY
            ebuoy_fac = 1 / (
                1 + 2 * (alpha**2) * (Radius**2) / ((EL - LFC) ** 2)
            )  # SCALE FACTOR THAT ACCOUNTS FOR EFFECITVE BUOYANCY
            # ns_drag = -2.5*c_d*(3/8)/Radius #COEFICIENT ON THE NON-SHEARED PART OF DRAG

            N = np.max((g / th0[iz]) * (th0[iz + 1] - th0[iz]) / (z0[iz + 1] - z0[iz]), 0)
            N = (
                np.minimum(z0[iz] / Radius, 1) * 1 / 2
                + np.minimum(np.maximum((z0[iz] - Radius) / Radius, 0), 1) * (2 / 3 - 1 / 2)
            ) * N

            F = np.sqrt(np.max(WSQ_prof / 2, 0)) / (np.sqrt(N) * Radius)

            if N > 0:
                ns_drag = -(c_d + comp_cdwave(F)) * (3 / 8) / Radius  # COEFICIENT ON THE NON-SHEARED PART OF DRAG
            else:
                ns_drag = -(c_d) * (3 / 8) / Radius  # COEFICIENT ON THE NON-SHEARED PART OF DRAG

            if (
                np.sqrt(2 * WSQ_prof[iz - 1]) < 1
            ):  # IF WE HAVE VERY SMALL VERTICAL VELOICTY (LESS THAN 1 M/S, WE NEED TO ZERO OUT THE SHEAR DRAG TERM OR THINGS BLOW UP)
                sh_drag = 0

            # NOW VERTICALLY INTEGRATE THE UPRIME AND WSQ EQUATIONS TOGETHER, FOLLOWING EQ. XX AND XX IN XX RESPECTIVELY
            WSQ_prof[iz + 1] = WSQ_prof[iz] + (z0[iz + 1] - z0[iz]) * (ebuoy_fac * B_on + ns_drag * WSQ_prof[iz])

        # WE WILL OUTPUT THE MAXIMUM KE AS THE "CAPE" ARGUMENT
        CAPE = np.nanmax(WSQ_prof)

        # SET THE EL TO THE HEIGHT OF MAXIMUM VERTICAL VELOCITY
        mxval = np.nanmax(WSQ_prof)
        fnval = np.where(WSQ_prof == mxval)
        LFC = LCL
        if fnval[0].shape[0] > 0:
            EL = z0[fnval[0][0]]
        else:
            EL = np.nan
    else:
        # IF WE HAVE NO POSITIVE BUOYANCY, SET EVERYTHING TO 0S AND NANS
        CAPE = 0
        LFC = np.nan
        EL = np.nan
        B_pos = np.zeros(T0.shape)

    return CAPE


# ==============================================================================
# ==============================================================================
# =======================================1/======================================
# END FUNCTION DEFINITIONS======================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
