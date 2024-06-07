#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]  # for \text command
import csv

import matplotlib.gridspec as gridspec
import metpy.calc as mpcalc

#
from ECAPE_FUNCTIONS import *
from metpy.plots import Hodograph, SkewT
from metpy.units import units

#
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate

do_snd = True

# plt_smp = 25
plt_smp = 50

# CONSTANTS, SET TO THEIR VALUES IN CM1
Rd = 287.04  # DRY GAS CONSTANT
Rv = 461.5  # WATER VAPOR GAS CONSTANT
epsilon = Rd / Rv  # RATIO OF THE TWO
cp = 1005  # SPECIFIC HEAT OF DRY AIR AT CONSTANT PRESSURE
g = 9.81  # GRAVIATIONAL CONSTANT
xlv = 2501000  # REFERENCE LATENT HEAT OF FREEZING AT THE TRIPLE POINT TEMP
xls = 2834000  # REFERENCE LATENT HEAT OF SUBLIMATION AT THE TRIPLE POINT TEMP
cpv = 1870  # SPECIFIC HEAT OF WATER VAPOR AT CONSTANT PRESSURE
cpl = 4190  # SPECIFIC HEAT OF LIQUID WATER
cpi = 2106  # SPECIFIC HEAT OF ICE
pref = 611.65  # PARTIAL PRESSURE OF WATER VAPOR AT TRIPLE POINT TEMP
ttrip = 273.15  # TRIPLE POINT TEMP
ksq = 0.18  # VON KARMAN CONSTANT
P_r = 1 / 3  # PRANDTL NUMBER
L_m = 250  # HORIZONTAL MIXING LENGTH
T1 = 273.15
T2 = 253.15
RA_filter_fac = 0.9


# ==============================================================================
# ==============================================================================
# LOAD DATA FROM A CM1 INPUT FILE
# GRID WE ARE GOING TO INTERPOLATE TO
dz = 100
znew = np.arange(0, 20 * 1000, dz, dtype=float)

data_ctr = 0
Z = []
Th = []
r_v = []
u = []
v = []
# with open('input_sounding', newline = '') as int_snd:
# with open('input_sounding_AMAZON', newline = '') as int_snd:
with open("input_sounding", newline="") as int_snd:
    # with open('input_sounding_HIGHMED', newline = '') as int_snd:
    # snd_dat = csv.reader(int_snd, delimiter='\t')
    snd_dat = csv.reader(
        int_snd
    )  # MIGHT NEED TO CHANGE THIS LINE BASED ON WHAT DELIMITER YOU USE IN YOUR CM1 INPUT FILE
    for idat in snd_dat:
        data_ctr = data_ctr + 1
        dataon = idat[0].split()
        if data_ctr == 1:  # FIRST GET SURFACE DATA FROM LINE 1
            p_sfc = float(dataon[0]) * 100.0
            Th_sfc = float(dataon[1])
            r_sfc = float(dataon[2]) / 1000.0
            Z.append(0.0)
            Th.append(Th_sfc)  # MAKE SURE TO PUT SURFACE VALUES IN THE FIRST ELEMENT OF LISTS
            r_v.append(r_sfc)
            u.append(0.0)
            v.append(0.0)
        else:
            Z.append(float(dataon[0]))  # LOAD THE REST OF THE PROFILE
            Th.append(float(dataon[1]))
            r_v.append(float(dataon[2]) / 1000.0)
            u.append(float(dataon[3]))
            v.append(float(dataon[4]))

u[0] = 2 * u[1] - u[2]  # INTERPOLATE WINDS TO THE SURFACE
v[0] = 2 * v[1] - v[2]

if Z[0] == 0:
    znew[
        0
    ] = 0.00000001  # FOR SOME REASON, THE INTERPOLATION BREAKS IF THE SURFACE HEIGHT IS SET TO ZERO.  SET TO VERY SMALL VALUE

# INTERPOLATE VERYTHING ONTO A REGULAR GRID
f = interpolate.interp1d(Z, Th, fill_value="extrapolate", kind="linear")
Th = f(znew)
f = interpolate.interp1d(Z, r_v, fill_value="extrapolate", kind="linear")
r_v = f(znew)
f = interpolate.interp1d(Z, u, fill_value="extrapolate", kind="linear")
u = f(znew) - 1
f = interpolate.interp1d(Z, v, fill_value="extrapolate", kind="linear")
v = f(znew) + 3.5
Z = znew

Z = np.array(Z, dtype=float)  # CONVERT TO FLOAT ARRAYS
Th = np.array(Th, dtype=float)
r_v = np.array(r_v, dtype=float)
u = np.array(u, dtype=float)
v = np.array(v, dtype=float)

# GET THE PRESSURE USING THE HYDROSTATIC EQUATION
P = np.zeros(Z.shape[0])
Pii = np.zeros(Z.shape[0])
Pii[0] = (p_sfc / (1000.0 * 100.0)) ** (Rd / cp)
Th_rho = Th * (1 + r_v / epsilon) / (1 + r_v)
intarg = -(g / cp) / Th_rho
for iz in np.arange(1, Z.shape[0], 1):
    Pii[iz] = Pii[iz - 1] + 0.5 * (intarg[iz] + intarg[iz - 1]) * (Z[iz] - Z[iz - 1])
P = 1000.0 * 100.0 * Pii ** (cp / Rd)
T_rho = Th * Pii * (1 + r_v / epsilon) / (1 + r_v)
rho = P / (Rd * T_rho)

# GET THE TEMPERATIRE
T0 = Th * (P / (1000 * 100.0)) ** (Rd / cp)
q0 = (1 - r_v) * r_v
p0 = P
z0 = Z
u = u * units("m/s^2")
v = v * units("m/s^2")

# GET THE SURFACE-BASED CAPE, CIN, LFC, EL
CAPE, CIN, LFC, EL = compute_CAPE_AND_CIN(T0, p0, q0, 0, 0, 0, z0, T1, T2)
WCAPE, WCIN, WLFC, WEL = compute_w(T0, p0, q0, 0, 0, 0, z0, T1, T2, 1000)

# GET NCAPE, WHICH IS NEEDED FOR ECAPE CALULATION
NCAPE, MSE0_star, MSE0bar = compute_NCAPE(T0, p0, q0, z0, T1, T2, LFC, EL)

# GET THE 0-1 KM MEAN STORM-RELATIVE WIND, ESTIMATED USING BUNKERS METHOD FOR RIGHT-MOVER STORM MOTION
V_SR, C_x, C_y = compute_VSR(z0, u.magnitude, v.magnitude)

# GET E_TILDE, WHICH IS THE RATIO OF ECAPE TO CAPE.  ALSO, VAREPSILON IS THE FRACITONAL ENTRAINMENT RATE, AND RADIUS IS THE THEORETICAL UPRAFT RADIUS
E_tilde, varepsilon, Radius = compute_ETILDE(CAPE, NCAPE, V_SR, EL, 250)

# CI THEORETICAL MODEL
R_TS = CI_model(T0, p0, q0, z0, u.magnitude, v.magnitude, T1, T2, np.arange(300, 6000, 100), 20, 250)

plt.plot(np.transpose(R_TS))

# ==============================================================================
# COMPUTE DEWPOINT AND TEMPERATURE, ASSIGN UNITS IN ACCORDANCE WITH METPY VERNACULAR
# ==============================================================================

P = p0
T = T0
p = P * units.Pa
e = mpcalc.vapor_pressure(P * units.Pa, q0 * units("kg/kg"))
Td = mpcalc.dewpoint(e)
for iz in np.arange(0, Td.shape[0], 1):
    T[iz] = max(T[iz], Td[iz].to("K").magnitude)
T = T * units.degK

# COMPUTE LIFTED PARCEL PROPERTIES FOR AN UNDILUTED PARCEL
fracent = 0
# prate=3e-5
T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T.magnitude, p0, q0, 0, fracent, 0, z0, T1, T2)
T_rho = T_lif * (1 + (Rv / Rd) * Qv_lif - Qt_lif)
T_rho = T_rho * units("K")

# PLOT THE SKEW-T SKELETON
params = {
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "font.size": 12,
    "text.usetex": True,
}
plt.rcParams.update(params)


gs = gridspec.GridSpec(3, 3)
fig = plt.figure(figsize=(9, 9))  # axs = plt.subplots(1, 2, constrained_layout=True)

skew = SkewT(fig, rotation=45, subplot=gs[:, :2])

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot.
skew.plot(p, T, "r")
skew.plot(p, Td, "g")
skew.plot_barbs(p[0::5], u[0::5], v[0::5], x_clip_radius=0.1, y_clip_radius=0.08)
skew.ax.set_ylim(1020, 100)
skew.ax.set_xlim(-20, 40)

skew.ax.text(
    -15,
    900,
    "CAPE: "
    + str(round(CAPE))
    + " J kg$^{-1}$\n ECAPE: "
    + str(round(E_tilde * CAPE))
    + " J kg$^{-1}$\n$\widetilde{E}$: "
    + str(round(E_tilde * 100))
    + "\%\nR: "
    + str(round(Radius))
    + " m",
    ha="center",
    va="center",
    size=7,
    bbox=dict(boxstyle="square,pad=0.3", fc="lightblue", ec="steelblue", lw=2),
)

# Set some better labels than the default
skew.ax.set_xlabel(f"Temperature ({T.units:~P})")
skew.ax.set_ylabel(f"Pressure ({p.units:~P})")

lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, "ko", markerfacecolor="black")

# Calculate full parcel profile and add to plot as black line
# prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
prof = T_rho.to("degC")
skew.plot(p, prof, "k", linewidth=2)
T_rho0 = T.magnitude * (1 + (Rv / Rd - 1) * q0)
T_rho0 = T_rho0 * units("K")

skew.plot(p, T_rho0, "r", linewidth=0.5)
# skew.plot(p,T_rho.to('degC'),'k',linewidth=2)

# Shade areas of CAPE and CIN

try:
    skew.shade_cin(p, T_rho0.to("degC"), prof, Td)
except:
    print("NO CIN")
try:
    skew.shade_cape(p, T_rho0.to("degC"), prof)
except:
    print("NO CAPE")

fracent = varepsilon
# prate=3e-5
T_lif, Qv_lif, Qt_lif, B_lif = lift_parcel_adiabatic(T.magnitude, p0, q0, 0, fracent, 0, z0, T1, T2)
ECAPE, ECIN, ELFC, EEL = compute_CAPE_AND_CIN(T0, p0, q0, 0, fracent, 0, z0, T1, T2)
# COMPUTE DENSITY TEMEPRATURE FOR THE LIFTED PARCEL AND ASSIGN UNITS
T_rho = T_lif * (1 + (Rv / Rd) * Qv_lif - Qt_lif)
T_rho = T_rho * units("K")

skew.plot(p, T_rho, "b--", linewidth=1)
prof = T_rho.to("degC")
try:
    skew.shade_cape(p, T_rho0.to("degC"), prof, facecolor=(0.5, 0.5, 0.5, 0.75))
except:
    print("NO CAPE")


# An example of a slanted line at constant T -- in this case the 0
# isotherm
skew.ax.axvline(0, color="c", linestyle="--", linewidth=2)

# Add the relevant special lines
skew.plot_dry_adiabats(linewidths=0.5)
skew.plot_moist_adiabats(linewidths=0.5)
skew.plot_mixing_lines(linewidths=0.5)

ax_hod = inset_axes(skew.ax, "40%", "40%", loc=1)
h = Hodograph(ax_hod, component_range=55.0)
h.add_grid(increment=20, linewidth=0.5)

fplt = np.where(z0 <= 6000)[0]

cmap = plt.get_cmap("autumn_r", len(fplt))
h.plot_colormapped(
    u[fplt], v[fplt], np.floor(z0[fplt] / 1000), linewidth=2, cmap=plt.get_cmap("gist_rainbow_r", 6)
)  # Plot a line colored by wind speed
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.tick_params(
    axis="both",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    left=False,
    top=False,  # ticks along the top edge are off
    labelleft=False,
    labelbottom=False,
)  # labels along the bottom edge are off
plt.text(-5, -21, "20 kt", fontsize=8)
plt.text(-5, -41, "40 kt", fontsize=8)
plt.plot(C_x, C_y, "ko", markersize=2.5)

plt.savefig("sndfig.pdf")


# ==============================================================================
# ==============================================================================
# ==============================================================================
# ACTUALLY RUN THE MODEL BELOW==================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
