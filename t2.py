import src.nzthermo._core as C
import numpy as np
import metpy.calc.tools as mptools
import time

TIME = 144
Y = 134

lcl_p = np.array([93290.11, 92921.01, 92891.83, 93356.17, 94216.14] * 100)  # (N,)
pressure_levels = np.array(
    [101300.0, 100000.0, 97500.0, 95000.0, 92500.0, 90000.0, 87500.0, 85000.0, 82500.0, 80000.0],
)  # (Z,)
temperature = np.array(
    [
        [303.3, 302.36, 300.16, 298.0, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
        [303.58, 302.6, 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6],
        [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
        [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1],
        [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47],
    ]
    * 100,
)  # (N, Z)
dewpoint = np.array(
    [
        [297.61, 297.36, 296.73, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
        [297.62, 297.36, 296.79, 296.18, 294.5, 292.07, 287.74, 286.67, 285.15, 284.02],
        [297.76, 297.51, 296.91, 296.23, 295.05, 292.9, 288.86, 287.12, 285.99, 283.98],
        [297.82, 297.56, 296.95, 296.23, 295.0, 292.47, 289.97, 288.45, 287.09, 285.17],
        [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31],
    ]
    * 100,
)  # (N, Z)

start = time.time()
x = C.interpolate_nz(lcl_p, pressure_levels, temperature)
print(time.time() - start)

r = []
start = time.time()
for i in range(temperature.shape[0]):
    y = mptools.interpolate_1d(lcl_p[i], pressure_levels, temperature[i])
    r.append(y)
print(time.time() - start)
# print(
#     x,
#     np.array(r),
#     sep="\n",
# )

# y = mptools.interpolate_1d(pressure_levels, lcl_p, temperature)
# print(time.time() - start)


# print(C)
# print(
#     C.interpolate_nz(lcl_p, pressure_levels, temperature),
#     mptools.interpolate_1d(pressure_levels, lcl_p, temperature),
#     # F.interpolate_nz(lcl_p, pressure_levels, temperature),
#     C.interpolate_nz(lcl_p, pressure_levels, temperature, log_x=True),
#     C.interpolate_nz(lcl_p, pressure_levels, temperature, interp_nan=True),
#     # F.interpolate_nz(lcl_p, pressure_levels, temperature, log_x=True),
# )
