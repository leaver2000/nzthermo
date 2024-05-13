# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false

from cython.parallel cimport parallel, prange

import numpy as np
cimport numpy as np


cdef extern from "wbgt.h" nogil:
    int calc_wbgt(
        int year,
        int hour,
        int day,
        int month,
        int minute,
        int gmt,
        int avg,
        double lat,
        double lon,
        double solar,
        double pressure,
        double temperature,
        double relhum,
        double speed,
        double zspeed,
        double dT,
        int urban,
        double* est_speed,
        double* Tg,
        double* Tnwb,
        double* Tpsy,
        double* Twbg
    )


cdef void wbgt_(
    double[:, :, :, :] out,
    int[:] years,
    int[:] hours,
    int[:] days,
    int[:] months,
    int[:] minutes,
    int gmt,
    int avg,
    double[:, :] latitude,
    double[:, :] longitude,
    double[:, :, :] pressure,
    double[:, :, :] temperature,
    double[:, :, :] relative_humidity,
    double[:, :, :] wind_speed,
    double solar_radiation = 1.0,
    double wind_speed_height = 10.0,
    double delta_z = 0.0,
    bint urban_flag = 0,
) noexcept nogil:
    cdef size_t T, Y, X, i, t, y, x

    T = years.shape[0]
    Y = latitude.shape[0]
    X = longitude.shape[1]

    with nogil, parallel():
        for i in prange(T * Y * X, schedule='dynamic'):
            t = i // (Y * X)
            y = (i // X) % Y
            x = i % X

            calc_wbgt(
                years[t], hours[t], days[t], months[t], minutes[t], 
                gmt, 
                avg, 
                latitude[y, x],
                longitude[y, x],
                solar_radiation,         # solar radiation, W/m^2
                pressure[t, y, x], 
                temperature[t, y, x], 
                relative_humidity[t, y, x], 
                wind_speed[t, y, x], 
                wind_speed_height, 
                delta_z,                # vertical temperature difference (upper minus lower), degC
                urban_flag,             # urban flag
                # <int>500,
                &out[0, t, y, x], 
                &out[1, t, y, x], 
                &out[2, t, y, x], 
                &out[3, t, y, x],
                &out[4, t, y, x]
            )

def wbgt(
    np.ndarray datetime,
    np.ndarray latitude,
    np.ndarray longitude,
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray relative_humidity,
    np.ndarray wind_speed,
    *,
    double wind_speed_height = 10.0,
    double solar_radiation = 1.0,
    double delta_z = 0.0,
    bint urban_flag = 0,
):
    cdef size_t T, Y, X
    cdef int[:] years, months, days, hours, minutes
    cdef double[:, :, :] pres, Tair, relhum, speed
    cdef double[:, :, :, :] out
    cdef double[:, :] lat, lon
    cdef int gmt = 0
    cdef int avg = 0
    
    # [ time ]
    years = datetime.astype('datetime64[Y]').astype(np.int32) + 1970
    months = datetime.astype('datetime64[M]').astype(np.int32) % 12 + 1
    days = datetime.astype('datetime64[D]').astype(np.int32) % 365 + 1
    hours = datetime.astype('datetime64[h]').astype(np.int32) % 24
    minutes = datetime.astype('datetime64[m]').astype(np.int32) % 60
    
    # [ lattitude & longitude ]
    if longitude.ndim == 1 == latitude.ndim:
        longitude, latitude = np.meshgrid(longitude, latitude, indexing='xy')
    elif longitude.ndim != latitude.ndim:
        raise ValueError("latitude and longitude must be 1D or 2D arrays.")

    lat = latitude.astype(np.float64)
    lon = longitude.astype(np.float64)

    T = datetime.shape[0]
    Y = lat.shape[0]
    X = lon.shape[1]

    # [ parameters ]
    pres = pressure.astype(np.float64)
    Tair = temperature.astype(np.float64)
    relhum = relative_humidity.astype(np.float64)
    speed = wind_speed.astype(np.float64)

    
    wbgt_(
        out := np.empty((5, T, Y, X), dtype=np.float64),
        years, hours, days, months, minutes, 
        gmt, avg, lat, lon, pres, Tair, relhum, speed,
        wind_speed_height=wind_speed_height,
        solar_radiation=solar_radiation,
        delta_z=delta_z,
        urban_flag=urban_flag,
    )
    return np.array(out, dtype=np.float64, copy=False)
