# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false, reportUnusedExpression=false, reportMissingImports=false

from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray
from libc.math cimport ceil, exp, isnan, sin, cos, fmax, fmin

import numpy as np
cimport numpy as np
from . import const as _const
from cython cimport floating
np.import_array()


cdef extern from *:
    """
    #if defined(OMP_SCHEDULE)
      #define _OMP_ENABLED 1
    #else
      #define _OMP_ENABLED 0
    #endif
    """
    cdef bint _OMP_ENABLED

cdef extern from "<math.h>" nogil:
    double sin(double x)
    double cos(double x)
    double tan(double x)
    double asin(double x)
    double acos(double x)
    double atan(double x)
    double atan2(double y, double x)
    const double pi "M_PI"  # as in Python's math module

OPENMP_ENABLED = bool(_OMP_ENABLED)

cdef:
    float Rd        = _const.Rd
    float Rv        = _const.Rv
    float Lv        = _const.Lv
    float Cpd       = _const.Cpd
    float epsilon   = _const.epsilon
    float T0        = _const.T0
    float E0        = _const.E0
    float nan       = float('nan')
    float inf       = float('inf')

del _const


cdef floating saturation_mixing_ratio(floating pressure, floating temperature) noexcept nogil:
    cdef floating P
    P = E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65)) # saturation vapor pressure
    return 0.6219 * P / (pressure - P)

# =================================================================================================
# moist_lapse
# =================================================================================================
cdef floating solver(floating pressure, floating temperature) noexcept nogil:
    cdef floating r
    r = saturation_mixing_ratio(pressure, temperature)
    r = (Rd * temperature + Lv * r) / (Cpd + (Lv * Lv * r * epsilon / (Rd * temperature**2)))
    r /= pressure
    return r


cdef floating intergrator(
    floating pressure, floating next_pressure, floating temperature, floating step
) noexcept nogil:
    """``2nd order Runge-Kutta (RK2)``
    
    Integrate the moist lapse rate ODE, using a RK2 method, from an initial pressure lvl
    to a final one.
    """
    cdef int N
    cdef floating delta, k1

    N = 1
    if abs(delta := next_pressure - pressure) > step:
        N =  <int> ceil(abs(delta) / step)
        delta = delta / <floating> N

    for _ in range(N):
        k1 = delta * solver(pressure, temperature)
        temperature += delta * solver(pressure + delta * 0.5, temperature + k1 * 0.5)
        pressure += delta

    return temperature


cdef void moist_lapse_1d_(
    floating[:] out, floating[:] pressure, floating reference_pressure, floating temperature, floating step
) noexcept nogil:
    """Moist adiabatic lapse rate for a 1D array of pressure levels."""
    cdef size_t i, Z
    cdef floating next_pressure

    Z = pressure.shape[0]
    if isnan(temperature) or isnan(reference_pressure): # don't bother with the computation
        for i in prange(Z):
            out[i] = nan
        return

    for i in range(Z):
        if isnan(next_pressure := pressure[i]): 
            # This can be usefull in 2D case where we might want to mask out some 
            # values below the lcl for a particular column.
            # The solver will start at the first non nan value
            out[i] = nan
        else:
            out[i] = temperature = intergrator(reference_pressure, next_pressure, temperature, step=step)
            reference_pressure = pressure[i]


cdef enum ComputationMode:
    INFER = 0
    BROADCAST = 1
    MATRIX = 2
    SLICE = 3


cdef float[:, :] moist_lapse_f32(
    (size_t, size_t) shape,
    float[:, :] pressure, 
    float[:] reference_pressure, 
    float[:] temperature, 
    float step,
    ComputationMode mode
):
    cdef int i
    cdef float[:, :] out
    out = cvarray(shape, itemsize=sizeof(float), format='f', mode='c')

    with nogil, parallel():
        if mode == MATRIX:
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[i], reference_pressure[i], temperature[i], step=step)
        elif mode == BROADCAST:
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[0], reference_pressure[i], temperature[i], step=step)
        else: # SLICE
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[0, i:i+1], reference_pressure[i], temperature[i], step=step)

    return out


cdef double[:, :] moist_lapse_f64(
    (size_t, size_t) shape,
    double[:, :] pressure, 
    double[:] reference_pressure, 
    double[:] temperature, 
    double step,
    ComputationMode mode
):
    cdef int i
    cdef double[:, :] out
    out = cvarray(shape, itemsize=sizeof(double), format='d', mode='c')

    with nogil, parallel():
        if mode == MATRIX:
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[i], reference_pressure[i], temperature[i], step=step)
        elif mode == BROADCAST:
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[0], reference_pressure[i], temperature[i], step=step)
        else: # SLICE
            for i in prange(out.shape[0], schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[0, i:i+1], reference_pressure[i], temperature[i], step=step)

    return out


def moist_lapse(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray reference_pressure = None,
    *,
    floating step = 1000.0,
    object dtype = None,
):
    """
    If reference_pressure is not provided and the pressure array is 2D, the reference pressure
    will be determined by finding the first non-nan value in each row.
    ```
    >>> pressure = np.array([
        [1013.12, 1000, 975, 950, 925, 900, ...],
        [1013.93, 1000, 975, 950, 925, 900, ...],
        [np.nan, np.nan, 975, 950, 925, 900, ...]
    ]) * 100.0 # (N, Z) :: pressure profile
    >>> reference_pressure = pressure[np.arange(len(pressure)), np.argmin(np.isnan(pressure), axis=1)]
    >>> reference_pressure
    array([101312., 101393.,  97500.  ])
    ```
    """
    cdef size_t N, Z, ndim
    cdef ComputationMode mode = INFER
    

    if dtype is None:
        dtype = pressure.dtype
    else:
        dtype = np.dtype(dtype)
    
    pressure = np.atleast_1d(pressure.squeeze())    # (Z,) | (N, Z)
    temperature = temperature.ravel()               # (N,)
    ndim = pressure.ndim

    if (N := temperature.shape[0]) == 0:
        return np.full_like(pressure, nan, dtype=dtype)

    # - determine the reference pressure
    if reference_pressure is not None:
        reference_pressure = reference_pressure.ravel()
        if (
            pressure.size == reference_pressure.size == temperature.size and 
            SLICE == sum(x.ndim for x in (pressure, temperature, reference_pressure)) 
        ):
            mode = SLICE
        else:
            mode = <ComputationMode> ndim
    elif BROADCAST == ndim:
        mode = BROADCAST
        reference_pressure = np.repeat(pressure[np.argmin(np.isnan(pressure))], N)
    elif MATRIX == ndim:
        mode = MATRIX
        reference_pressure = pressure[np.arange(N), np.argmin(np.isnan(pressure), axis=1)]
    else:
        raise ValueError("pressure must be 1D or 2D array")

    # - reshape pressure for the supported computation modes
    if BROADCAST == mode:
        pressure = pressure.reshape(1, -1)
        Z = pressure.shape[1]
    elif MATRIX == mode:
        Z = pressure.shape[1]
    elif SLICE == mode:
        pressure = pressure.reshape(1, -1)
        Z = 1
    else:
        raise ValueError("Could not infer computation mode, check the shape of the input arrays")

    cdef np.ndarray x = np.empty((N, Z), dtype=dtype)
    if dtype == np.float32:
        x[:, :] = moist_lapse_f32(
            (N, Z),
            pressure.astype(np.float32), 
            reference_pressure.astype(np.float32),
            temperature.astype(np.float32), 
            step=step,
            mode=mode
        )
    else:
        x[:, :] = moist_lapse_f64(
            (N, Z),
            pressure.astype(np.float64),
            reference_pressure.astype(np.float64),
            temperature.astype(np.float64),
            step=step,
            mode=mode
        )

    return x

# =================================================================================================
# in development...
# =================================================================================================
cdef floating czda(floating h_start, floating h_end, floating h_sunrise, floating h_sunset, floating lat, floating Decl, floating interval) nogil:
    # h_start: hour angle of the starting point of each interval (radian)
    # h_end: hour angle of the end point of each interval (radian)
    # h_sunrise: hour angle at sunrise (radian)
    # h_sunrise: hour angle at sunset (radian)
    # lat: latitude (radian)
    # Decl: solar declination angle (radian)
    # interval: length of interval (e.g. 3 for 3-hourly interval)
    # return: cosine zenith angle during only the sunlit part of each interval
    cdef floating h_min, h_max, cosz, h_min1, h_max1, h_min2, h_max2
    if isnan(h_sunrise) and (lat * Decl) > 0:
        h_min = h_start
        h_max = h_end
        cosz = sin(Decl) * sin(lat) + cos(Decl) * cos(lat) * (sin(h_max) - sin(h_min)) * ((interval * 15.0 / 180.0 * pi)**(-1))
    elif isnan(h_sunrise) and lat*Decl<0:
        cosz = 0
    elif (
        (h_start > h_sunset and h_end < h_sunrise) 
        or (h_start < h_sunrise and h_end < h_sunrise) 
        or (h_start > h_sunset and h_end > h_sunset)
    ):
        cosz=0
    elif (h_start>h_sunset and h_end<0 and h_end>h_sunrise):
        h_min=h_sunrise
        h_max=h_end
        cosz= sin(Decl) * sin(lat)+ cos(Decl) * cos(lat) *(sin(h_max)- sin(h_min))*((h_max-h_min)**(-1))
    elif (h_start>0 and h_start<h_sunset and h_end<h_sunrise):
        h_min=h_start
        h_max=h_sunset
        cosz= sin(Decl)* sin(lat)+ cos(Decl)* cos(lat)*( sin(h_max)- sin(h_min))*((h_max-h_min)**(-1))
    elif (h_start > 0 and h_start < h_sunset and h_end < 0 and h_end > h_sunrise):
        h_min1 = h_start
        h_max1 = h_sunset
        h_min2 = h_sunrise
        h_max2 = h_end
        cosz = (
            (sin(Decl)* sin(lat) * (h_max1-h_min1) + cos(Decl) * cos(lat)*(sin(h_max1) - sin(h_min1))
            + sin(Decl)* sin(lat)*(h_max2-h_min2) + cos(Decl)* cos(lat)*(sin(h_max2) - sin(h_min2)))
            *((h_max1-h_min1+h_max2-h_min2)**(-1))
        )
    else:
        h_min = fmax(h_sunrise, h_start)
        h_max = fmin(h_sunset, h_end)
        cosz = sin(Decl)* sin(lat)+ cos(Decl)* cos(lat)*( sin(h_max)- sin(h_min))*((h_max-h_min)**(-1))
    return cosz



def black_globe_temperature():...
def wet_bulb_globe_temperature():...