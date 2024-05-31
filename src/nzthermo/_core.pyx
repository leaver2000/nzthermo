# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
"""
This is the Cython implementation of the parcel analysis functions.

The header file makes avalaible several atmospheric functions in the _core.cpp file. The function
templates support `std::floating_point` types. Additonaly in the header is a fused type `T` which
can be assumed to suport either ``float`` or ``double`` types.

```cpp
template <typename T>
    requires std::floating_point<T>
T fn(T ...){...}
```
"""
cimport cython
import cython

from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np

cimport nzthermo._C as C

np.import_array()
np.import_ufunc()


cdef extern from *:
    """
    #ifdef _OPENMP
    #define OPENMP 1
    #else
    #define OPENMP 0
    #endif /* OPENMP */
    """
    cdef bint OPENMP


cdef extern from "<math.h>" nogil:
    bint isnan(long double x)
    const float NaN "NAN"


ctypedef fused T:
    float
    double


ctypedef fused integer:
    short
    long


ctypedef enum BroadcastMode:
    BROADCAST = 1
    MATRIX = 2
    ELEMENT_WISE = 3

ctypedef enum ProfileStrategy:
    SURFACE_BASED = 0
    MOST_UNSTABLE = 1
    MIXED_LAYER = 2
    EQUILIBRIUM = 3
    VIRTUAL = 4


OPENMP_ENABLED = bool(OPENMP)


cdef cvarray nzarray((size_t, size_t) shape, size_t size):
    return cvarray(
        shape, 
        itemsize=size, 
        format='f' if size == sizeof(float) else 'd', 
        mode='c',
        allocate_buffer=True,
    )

cdef T linear_interpolate(T x, T x0, T x1, T y0, T y1) noexcept nogil:
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    
    
# -------------------------------------------------------------------------------------------------
# thermodynamic functions
# -------------------------------------------------------------------------------------------------
cdef void moist_lapse_1d_(
    T[:] out, T[:] pressure, T reference_pressure, T temperature, T step
) noexcept nogil:
    """Moist adiabatic lapse rate for a 1D array of pressure levels."""
    cdef size_t Z, i
    cdef T next_pressure

    Z = pressure.shape[0]
    if isnan(temperature) or isnan(reference_pressure): # don't bother with the computation
        for i in prange(Z):
            out[i] = NaN
        return

    for i in range(Z):
        if isnan(next_pressure := pressure[i]): 
            # This can be usefull in 2D case where we might want to mask out some 
            # values below the lcl for a particular column.
            # The solver will start at the first non nan value
            out[i] = NaN
        else:
            out[i] = temperature = C.moist_lapse(
                reference_pressure, next_pressure, temperature, step
            )
            reference_pressure = next_pressure


cdef T[:, :] _moist_lapse(T[:, :] pressure, 
    T[:] reference_pressure, T[:] temperature, T step, BroadcastMode mode,
) noexcept:
    cdef size_t N, Z, i
    cdef T[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]

    out = nzarray((N, Z), pressure.itemsize)
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[0, :], reference_pressure[i], temperature[i], step=step)
        elif MATRIX is mode:
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[i, :], reference_pressure[i], temperature[i], step=step)
        else: # ELEMENT_WISE
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d_(out[i], pressure[i:i + 1, 0], reference_pressure[i], temperature[i], step=step)

    return out


def moist_lapse(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray reference_pressure = None,
    *,
    T step = 1000.0,
    object dtype = None,
):
    """
    Calculate the moist adiabatic lapse rate.

    Parameters
    ----------
    pressure : `np.ndarray` (shape: N | Z | 1 x Z | N x Z)
        Atmospheric pressure level's of interest. Levels must be in decreasing order. `NaN` values
        can be used to mask out values for a particular column.

    temperature : `np.ndarray` (shape: N)
        Starting temperature

    reference_pressure : `np.ndarray (optional)` (shape: N)
        Reference pressure; if not given, it defaults to the first non nan element of the
        pressure array.

    step :
        The step size for the calculation (default: 1000.0).

    dtype :
        The data type for the output array (optional).

    Returns
    -------
    `np.ndarray`
       The resulting parcel temperature at levels given by `pressure`

    Examples:
    ---------
    This function attempts to automaticly resolve the broadcast mode. If all 3 arrays have the same
    shape, and you want to broadcast ``N x Z`` reshape the ``pressure`` array to ``(1, Z)``,
    otherwise the function will execute element-wise.

    >>> import numpy as np
    >>> import nzthermo as nzt
    >>> pressure = np.linspace(100000, 31000, 20)
    >>> temperature = np.random.uniform(300, 220, 20)
    >>> refrence_pressures = np.random.uniform(1001325, 100001, 20)
    >>> nzt.moist_lapse(pressure, temperature, refrence_pressures)
    array([136.21, 193.77, 154.62, ..., 112.51, 155.1 , 119.41])
    >>> nzt.moist_lapse(pressure[np.newaxis, :], temperature, refrence_pressures)
    array([[136.21, 134.78, 133.31, ..., 103.52, 100.61,  97.47],
           [195.83, 193.77, 191.66, ..., 148.83, 144.65, 140.14],
           [157.99, 156.33, 154.62, ..., 120.07, 116.7 , 113.06],
           ...,
           [148.05, 146.49, 144.89, ..., 112.51, 109.35, 105.94],
           [209.97, 207.76, 205.5 , ..., 159.58, 155.1 , 150.27],
           [166.86, 165.11, 163.31, ..., 126.81, 123.25, 119.41]])


    If ``reference_pressure`` is not provided and the pressure array is 2D, the reference pressure
    will be determined by finding the first non-nan value in each row.

    >>> pressure = np.array([
        [1013.12, 1000, 975, 950, 925, 900, ...],
        [1013.93, 1000, 975, 950, 925, 900, ...],
        [np.nan, np.nan, 975, 950, 925, 900, ...]
    ]) * 100.0 # (N, Z) :: pressure profile
    >>> reference_pressure = pressure[np.arange(len(pressure)), np.argmin(np.isnan(pressure), axis=1)]
    >>> reference_pressure
    array([101312., 101393.,  97500.  ])
    """
    cdef size_t N, Z, ndim
    cdef np.ndarray x
    cdef BroadcastMode mode

    if dtype is None:
        dtype = pressure.dtype
    else:
        dtype = np.dtype(dtype)

    if dtype != np.float32 and dtype != np.float64:
        raise ValueError("dtype must be either np.float32 or np.float64.")

    # [ pressure ]
    if (ndim := pressure.ndim) == 1:
        pressure = pressure.reshape(1, -1) # (1, Z)
    elif ndim != 2:
        raise ValueError("pressure must be 1D or 2D array.")

    # [ temperature ]
    temperature = temperature.ravel()  # (N,)
    if not (N := temperature.shape[0]):
        return np.full_like(pressure, NaN, dtype=dtype)

    # [ reference_pressure ]
    if reference_pressure is not None:
        reference_pressure = reference_pressure.ravel()
        if (
            ndim == <size_t> temperature.ndim == <size_t> reference_pressure.ndim
            and pressure.size == temperature.size == reference_pressure.size
        ):
            mode = ELEMENT_WISE  # (N,) (N,) (N,)
            pressure = pressure.reshape(N, 1)
        elif N != <size_t> reference_pressure.shape[0]:
            raise ValueError("reference_pressure and temperature arrays must be the same size.")
        elif 1 == <size_t> pressure.shape[0]:
            mode = BROADCAST    # (1, Z) (N,) (N,)
        elif N == <size_t> pressure.shape[0]:
            mode = MATRIX       # (N, Z) (N,) (N,)
        else:
            raise ValueError("Unable to determine the broadcast mode.")
    # no reference_pressure provided can only be MATRIX or BROADCAST
    elif 2 == ndim and N == <size_t> pressure.shape[0]:
        mode = MATRIX           # (N, Z) (N,)
        reference_pressure = pressure[np.arange(N), np.argmin(np.isnan(pressure), axis=1)]
    elif 1 == pressure.shape[0]:
        mode = BROADCAST        # (1, Z) (N,)
        reference_pressure = np.repeat(pressure[0, np.argmin(np.isnan(pressure[0]))], N)
    else:
        raise ValueError("Unable to determine the broadcast mode.")

    Z = pressure.shape[1]

    x = np.empty((N, Z), dtype=dtype)
    if np.float32 == dtype:
        x[...] = _moist_lapse[float](
            pressure.astype(np.float32), 
            reference_pressure.astype(np.float32),
            temperature.astype(np.float32), 
            step=step,
            mode=mode,
        )
    else:
        x[...] = _moist_lapse[double](
            pressure.astype(np.float64),
            reference_pressure.astype(np.float64),
            temperature.astype(np.float64),
            step=step,
            mode=mode,
        )
    if mode == ELEMENT_WISE:
        x = x.squeeze(1)

    return x

# -------------------------------------------------------------------------------------------------
# lcl
# -------------------------------------------------------------------------------------------------
cdef T[:, :] view_lcl(
    T[:] pressure, T[:] temperature, T[:] dewpoint, T eps, size_t max_iters
) noexcept:
    cdef:
        size_t N, i
        C.pair[T, T] result
        T[:, :] out

    N = pressure.shape[0]
    out = nzarray((2, N), pressure.itemsize)

    with nogil, parallel():
        for i in prange(N, schedule='dynamic'):
            result = C.lcl(pressure[i], temperature[i], dewpoint[i], eps, max_iters)
            out[0, i] = result.pressure
            out[1, i] = result.temperature

    return out


def lcl(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    size_t max_iters = 50,
    T eps = 0.1,
    object dtype = None,
):
    """
    The Lifting Condensation Level (LCL) is the level at which a parcel becomes saturated.
    It is a reasonable estimate of cloud base height when parcels experience forced ascent.
    The height difference between this parameter and the LFC is important when determining
    convection initiation. The smaller the difference between the LCL and the LFC, the more likely
    deep convection becomes. The LFC-LCL difference is similar to CIN (convective inhibition).
    LCL heights from approximately 500 m (1600 ft) to 800 m (2600 ft) above ground level are
    associated with F2 to F5 tornadoes. Low LCL heights and low surface dewpoint depressions
    (high low level RH) suggest a warm RFD which may play a role in tornado development.
    """
    cdef size_t N
    cdef np.ndarray x
    pressure, temperature, dewpoint = map(np.ravel, (pressure, temperature, dewpoint))
    if not pressure.size == temperature.size == dewpoint.size:
        raise ValueError("pressure, temperature, and dewpoint arrays must be the same size.")

    if dtype is None:
        dtype = pressure.dtype
    else:
        dtype = np.dtype(dtype)

    N = pressure.size

    x = np.empty((2, N), dtype=dtype)
    if np.float32 == dtype:
        x[...] = view_lcl[float](
            pressure.astype(np.float32), 
            temperature.astype(np.float32),
            dewpoint.astype(np.float32),
            eps,
            max_iters,
        )
    else:
        x[...] = view_lcl[double](
            pressure.astype(np.float64),
            temperature.astype(np.float64),
            dewpoint.astype(np.float64),
            eps,
            max_iters,
        )

    return x[0], x[1]


# -------------------------------------------------------------------------------------------------
# time
# -------------------------------------------------------------------------------------------------
@cython.ufunc
cdef double delta_t(integer year, integer month) noexcept nogil:
    """
    POLYNOMIAL EXPRESSIONS FOR DELTA T (ΔT)
    see: https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html
    Using the ΔT values derived from the historical record and from direct observations
    (see: Table 1 and Table 2), a series of polynomial expressions have been created to simplify
    the evaluation of ΔT for any time during the interval -1999 to +3000.
    """
    cdef double y, u, delta_t
    y = year + (month - 0.5) / 12

    if year < -500:
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2
    elif year < 500:
        u = y / 100
        delta_t = (
            10583.6
            - 1014.41 * u
            + 33.78311 * u**2
            - 5.952053 * u**3
            - 0.1798452 * u**4
            + 0.022174192 * u**5
            + 0.0090316521 * u**6
        )
    elif year < 1600:
        u = (y - 1000) / 100
        delta_t = (
            1574.2
            - 556.01 * u
            + 71.23472 * u**2
            + 0.319781 * u**3
            - 0.8503463 * u**4
            - 0.005050998 * u**5
            + 0.0083572073 * u**6
        )
    elif year < 1700:
        t = y - 1600
        delta_t = 120 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129
    elif year < 1800:
        t = y - 1700
        delta_t = (
            8.83
            + 0.1603 * t
            - 0.0059285 * t**2
            + 0.00013336 * t**3
            - t**4 / 1174000
        )
    elif year < 1860:
        t = y - 1800
        delta_t = (
            13.72
            - 0.332447 * t
            + 0.0068612 * t**2
            + 0.0041116 * t**3
            - 0.00037436 * t**4
            + 0.0000121272 * t**5
            - 0.0000001699 * t**6
            + 0.000000000875 * t**7
        )
    elif year < 1900:
        t = y - 1860
        delta_t = (
            7.62
            + 0.5737 * t
            - 0.251754 * t**2
            + 0.01680668 * t**3
            - 0.0004473624 * t**4
            + t**5 / 233174
        )
    elif year < 1920:
        t = y - 1900
        delta_t = (
            -2.79
            + 1.494119 * t
            - 0.0598939 * t**2
            + 0.0061966 * t**3
            - 0.000197 * t**4
        )
    elif year < 1941:
        t = y - 1920
        delta_t = 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    elif year < 1961:
        t = y - 1950
        delta_t = 29.07 + 0.407 * t - t**2 / 233 + t**3 / 2547
    elif year < 1986:
        t = y - 1975
        delta_t = 45.45 + 1.067 * t - t**2 / 260 - t**3 / 718
    elif year < 2005:
        t = y - 2000
        delta_t = (
            63.86
            + 0.3345 * t
            - 0.060374 * t**2
            + 0.0017275 * t**3
            + 0.000651814 * t**4
            + 0.00002373599 * t**5
        )
    elif year < 2050:
        t = y - 2000
        delta_t = 62.92 + 0.32217 * t + 0.005589 * t**2
    elif year < 2150:
        delta_t = -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)
    else:
        u = (y - 1820) / 100
        delta_t = -20 + 32 * u**2

    return delta_t


# -------------------------------------------------------------------------------------------------
# parcel_profile
# -------------------------------------------------------------------------------------------------
cdef T[:, :] surface_based_parcel_profile(
    T[:] pressure, T[:] temperature, T[:] dewpoint, T step = 1000.0
) noexcept:
    cdef:
        size_t N, Z, n, z
        T[:, :] profile
        T lcl_p, lcl_t, r, p, t, p0
        T eps = 0.1
        size_t max_iters = 50
        C.pair[T, T] lcl

    N = temperature.shape[0]
    Z = pressure.shape[0]
    profile = nzarray((N, Z), pressure.itemsize)
    with nogil, parallel():
        p0 = pressure[0]
        for n in prange(N, schedule='dynamic'):
            # - parcel temperature from the surface up to the LCL ( dry ascent ) 
            lcl = C.lcl(p0, temperature[n], dewpoint[n], eps, max_iters)
            p = p0
            z = 0
            while p < lcl.pressure:
                profile[n, z] = C.dry_lapse(p, p0, temperature[n])
                p = pressure[z]
                z += 1

            # - parcel temperature from the LCL to the top of the atmosphere ( moist ascent )
            p = lcl.pressure
            t = lcl.temperature
            for z in range(z, Z):
                profile[n, z] = t = C.moist_lapse(p, pressure[z], t, step)
                p = pressure[z]

    return profile

cdef T[:, :] mu_parcel_profile(T[:] pressure, T[:] temperature,  T[:] dewpoint) noexcept:
    cdef: # TODO:...
        size_t N, Z, n, z
        T[:, :] profile
        T lcl_p, lcl_t, r, p, t, p0
        T eps = 0.1
        size_t max_iters = 50
        C.pair[T, T] lcl

    N = temperature.shape[0]
    Z = pressure.shape[0]
    profile = nzarray((N, Z), pressure.itemsize)
    with nogil, parallel():
        p0 = pressure[0]
        for n in prange(N, schedule='dynamic'):
            # - parcel temperature from the surface up to the LCL ( dry ascent ) 
            lcl = C.lcl(p0, temperature[n], dewpoint[n], eps, max_iters)
            p = p0
            z = 0
            while p < lcl.pressure:
                profile[n, z] = C.dry_lapse(p, p0, temperature[n])
                p = pressure[z]
                z += 1

            # - parcel temperature from the LCL to the top of the atmosphere ( moist ascent )
            p = lcl.pressure
            t = lcl.temperature#dewpoint_from_mixing_ratio(lcl_p, r)
            for z in range(z, Z):
                profile[n, z] = t = C.moist_lapse(p, pressure[z], t, <T>1000.0)
                p = pressure[z]

    return profile



def parcel_profile(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    ProfileStrategy strategy = SURFACE_BASED,
):
    cdef np.ndarray profile = np.empty((temperature.size, pressure.size), dtype=pressure.dtype)
    if strategy == SURFACE_BASED:
        if pressure.dtype == np.float64:
            profile[...] = surface_based_parcel_profile[double](pressure, temperature, dewpoint)
        else:
            profile[...] = surface_based_parcel_profile[float](pressure, temperature, dewpoint)
    elif strategy == MOST_UNSTABLE:
        if pressure.dtype == np.float64:
            profile[...] = mu_parcel_profile[double](pressure, temperature, dewpoint)
        else:
            profile[...] = mu_parcel_profile[float](pressure, temperature, dewpoint)

    else:
        raise ValueError("Invalid strategy.")

    return profile
