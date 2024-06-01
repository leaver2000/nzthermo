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

from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray
# from libcpp.memory cimport unique_ptr, make_unique
# from libcpp.vector cimport vector

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
        C.LCL[T] result
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
# parcel_profile
# -------------------------------------------------------------------------------------------------
cdef surface_based_parcel_profile(
    T[:] pressure, T[:] temperature, T[:] dewpoint, T step = 1000.0, T eps = 0.1, size_t max_iters = 50
) noexcept:
    cdef:
        size_t N, Z, n, z
        T r, p, t, td, p0
        T[:, :] profile_t
        C.LCL[T] lcl
        size_t[:] lcl_index
        T[:] lcl_p
        T[:] lcl_t

    N = temperature.shape[0]
    Z = pressure.shape[0]

    profile_t = nzarray((N, Z), pressure.itemsize)
    dtype = np.float32 if sizeof(float) == pressure.itemsize else np.float64
    lcl_p, lcl_t, lcl_index = np.empty(N, dtype=dtype), np.empty(N, dtype=dtype), np.empty(N, dtype=np.uintp)
    profile_t[:, 0] = temperature
    
    with nogil, parallel():
        p0 = pressure[0]
        for n in prange(N, schedule='dynamic'):
            # compute lcl
            t, td = temperature[n], dewpoint[n]
            lcl = C.lcl(p0, t, td, eps, max_iters)
            lcl_p[n], lcl_t[n] = lcl.pressure, lcl.temperature
            
            # - parcel temperature from the surface up to the LCL ( dry ascent ) 
            p, z = pressure[1], 1 # we start at the second level
            while p >= lcl_p[n]:
                profile_t[n, z] = C.dry_lapse(p, p0, t)
                z += 1
                p = pressure[z]
            
            lcl_index[n] = z
            p, t = lcl_p[n], lcl_t[n]
            for z in range(z, Z):
                profile_t[n, z] = t = C.moist_lapse(p, pressure[z], t, step)
                p = pressure[z]

    return {
        "temperature": np.asarray(profile_t),
        "lcl":{
            "pressure": lcl_p,
            "temperature": lcl_t,
            "index": lcl_index
        }
    }


def parcel_profile(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    ProfileStrategy strategy = SURFACE_BASED,
):
    # cdef np.ndarray profile = np.empty((temperature.size, pressure.size), dtype=pressure.dtype)
    if strategy == SURFACE_BASED:
        if pressure.dtype == np.float64:
            return surface_based_parcel_profile[double](pressure, temperature, dewpoint)
        else:
            return  surface_based_parcel_profile[float](pressure, temperature, dewpoint)

    # else:
    raise ValueError("Invalid strategy.")


cdef T[:, :] surface_based_parcel_profile_with_lcl(
    T[:, :] profile,
    T[:] lcl_temperature,
    T[:] lcl_dewpoint, 
    size_t[:] lcl_index,
    # LCL[:] lcl_results,
    T[:] pressure,
    T[:] temperature,
    T[:] dewpoint,
    T step = 1000.0
) noexcept:
    cdef:
        size_t N, Z, n, z
        # T[:, :] profile
        T[:, :] lcl_values
        T lcl_p, lcl_t, r, p, t, p0
        T eps = 0.1
        size_t max_iters = 50
        C.LCL[T] lcl

    
    N = temperature.shape[0]
    Z = pressure.shape[0]
    assert Z == profile.shape[1] - 1
    # profile = nzarray((N, Z), pressure.itemsize)
    # lcl_values = nzarray((N, Z), pressure.itemsize)

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
            # lcl_temperature[n] = p = lcl[0]
            p = lcl.pressure
            t = lcl.temperature
            # lcl_dewpoint[n] = t = lcl[1]
            # lcl_index[n] = z
            profile[n, z] = t
            # z+=1
            for z in range(z, Z):
                profile[n, z + 1] = t = C.moist_lapse(p, pressure[z], t, step)
                p = pressure[z]

    return profile


# cdef packed struct LCL:
#     float pressure
#     float temperature
#     size_t index

def parcel_profile_with_lcl(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    ProfileStrategy strategy = SURFACE_BASED,
):
    cdef np.ndarray profile = np.empty((temperature.size, pressure.size + 1), dtype=pressure.dtype)
    cdef np.ndarray lcl_temperature = np.empty(temperature.size, dtype=pressure.dtype)
    cdef np.ndarray lcl_dewpoint = np.empty(temperature.size, dtype=pressure.dtype)
    cdef np.ndarray lcl_index = np.empty(temperature.size, dtype=np.uintp)
    
    if strategy == SURFACE_BASED:
        if pressure.dtype == np.float64:
            surface_based_parcel_profile_with_lcl[double](
                profile,
                lcl_temperature,
                lcl_dewpoint,
                lcl_index,
                pressure, temperature, dewpoint
            )
        else:
            surface_based_parcel_profile_with_lcl[float](
                profile,
                lcl_temperature,
                lcl_dewpoint,
                lcl_index,
                pressure, temperature, dewpoint
            )
    else:
        raise ValueError("Invalid strategy.")
    # elif strategy == MOST_UNSTABLE:
        # if pressure.dtype == np.float64:
        #     profile[...] = mu_parcel_profile[double](pressure, temperature, dewpoint)
        # else:
        #     profile[...] = mu_parcel_profile[float](pressure, temperature, dewpoint)

    return profile
# ............................................................................................... #
cdef T[:] _interpolate_nz(
    T[:] x,     # (N,)
    T[:] xp,    # (Z,)
    T[:, :] fp,  # (N, Z)
    bint log_x = 0,
) noexcept:
    cdef:
        size_t N, Z, n
        T[:] out

    N = x.shape[0]
    Z = xp.shape[0]
    out = np.empty(N, dtype=np.float32 if sizeof(float) == x.itemsize else np.float64)
    
    with nogil, parallel():
        for n in prange(N, schedule='dynamic'):
            out[n] = C.interpolate_z(Z, x[n], &xp[0], &fp[n, 0])

    return out


def interpolate_nz(
    np.ndarray x,
    np.ndarray xp,
    *args,
    bint log_x = 0,
):
    cdef np.ndarray out = np.empty((len(args), x.shape[0]), dtype=x.dtype)
    
    for i in range(len(args)):
        if x.dtype == np.float64:
            out[i] = _interpolate_nz[double](x, xp, args[i], log_x)
        else:
            out[i] = _interpolate_nz[float](x, xp, args[i], log_x)
        
    
    if out.shape[0] == 1:
        return out[0]

    return out
# ............................................................................................... #