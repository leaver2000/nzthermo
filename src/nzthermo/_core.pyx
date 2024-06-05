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

# ............................................................................................... #
# helpers
# ............................................................................................... #
cdef cvarray nzarray((size_t, size_t) shape, size_t size):
    return cvarray(
        shape, 
        itemsize=size, 
        format='f' if size == sizeof(float) else 'd', 
        mode='c',
        allocate_buffer=True,
    )


cdef pressure_mode(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
):
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)
        mode = BROADCAST
    elif pressure.ndim == 2 and pressure.shape[0] == 1:
        mode = BROADCAST
    elif pressure.ndim == 2 and pressure.shape == temperature.shape == dewpoint.shape:
        mode = MATRIX
    else:
        raise ValueError("Invalid pressure array shape.")
    
    return (pressure, temperature, dewpoint), mode

# need to figuoure out a way to possibly pass in **kwargs maybe via a options struct
ctypedef T (*Dispatch)(const T*, const T*, const T*, size_t) noexcept nogil

cdef T[:] dispatch(
    Dispatch fn, 
    const T[:, :] pressure, 
    const T[:, :] temperature, 
    const T[:, :] dewpoint,
    const BroadcastMode mode
) noexcept:
    """
    ```
    def cape_cin(
        np.ndarray pressure,
        np.ndarray temperature,
        np.ndarray dewpoint,
    ):
        cdef np.ndarray profile = np.empty(temperature.shape[0], dtype=np.float64)
        cdef double[:] p = pressure.astype(np.float64)
        cdef double[:, :] t = temperature.astype(np.float64)
        cdef double[:, :] td = dewpoint.astype(np.float64)
        profile[...] = dispatch[double](C.cape_cin[double], p, t, td)
        return profile
    ```
    """
    cdef:
        size_t N, Z, i
        T[:] out

    N, Z = temperature.shape[0], pressure.shape[1]
    out = np.empty((N,), dtype=np.float64 if sizeof(double) == pressure.itemsize else np.float32)

    with nogil:
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                out[i] = fn(&pressure[0, 0], &temperature[i, 0], &dewpoint[i, 0], Z)
        else:
            for i in prange(N, schedule='dynamic'):
                out[i] = fn(&pressure[i, 0], &temperature[i, 0], &dewpoint[i, 0], Z)

    return out


# ............................................................................................... #
# moist adibatic lapse rate
# ............................................................................................... #
cdef void moist_lapse_1d(
    T[:] out, T[:] pressure, T reference_pressure, T temperature, T step
) noexcept nogil:
    """Moist adiabatic lapse rate for a 1D array of pressure levels."""
    cdef:
        size_t Z, i
        T next_pressure

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


cdef T[:, :] view_moist_lapse(
    T[:, :] pressure, 
    T[:] reference_pressure, 
    T[:] temperature, 
    BroadcastMode mode,
    T step, 
) noexcept:
    cdef:
        size_t N, Z, i
        T[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]

    out = nzarray((N, Z), pressure.itemsize)
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d(out[i], pressure[0, :], reference_pressure[i], temperature[i], step=step)
        else: # MATRIX
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d(out[i], pressure[i, :], reference_pressure[i], temperature[i], step=step)
        # else: # ELEMENT_WISE TODO: this is deprecated in favor of the C.moist_lapse ufunc
        #     for i in prange(N, schedule='dynamic'):
        #         moist_lapse_1d(out[i], pressure[i:i + 1, 0], reference_pressure[i], temperature[i], step=step)

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
    cdef:
        size_t N, Z, ndim
        BroadcastMode mode
        np.ndarray x

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
        x[...] = view_moist_lapse[float](
            pressure.astype(np.float32), 
            reference_pressure.astype(np.float32),
            temperature.astype(np.float32), 
            mode=mode,
            step=step,
        )
    else:
        x[...] = view_moist_lapse[double](
            pressure.astype(np.float64),
            reference_pressure.astype(np.float64),
            temperature.astype(np.float64),
            mode=mode,
            step=step,
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
cdef void parcel_profile_1d(
    T[:] out,      # (Z,)
    T[:] pressure,  # (Z,)
    T temperature,
    T dewpoint,
    T step = 1000.0,
    T eps = 0.1,
    size_t max_iters = 50,
) noexcept nogil:
    cdef:
        size_t Z, z, stop
        T p0, t0, t, p
        C.LCL[T] lcl

    Z = pressure.shape[0]
    p0 = pressure[0]
    t0 = out[0] = temperature

    lcl = C.lcl(p0, t0, dewpoint, eps, max_iters)

    # [dry ascent] 
    # parcel temperature from the surface up to the LCL
    z = 1 # we start at the second level
    while pressure[z] >= lcl.pressure:
        z += 1

    stop = z # stop the dry ascent at the LCL
    for z in prange(1, stop, schedule='dynamic'): # parallelize the dry ascent
        out[z] = C.dry_lapse(pressure[z], p0, t0)

    # [ moist ascent ]
    # parcel temperature from the LCL to the top of the atmosphere ( moist ascent )
    p, t = lcl.pressure, lcl.temperature
    for z in range(stop, Z):
        out[z] = t = C.moist_lapse(p, pressure[z], t, step)
        p = pressure[z]


cdef T[:, :] view_parcel_profile(
    T[:, :] pressure,
    T[:] temperature,
    T[:] dewpoint,
    BroadcastMode mode,
    T step = 1000.0,
    T eps = 0.1,
    size_t max_iters = 50,
) noexcept:
    cdef:
        size_t N, Z, i
        T[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]
    out = nzarray((N, Z), pressure.itemsize)

    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                parcel_profile_1d(out[i], pressure[0, :], temperature[i], dewpoint[i], step, eps, max_iters)

        elif MATRIX is mode:
            for i in prange(N, schedule='dynamic'):
                parcel_profile_1d(out[i], pressure[i, :], temperature[i], dewpoint[i], step, eps, max_iters)
 
    return out


def parcel_profile(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    ProfileStrategy strategy = SURFACE_BASED,
    double step = 1000.0,
    double eps = 0.1,
    size_t max_iters = 50,
):
    cdef:
        size_t N, Z
        BroadcastMode mode
        np.ndarray out

    (pressure, temperature, dewpoint), mode = pressure_mode(pressure, temperature, dewpoint)
    N, Z = temperature.shape[0], pressure.shape[1]
        

    out = np.empty((N, Z), dtype=pressure.dtype)
    if strategy == SURFACE_BASED:
        if pressure.dtype == np.float64:
            out[...] = view_parcel_profile[double](
                pressure, temperature, dewpoint, mode, step, eps, max_iters
            )
        else:
            out[...] = view_parcel_profile[float](
                pressure, temperature, dewpoint, mode, <float>step, <float>eps, max_iters
            )

    else:
        raise ValueError("Invalid strategy.")
    
    return out


cdef T[:, :] surface_based_parcel_profile_with_lcl(
    T[:, :] profile,
    T[:] lcl_temperature,
    T[:] lcl_dewpoint, 
    size_t[:] lcl_index,
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
            
            profile[n, z] = t
            
            for z in range(z, Z):
                profile[n, z + 1] = t = C.moist_lapse(p, pressure[z], t, step)
                p = pressure[z]

    return profile


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


# def cape_cin(
#     np.ndarray pressure,
#     np.ndarray temperature,
#     np.ndarray dewpoint,
# ):
#     cdef np.ndarray profile = np.empty(temperature.shape[0], dtype=np.float64)
#     cdef double[:] p = pressure.astype(np.float64)
#     cdef double[:, :] t = temperature.astype(np.float64)
#     cdef double[:, :] td = dewpoint.astype(np.float64)
#     profile[...] = dispatch[double](C.cape_cin[double], p, t, td)
#     return profile


# ............................................................................................... #
cdef T[:] _interpolate_nz(
    T[:] x,      # (N,)
    T[:] xp,     # (Z,)
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
        for n in prange(N, schedule='runtime'):
            out[n] = C.interpolate_z(Z, x[n], &xp[0], &fp[n, 0])

    return out


def interpolate_nz(
    np.ndarray __x,
    np.ndarray __xp,
    *args,
    bint log_x = 0,
    bint interp_nan = 0
):
    """
    Interpolates values for multiple batches of data.

    Args:
        x: Input array of shape (N,) containing the values to be interpolated.
        xp: Input array of shape (Z,) containing the reference values.
        *args: Variable number of input arrays of shape (N, Z) containing additional data.

    Returns:
        np.ndarray or tuple of np.ndarray: Interpolated values for each batch.

    Raises:
        None

    Examples:
    >>> import numpy as np
    >>> import nzthermo as nzt
    >>> import nzthermo.functional as F
    >>> temperature = np.array(
    ...     [
    ...         [303.3, 302.36, 300.16, 298.0, 296.09, 296.73, 295.96, 294.79, 293.51, 291.81],
    ...         [303.58, 302.6, 300.41, 298.24, 296.49, 295.35, 295.62, 294.43, 293.27, 291.6],
    ...         [303.75, 302.77, 300.59, 298.43, 296.36, 295.15, 295.32, 294.19, 292.84, 291.54],
    ...         [303.46, 302.51, 300.34, 298.19, 296.34, 295.51, 295.06, 293.84, 292.42, 291.1],
    ...         [303.23, 302.31, 300.12, 297.97, 296.28, 295.68, 294.83, 293.67, 292.56, 291.47],
    ...     ]
    ... )  # (N, Z)
    >>> dewpoint = np.array(
    ...     [
    ...         [297.61, 297.36, 296.73, 296.05, 294.69, 289.18, 286.82, 285.82, 284.88, 283.81],
    ...         [297.62, 297.36, 296.79, 296.18, 294.5, 292.07, 287.74, 286.67, 285.15, 284.02],
    ...         [297.76, 297.51, 296.91, 296.23, 295.05, 292.9, 288.86, 287.12, 285.99, 283.98],
    ...         [297.82, 297.56, 296.95, 296.23, 295.0, 292.47, 289.97, 288.45, 287.09, 285.17],
    ...         [298.22, 297.95, 297.33, 296.69, 295.19, 293.16, 291.42, 289.66, 287.28, 284.31],
    ...     ]
    ... )  # (N, Z)
    >>> surface_pressure = np.array([101210.0, 101300.0, 101373.0, 101430.0, 101470.0])  # (N,)
    >>> pressure_levels = np.array(
    ...     [101300.0, 100000.0, 97500.0, 95000.0, 92500.0, 90000.0, 87500.0, 85000.0, 82500.0, 80000.0]
    ... )  # (Z,)
    >>> lcl_p, lcl_t = nzt.lcl(
    ...     surface_pressure,  # (N,)
    ...     temperature[:, 0],  # (N,)
    ...     dewpoint[:, 0],  # (N,)
    ... )  # (N,), (N,)
    >>> lcl_p
    array([93214.26240694, 92938.06420037, 92967.83292536, 93487.43780492, 94377.76028999])
    >>> F.interpolate_nz(lcl_p, pressure_levels, temperature, dewpoint)  # temp & dwpt values interpolated at LCL pressure
    (
        array([296.63569648, 296.79664494, 296.74736566, 297.07070398, 297.54936596]),
        array([295.07855875, 294.79437914, 295.27081714, 295.4858194, 296.31665617])
    )

    """
    dtype = __x.dtype
    cdef np.ndarray xp = np.asarray(__xp, dtype=dtype)
    cdef np.ndarray fp = np.asarray(args, dtype=dtype)
    cdef np.ndarray out = np.empty((fp.shape[0], __x.shape[0]), dtype=dtype)
    
    for i in range(fp.shape[0]):
        if dtype == np.float64:
            out[i] = _interpolate_nz[double](__x, xp, fp[i], log_x)
        else:
            out[i] = _interpolate_nz[float](__x, xp, fp[i], log_x)

        if interp_nan:            
            mask = np.isnan(out[i])
            out[i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), __x[~mask])

    if out.shape[0] == 1:
        return out[0]

    return tuple(out)
    
# ............................................................................................... #
# this is actually not implemented in the C++ code
# ............................................................................................... #
# def downdraft_cape(
#     np.ndarray pressure,
#     np.ndarray temperature,
#     np.ndarray dewpoint,
# ):  
#     cdef:
#         double[:, :] p, t, td
#         size_t size = temperature.shape[1]
#     p = pressure.astype(np.float64)
#     t = dewpoint.astype(np.float64)
#     td = temperature.astype(np.float64)
#     out = np.empty(temperature.shape[0], dtype=np.float64)

#     for i in range(temperature.shape[0]):
#         out[i] = C.downdraft_cape[double](&p[i, 0], &t[i, 0], &td[i, 0], size)
        
#     return out
    