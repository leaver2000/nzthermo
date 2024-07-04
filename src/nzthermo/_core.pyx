# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
import abc
import warnings
from typing import TypeVar

from cython.parallel cimport parallel, prange

import numpy as np

cimport numpy as np

cimport nzthermo._C as C

from nzthermo._ufunc import (
    between_or_close,
    greater_or_close,
    less_or_close,
    standard_height,
    standard_pressure,
)

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
    bint isnan(long double x) noexcept
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
g  = C.g  # (m/s^2)  - acceleration due to gravity
T0 = C.T0 # (K)      - freezing point of water
E0 = C.E0 # (Pa)     - vapor pressure at T0
Cp = C.Cp # (J/kg*K) - specific heat of dry air
Rd = C.Rd # (J/kg*K) - gas constant for dry air
Rv = C.Rv # (J/kg*K) - gas constant for water vapor
Lv = C.Lv # (J/kg)   - latent heat of vaporization
P0 = C.P0 # (Pa)     - standard pressure at sea level
Mw = C.Mw # (g/mol)  - molecular weight of water
Md = C.Md # (g/mol)  - molecular weight of dry air
epsilon = C.epsilon
kappa = C.kappa

_S = TypeVar("_S")
_T = TypeVar("_T")

# ............................................................................................... #
# helpers
# ............................................................................................... #
class vertical_vector(np.ndarray[_S, np.dtype[_T]], abc.ABC):
    def __new__(cls, x):
        if isinstance(x, cls):
            return x
        x = np.asarray(x).view(cls)

        return x

    @abc.abstractmethod
    def is_above(self, bottom, close=True): ...
    @abc.abstractmethod
    def is_below(self, top, close=True): ...
    @abc.abstractmethod
    def is_between(self, bottom, top, close=True): ...
    @abc.abstractmethod
    def where(self, condition, fill=np.nan):
        return np.where(condition, self, fill).view(self.__class__)



class pressure_vector(vertical_vector[_S, np.dtype[_T]]):
    def is_above(self, bottom, close=False):
        bottom = np.asarray(bottom)
        if not close:
            return np.asarray(self < bottom, np.bool_)

        return np.asarray(less_or_close(self, bottom), np.bool_)

    def is_below(self, top, close=False):
        top = np.asarray(top)
        if not close:
            return np.asarray(self > top, np.bool_)

        return np.asarray(greater_or_close(self, top), np.bool_)

    def is_between(self, bottom, top, close=False):
        bottom, top = np.asarray(bottom), np.asarray(top)
        if not close:
            return np.asarray((self < bottom) & (self > top), np.bool_)

        return np.asarray(between_or_close(self, top, bottom), np.bool_)

    def to_standard_height(self):
        return standard_height(self).view(height_vector)

    @staticmethod
    def from_standard_height(height):
        return standard_pressure(height).view(pressure_vector)



class height_vector(vertical_vector[_S, np.dtype[_T]]):
    def is_above(self, bottom, close=False):
        bottom = np.asarray(bottom)
        if not close:
            return np.asarray(self > bottom, np.bool_)

        return np.asarray(greater_or_close(self, bottom), np.bool_)

    def is_below(self, top, close=False):
        top = np.asarray(top)
        if not close:
            return np.asarray(self < top, np.bool_)

        return np.asarray(less_or_close(self, top), np.bool_)

    def is_between(self, bottom, top, close=False):
        bottom, top = np.asarray(bottom), np.asarray(top)
        if not close:
            return np.asarray((self > bottom) & (self < top), np.bool_)

        return np.asarray(between_or_close(self, bottom, top), np.bool_)

    def to_standard_pressure(self):
        return standard_pressure(self).view(pressure_vector)

    @staticmethod
    def from_standard_pressure(pressure):
        return standard_height(pressure).view(height_vector)


def broadcast_and_nanmask(
    np.ndarray pressure not None,
    np.ndarray temperature not None, 
    np.ndarray dewpoint, 
    long ndim = 2,
    object where = None,
    object dtype = None,
):
    """
    TODO: ideally this function needs to be exposed to the `core.py` or implmented as part of the
    decorator, so that it is only ever called once. As part of the masking operation it would be
    useful to properly mask and sort the data once per function call.
    

    >>> where &= np.isfinite(temperature) & np.isfinite(dewpoint) & np.isfinite(pressure)

    Validates the input arrays and determines the broadcast mode. by brodcasting the ``pressure``, 
    ``temperature``, and ``dewpoint`` arrays to a suitable shape for the calling function.

    In some cases temperature and dewpoint arrays can be 1D, if this is the case the ``ndim``
    argument must be set to 1.

    masks the data with nan values and sorts in descending pressure order (bottom -> top) with nan
    values at the end, this allows us to short circuit some computations as some of the functions
    will stop at the first nan value.

    Insures that all arays are in ``C_CONTIGUOUS`` memory layout. This is important for the Cython
    memoryview to work correctly.
    """
    cdef:
        size_t N, Z
        BroadcastMode mode

    if temperature.ndim != ndim:
        raise ValueError(f"temperature must be a {ndim}D array.")
    elif dewpoint.ndim != ndim:
        raise ValueError(f"dewpoint must be a {ndim}D array.")
    elif ndim == 2:
        if temperature.shape[0] != dewpoint.shape[0]:
            raise ValueError("temperature and dewpoint must have the same number of rows.")
        elif temperature.shape[1] != dewpoint.shape[1]:
            raise ValueError("temperature and dewpoint must have the same number of columns.")
    elif ndim == 1:
        if temperature.shape[0] != dewpoint.shape[0]:
            raise ValueError("temperature and dewpoint must have the same number of elements.")

    if pressure.ndim == 1 or pressure.shape[0] == 1:
        pressure = pressure.reshape(1, -1)
        mode = BROADCAST
    else:
        mode = MATRIX

    if where is not None:
        if not isinstance(where, np.ndarray):
            raise ValueError("where must be a numpy array.")

        mode = MATRIX
        N = temperature.shape[0]
        Z = temperature.shape[1]

        where = np.atleast_2d(where)
        # where &= np.isfinite(temperature) & np.isfinite(dewpoint) & np.isfinite(pressure)

        pressure = np.broadcast_to(pressure.squeeze(), (N, Z))
        pressure = np.where(where, pressure, -np.inf)

        sort = np.arange(N)[:, np.newaxis], np.argsort(pressure, axis=1)
        pressure = pressure[sort][:, ::-1]
        temperature = temperature[sort][:, ::-1]
        dewpoint = dewpoint[sort][:, ::-1]
        m = np.isneginf(pressure)
        pressure[m] = np.nan
        temperature[m] = np.nan
        dewpoint[m] = np.nan

    if dtype is None:
        dtype = temperature.dtype
    
    if dtype != np.float32 and dtype != np.float64:
        
        warnings.warn(f"the dtype {dtype} is not supported, defaulting to np.float64.")
        dtype = np.float64
    
    pressure = np.ascontiguousarray(pressure, dtype=dtype)
    temperature = np.ascontiguousarray(temperature, dtype=dtype)
    dewpoint = np.ascontiguousarray(dewpoint, dtype=dtype)

    return (pressure, temperature, dewpoint), mode, dtype


# need to figure out a way to possibly pass in **kwargs maybe via a options struct
ctypedef T (*Dispatch)(const T*, const T*, const T*, size_t) noexcept nogil

cdef T[:] dispatch(
    Dispatch fn, 
    const T[:, :] pressure, 
    const T[:, :] temperature, 
    const T[:, :] dewpoint,
    const BroadcastMode mode
) noexcept:
    """
    ```python
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
    out = np.empty((N,), dtype=np.dtype(f"f{pressure.itemsize}"))
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
    T[:] out, T[:] pressure, T reference_pressure, T temperature
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
            out[i] = temperature = C.moist_lapse(reference_pressure, next_pressure, temperature)
            reference_pressure = next_pressure


cdef T[:, :] moist_lapse_2d(
    T[:, :] pressure, 
    T[:] reference_pressure, 
    T[:] temperature, 
    BroadcastMode mode,
) noexcept:
    cdef:
        size_t N, Z, i
        T[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]
    out = np.empty((N, Z), dtype=np.dtype(f"f{pressure.itemsize}"))
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d(
                    out[i], pressure[0, :], reference_pressure[i], temperature[i]
                )
        else: # MATRIX
            for i in prange(N, schedule='dynamic'):
                moist_lapse_1d(
                    out[i], pressure[i, :], reference_pressure[i], temperature[i]
                )

    return out


def moist_lapse(
    np.ndarray pressure not None,
    np.ndarray temperature not None,
    np.ndarray reference_pressure = None,
    *,
    object dtype = None,
    np.ndarray where = None,
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
    >>> reference_pressure = pressure[np.arange(pressure.shape[0]), np.argmin(np.isnan(pressure), axis=1)]
    >>> reference_pressure
    array([101312., 101393.,  97500.  ])
    """
    cdef:
        size_t N, Z, ndim
        BroadcastMode mode
        np.ndarray out

    if where is not None:
        raise NotImplementedError("where argument is not supported.")
        
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
    out = np.empty((N, Z), dtype=dtype)
    if np.float32 == dtype:
        out[...] = moist_lapse_2d[float](
            pressure.astype(np.float32), 
            reference_pressure.astype(np.float32),
            temperature.astype(np.float32), 
            mode=mode,
        )
    else:
        out[...] = moist_lapse_2d[double](
            pressure.astype(np.float64),
            reference_pressure.astype(np.float64),
            temperature.astype(np.float64),
            mode=mode,
        )
    if mode == ELEMENT_WISE:
        out = out.squeeze(1)

    return out


# ............................................................................................... #
# parcel_profile
# ............................................................................................... #
cdef void parcel_profile_1d(
    T[:] out,      # (Z,)
    T[:] pressure,  # (Z,)
    T temperature,
    T dewpoint,
) noexcept nogil:
    cdef:
        size_t Z, i, stop
        T p0, reference_pressure, next_pressure
        C.lcl[T] lcl

    Z = out.shape[0]
    p0 = pressure[0]
    lcl = C.lcl[T](pressure[0], temperature, dewpoint)

    # [dry ascent]
    # stop the dry ascent at the LCL
    stop = lcl.index_on(&pressure[0], Z)
    for i in prange(0, stop, schedule='dynamic'): # parallelize the dry ascent
        out[i] = C.dry_lapse(pressure[i], p0, temperature)

    # [ moist ascent ]
    if stop != Z:
        moist_lapse_1d(out[stop:], pressure[stop:], lcl.pressure, lcl.temperature)


cdef T[:, :] parcel_profile_2d(
    T[:, :] pressure,
    T[:] temperature,
    T[:] dewpoint,
    BroadcastMode mode,
) noexcept:
    cdef:
        size_t N, Z, i
        T[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]
    out = np.empty((N, Z), dtype=np.dtype(f"f{pressure.itemsize}"))
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                parcel_profile_1d(out[i], pressure[0, :], temperature[i], dewpoint[i])

        else: # MATRIX
            for i in prange(N, schedule='dynamic'):
                parcel_profile_1d(out[i], pressure[i, :], temperature[i], dewpoint[i])

    return out


def parcel_profile(
    np.ndarray pressure not None,
    np.ndarray temperature not None,
    np.ndarray dewpoint not None,
    *,
    np.ndarray where = None,
):
    cdef:
        size_t N, Z
        BroadcastMode mode
        np.ndarray out

    (pressure, temperature, dewpoint), mode, dtype = broadcast_and_nanmask(
        pressure, temperature, dewpoint, ndim=1, where=where
    )

    N, Z = temperature.shape[0], pressure.shape[1]
    out = np.empty((N, Z), dtype=dtype)
    if dtype == np.float64:
        out[...] = parcel_profile_2d[double](pressure, temperature, dewpoint, mode)
    else:
        out[...] = parcel_profile_2d[float](pressure, temperature, dewpoint, mode)

    return out


# ............................................................................................... #
# parcel_profile_with_lcl
# ............................................................................................... #
cdef void parcel_profile_with_lcl_1d(
    T[:] ep,        # environment pressure (Z + 1,)
    T[:] et,        # environment temperature (Z + 1,)
    T[:] etd,       # environment dewpoint (Z + 1,)
    T[:] pt,        # parcel temperature (Z + 1,)
    # 
    T[:] pressure,  # (Z,)
    T[:] temperature,
    T[:] dewpoint,
) noexcept nogil:
    cdef:
        size_t Z, i, stop
        T p0, t0, reference_pressure, next_pressure
        C.lcl[T] lcl

    Z = pressure.shape[0]
    p0, t0, td0 = pressure[0], temperature[0], dewpoint[0]

    lcl = C.lcl[T](p0, t0, td0)

    # [dry ascent] .. parcel temperature from the surface up to the LCL ..
    stop = lcl.index_on(&pressure[0], Z)
    
    if stop:
        ep[:stop] = pressure[:stop]
        et[:stop] = temperature[:stop]
        etd[:stop] = dewpoint[:stop]
        for i in prange(0, stop, schedule='dynamic'): # parallelize the dry ascent
            pt[i] = C.dry_lapse(pressure[i], p0, t0)

        # [ lcl ]
        ep[stop] = lcl.pressure
        et[stop] = C.linear_interpolate(
            lcl.pressure, 
            pressure[stop - 1], 
            pressure[stop], 
            temperature[stop - 1], 
            temperature[stop]
        )
        etd[stop] = C.linear_interpolate(
            lcl.pressure, 
            pressure[stop - 1], 
            pressure[stop], 
            dewpoint[stop - 1], 
            dewpoint[stop]
        )
        pt[stop] = lcl.temperature
    else:
        # the lcl was found at the surface which is a little odd. In the metpy implementation
        # this causes interpolation warnings, but we can just set the values to the surface values
        ep[0] = lcl.pressure
        et[0] = t0
        etd[0] = td0
        pt[0] = lcl.temperature

    # [ moist ascent ] .. parcel temperature from the LCL to the top of the atmosphere ..
    if stop != Z:
        ep[stop + 1:] = pressure[stop:]
        et[stop + 1:] = temperature[stop:]
        etd[stop + 1:] = dewpoint[stop:]
        moist_lapse_1d(pt[stop + 1:], pressure[stop:], lcl.pressure, lcl.temperature)


cdef T[:, :, :] parcel_profile_with_lcl_2d(
    T[:, :] pressure,
    T[:, :] temperature,
    T[:, :] dewpoint,
    BroadcastMode mode,
) noexcept:
    cdef:
        size_t N, Z, i, idx
        T[:, :] ep, et, etd, pt

    N, Z = temperature.shape[0], pressure.shape[1] + 1
    ep, et, etd, pt = np.full((4, N, Z), fill_value=NaN, dtype=np.dtype(f"f{pressure.itemsize}"))
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                parcel_profile_with_lcl_1d(
                    ep[i, :],
                    et[i, :],
                    etd[i, :],
                    pt[i, :],
                    pressure[0, :], # broadcast 1d pressure array
                    temperature[i, :], 
                    dewpoint[i, :],
                )
        else: # MATRIX
            for i in prange(N, schedule='dynamic'):
                parcel_profile_with_lcl_1d(
                    ep[i, :],
                    et[i, :],
                    etd[i, :],
                    pt[i, :],
                    pressure[i, :],
                    temperature[i, :],
                    dewpoint[i, :],
                )

    return np.asarray([ep, et, etd, pt])


def parcel_profile_with_lcl(
    np.ndarray pressure not None,
    np.ndarray temperature not None,
    np.ndarray dewpoint not None,
    *,
    np.ndarray where = None,
):
    cdef:
        size_t N, Z
        BroadcastMode mode
        np.ndarray out

    (pressure, temperature, dewpoint), mode, dtype = broadcast_and_nanmask(
        pressure, temperature, dewpoint, ndim=2, where=where
    )

    N, Z = temperature.shape[0], pressure.shape[1]
    out = np.empty((4, N, Z + 1), dtype=dtype)
    if dtype == np.float64:
        out[...] = parcel_profile_with_lcl_2d[double](pressure, temperature, dewpoint, mode)
    else:
        out[...] = parcel_profile_with_lcl_2d[float](pressure, temperature, dewpoint, mode)

    return out[0].view(pressure_vector), out[1], out[2], out[3]


# ............................................................................................... #
# vertical search
# ............................................................................................... #
cdef size_t[:] index_pressure_2d(T[:, :] pressure, T[:] values, BroadcastMode mode):
    cdef:
        size_t i, N, Z
        size_t[:] out 

    N, Z = values.shape[0], pressure.shape[1]
    out = np.empty(N, dtype=np.uintp)
    with nogil, parallel():
        if BROADCAST is mode:
            for i in prange(N, schedule='dynamic'):
                out[i] = C.index_pressure(&pressure[0, 0], values[i], Z)
        else: # MATRIX
            for i in prange(N, schedule='dynamic'):
                out[i] = C.index_pressure(&pressure[i, 0], values[i], Z)

    return out


def index_pressure(np.ndarray pressure not None, np.ndarray values not None):
    cdef size_t N
    N = values.shape[0]
    out = np.empty(N, dtype=np.uintp)
    pressure = pressure.squeeze()
    if pressure.ndim == 1:
        pressure = pressure.reshape(1, -1)
        mode = BROADCAST
    else:
        mode = MATRIX

    if pressure.dtype == np.float64:
        out[...] = index_pressure_2d[double](
            pressure.astype(np.float64), values.astype(np.float64), mode
        )
    else:
        out[...] = index_pressure_2d[float](
            pressure.astype(np.float32), values.astype(np.float32), mode
        )

    return out


