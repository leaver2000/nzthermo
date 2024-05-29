# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
cimport cython
import cython
from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np

from . import const as _const

np.import_array()
np.import_ufunc()
OPENMP_ENABLED = bool(OPENMP)

# -------------------------------------------------------------------------------------------------
# constant declarations
# -------------------------------------------------------------------------------------------------
cdef:
    float Rd        = _const.Rd
    float Rv        = _const.Rv
    float Lv        = _const.Lv
    float Cpd       = _const.Cpd
    float epsilon   = _const.epsilon
    float T0        = _const.T0
    float E0        = _const.E0

del _const


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
cdef floating saturation_mixing_ratio(floating pressure, floating temperature) noexcept nogil:
    cdef floating P
    P = E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65)) # saturation vapor pressure
    return (Rd / Rv) * P / (pressure - P)


cdef floating vapor_pressure(floating pressure, floating mixing_ratio) noexcept nogil:
    return pressure * mixing_ratio / (Rd / Rv + mixing_ratio)


cdef floating _dewpoint(floating vapor_pressure) noexcept nogil:
    cdef floating ln
    ln = log(vapor_pressure / E0)
    return T0 + 243.5 * ln / (17.67 - ln)


cdef floating mixing_ratio(
    floating partial_press, floating total_press, floating molecular_weight_ratio = Rd / Rv
) noexcept nogil:
    return molecular_weight_ratio * partial_press / (total_press - partial_press)


cdef floating saturation_vapor_pressure(floating temperature) noexcept nogil:
    """
    >>> saturation_mixing_ratio = mixing_ratio(saturation_vapor_pressure(temperature), pressure)
    """  
    return E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65))


# -------------------------------------------------------------------------------------------------
# moist_lapse
# -------------------------------------------------------------------------------------------------
cdef floating moist_lapse_solver(floating pressure, floating temperature) noexcept nogil:
    cdef floating r
    r = saturation_mixing_ratio(pressure, temperature)
    r = (Rd * temperature + Lv * r) / (Cpd + (Lv * Lv * r * epsilon / (Rd * temperature**2)))
    r /= pressure
    return r


cdef floating moist_lapse_integrator(
    floating pressure, floating next_pressure, floating temperature, floating step
) noexcept nogil:
    """``2nd order Runge-Kutta (RK2)``

    Integrate the moist lapse rate ODE, using a RK2 method, from an initial pressure lvl
    to a final one.
    """
    cdef int N
    cdef floating delta, abs_delta, k1

    N = 1
    delta = next_pressure - pressure
    if (abs_delta := abs(delta)) > step:
        N =  <int> ceil(abs_delta / step)
        delta = delta / <floating> N

    for _ in range(N):
        k1 = delta * moist_lapse_solver(pressure, temperature)
        temperature += delta * moist_lapse_solver(pressure + delta * 0.5, temperature + k1 * 0.5)
        pressure += delta

    return temperature


cdef void moist_lapse_1d_(
    floating[:] out, floating[:] pressure, floating reference_pressure, floating temperature, floating step
) noexcept nogil:
    """Moist adiabatic lapse rate for a 1D array of pressure levels."""
    cdef size_t Z, i
    cdef floating next_pressure

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
            out[i] = temperature = moist_lapse_integrator(
                reference_pressure, next_pressure, temperature, step=step
            )
            reference_pressure = next_pressure


cdef floating[:, :] _moist_lapse(
    floating[:, :] pressure, 
    floating[:] reference_pressure, 
    floating[:] temperature, 
    floating step,
    BroadcastMode mode,
) noexcept:
    cdef size_t N, Z, i
    cdef floating[:, :] out

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
    floating step = 1000.0,
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
cdef floating lcl_solver(
    floating pressure, floating reference_pressure, floating temperature, floating mixing_ratio
) noexcept nogil:
    cdef floating Td, P

    Td = _dewpoint(vapor_pressure(pressure, mixing_ratio))
    if isnan(P := reference_pressure * (Td / temperature) ** (1.0 / (Rd / Cpd))):
        return pressure

    return P


cdef floating lcl_integrator(
    floating pressure, floating temperature, floating mixing_ratio, size_t max_iters, floating eps
) noexcept nogil:
    cdef floating p0, p1, p2, delta, err

    p0 = pressure
    for _ in range(max_iters):
        p1 = lcl_solver(p0, pressure, temperature, mixing_ratio)
        p2 = lcl_solver(p1, pressure, temperature, mixing_ratio)

        if (delta := p2 - 2.0 * p1 + p0):
            p2 = p0 - (p1 - p0) ** 2 / delta    # delta squared

        err = abs((p2 - p0) / p0) if p0 else p2 # absolute relative error

        if err < eps:
            return p2

        p0 = p2

    return NaN


cdef floating[:, :] _lcl(
    floating[:] pressure, floating[:] temperature, floating[:] dewpoint, size_t max_iters, floating eps
) noexcept:
    cdef size_t N, i
    cdef floating P, lcl_p, r
    cdef floating[:, :] out

    N = pressure.shape[0]

    out = nzarray((2, N), pressure.itemsize)
    with nogil, parallel():
        for i in prange(N, schedule='dynamic'):
            P = pressure[i]
            r = mixing_ratio(saturation_vapor_pressure(dewpoint[i]), P)
            out[0, i] = lcl_p = lcl_integrator(P, temperature[i], r, max_iters, eps=eps)
            out[1, i] = _dewpoint(vapor_pressure(lcl_p, r))

    return out


def lcl(
    np.ndarray pressure,
    np.ndarray temperature,
    np.ndarray dewpoint,
    *,
    size_t max_iters = 50,
    floating eps = 0.1,
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
        x[...] = _lcl[float](
            pressure.astype(np.float32), 
            temperature.astype(np.float32),
            dewpoint.astype(np.float32),
            max_iters,
            eps,
        )
    else:
        x[...] = _lcl[double](
            pressure.astype(np.float64),
            temperature.astype(np.float64),
            dewpoint.astype(np.float64),
            max_iters,
            eps,
        )

    return x[0], x[1]


# -------------------------------------------------------------------------------------------------
# wet_bulb_temperature
# -------------------------------------------------------------------------------------------------
@cython.ufunc
cdef floating wet_bulb_temperature(
    floating pressure, floating temperature, floating dewpoint
) noexcept nogil:
    cdef floating r, lcl_p, lcl_t
    r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure)
    lcl_p = lcl_integrator(pressure, temperature, r, 50, eps=0.1)
    lcl_t = _dewpoint(vapor_pressure(lcl_p, r))

    return moist_lapse_integrator(lcl_p, pressure, lcl_t, 1000.0)

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


