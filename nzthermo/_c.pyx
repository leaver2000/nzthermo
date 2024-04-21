# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false

from cython.parallel cimport parallel, prange
from cython.view cimport array as cvarray

import numpy as np
cimport numpy as np

from . import const as _const

np.import_array()
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
    float nan       = float('nan')
    float inf       = float('inf')

del _const

# -------------------------------------------------------------------------------------------------
# thermodynamic functions
# -------------------------------------------------------------------------------------------------
cdef floating saturation_mixing_ratio(floating pressure, floating temperature) noexcept nogil:
    cdef floating P
    P = E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65)) # saturation vapor pressure
    return 0.6219 * P / (pressure - P)


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
            out[i] = nan
        return

    for i in range(Z):
        if isnan(next_pressure := pressure[i]): 
            # This can be usefull in 2D case where we might want to mask out some 
            # values below the lcl for a particular column.
            # The solver will start at the first non nan value
            out[i] = nan
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
):
    cdef size_t N, Z, i
    cdef floating[:, :] out

    N, Z = temperature.shape[0], pressure.shape[1]
    if pressure.itemsize == sizeof(float):
        out = cvarray((N, Z), itemsize=sizeof(float), format='f', mode='c')
    else:
        out = cvarray((N, Z), itemsize=sizeof(double), format='d', mode='c')

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
    pressure shape ``(N,) | (Z,) | (1, Z) | (N, Z)``

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

    # [ pressure ]
    if (ndim := pressure.ndim) == 1:
        pressure = pressure.reshape(1, -1) # (1, Z)
    elif ndim != 2:
        raise ValueError("pressure must be 1D or 2D array.")

    # [ temperature ]
    temperature = temperature.ravel()  # (N,)
    if not (N := temperature.shape[0]):
        return np.full_like(pressure, nan, dtype=dtype)

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
    cdef size_t N, i
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

    return nan


cdef floating[:, :] _lcl(
    floating[:] pressure, floating[:] temperature, floating[:] dewpoint, size_t max_iters, floating eps
):
    cdef size_t N, i
    cdef floating P, lcl_p, r
    cdef floating[:, :] out

    N = pressure.shape[0]
    if pressure.itemsize == sizeof(float):
        out = cvarray((2, N), itemsize=sizeof(float), format='f', mode='c')
    else:
        out = cvarray((2, N), itemsize=sizeof(double), format='d', mode='c')

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

    return x
