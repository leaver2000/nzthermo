"""
This module wraps many of the c++ templated functions as `numpy.ufuncs`, so that they can be used
in a vectorized manner.

TDOD: rather than maintaing the external stub file a better use case would to set up a simple script
that generates the stub file from the c++ header file.
"""

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false
from typing import TypeVar

import numpy as np

cimport cython
cimport numpy as np

cimport nzthermo._C as C
from nzthermo._C cimport Md, Mw


cdef extern from "<cmath>" namespace "std" nogil:
    T fabs[T](T) noexcept
    bint isnan(double) noexcept


np.import_array()
np.import_ufunc()

_S = TypeVar("_S")
_T = TypeVar("_T")


ctypedef fused T:
    float
    double

ctypedef fused integer:
    short
    long


@cython.ufunc
cdef bint less_or_close(T x, T y) noexcept nogil:
    return (
        not isnan(x) and not isnan(y)
        and (x < y or fabs(x - y) <= (1.0e-05 * fabs(y)))
    )

@cython.ufunc
cdef bint greater_or_close(T x, T y) noexcept nogil:
    return (
        not isnan(x) and not isnan(y)
        and (x > y or fabs(x - y) <= (1.0e-05 * fabs(y)))
    )

@cython.ufunc
cdef bint between_or_close(T x, T y0, T y1) noexcept nogil:
    return (
        not isnan(x) and not isnan(y0) and not isnan(y1)
        and (x > y0 or fabs(x - y0) <= (1.0e-05 * fabs(y0)))
        and (x < y1 or fabs(x - y1) <= (1.0e-05 * fabs(y1)))
    )

class pressure_vector(np.ndarray[_S, np.dtype[_T]]):
    def __new__(cls, pressure):
        if isinstance(pressure, pressure_vector):
            return pressure
        elif isinstance(pressure, np.ndarray):
            return pressure.view(cls)

        return np.asarray(pressure).view(cls)

    def is_above(self, bottom, close=True):
        bottom = np.asarray(bottom)
        if not close:
            return np.asarray(self > bottom, np.bool_)

        return np.asarray(less_or_close(self, bottom), np.bool_)

    def is_below(self, top, close=True):
        top = np.asarray(top)
        if not close:
            return np.asarray(self < top, np.bool_)

        return np.asarray(greater_or_close(self, top), np.bool_)

    def is_between(self, bottom, top, close=True):
        bottom, top = np.asarray(bottom), np.asarray(top)
        if not close:
            return np.asarray((self > bottom) & (self < top), np.bool_)

        return np.asarray(between_or_close(self, top, bottom), np.bool_)

    def where(self, condition, fill=np.nan):
        return np.where(condition, self, fill).view(pressure_vector)


# ............................................................................................... #
#  - wind
# ............................................................................................... #
@cython.ufunc
cdef T wind_direction(T u, T v) noexcept nogil:
    return C.wind_direction(u, v)


@cython.ufunc
cdef T wind_magnitude(T u, T v) noexcept nogil:
    return C.wind_magnitude(u, v)

# TODO: the signature....
# cdef (T, T) wind_components(T direction, T magnitude) noexcept nogil:...
# 
# Is unsupported by the ufunc signature. So the option are:
# - maintain gil
# - cast to double 
# - write the template in C
@cython.ufunc
cdef (double, double) wind_components(T d, T m) noexcept nogil: 
    cdef C.wind_components[T] uv = C.wind_components[T](C.wind_vector[T](d, m))
    return <double>uv.u, <double>uv.v


@cython.ufunc
cdef (double, double) wind_vector(T u, T v) noexcept nogil:
    cdef C.wind_vector[T] dm = C.wind_vector[T](C.wind_components[T](u, v))
    return <double>dm.direction, <double>dm.magnitude

# 1x1
@cython.ufunc
cdef T exner_function(T pressure) noexcept nogil:
    return C.exner_function(pressure)


@cython.ufunc
cdef T dewpoint(T vapor_pressure) noexcept nogil:
    return C.dewpoint(vapor_pressure)


@cython.ufunc
cdef T saturation_vapor_pressure(T temperature) noexcept nogil:
    return C.saturation_vapor_pressure(temperature)


@cython.ufunc
cdef T wobus(T temperature) noexcept nogil:
    r"""
    author John Hart - NSSFC KCMO / NWSSPC OUN

    Computes the difference between the wet-bulb potential temperatures for saturated and dry air 
    given the temperature.

    brief 
    
    The Wobus Function (wobf) is defined as the difference between
    the wet-bulb potential temperature for saturated air (WBPTS)
    and the wet-bulb potential temperature for dry air (WBPTD) given
    the same temperature in Celsius.

    WOBF(T) := WBPTS - WBPTD

    Although WBPTS and WBPTD are functions of both pressure and
    temperature, it is assumed their difference is a function of
    temperature only. The difference is also proportional to the
    heat imparted to a parcel.

    This function uses a polynomial approximation to the wobus function,
    fitted to values in Table 78 of PP.319-322 of the Smithsonian Meteorological
    Tables by Roland List (6th Revised Edition). Herman Wobus, a mathematician
    for the Navy Weather Research Facility in Norfolk, VA computed these
    coefficients a very long time ago, as he was retired as of the time of
    the documentation found on this routine written in 1981.

    It was shown by Robert Davies-Jones (2007) that the Wobus function has
    a slight dependence on pressure, which results in errors of up to 1.2
    degrees Kelvin in the temperature of a lifted parcel.
    
    Parameters
    ----------
    x : (K) array_like
        Input array.
    
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    """
    return C.wobus(temperature)

# 2x1
@cython.ufunc
cdef T mixing_ratio(T partial_pressure, T total_pressure) noexcept nogil:
    return Mw / Md * partial_pressure / (total_pressure - partial_pressure)


@cython.ufunc # theta
cdef T potential_temperature(T pressure, T temperature) noexcept nogil:
    return C.potential_temperature(pressure, temperature)


@cython.ufunc
cdef T dewpoint_from_specific_humidity(T pressure, T specific_humidity) noexcept nogil:
    cdef T Q = specific_humidity or 1e-9
    cdef T r = Q / (1 - Q)
    return C.dewpoint(pressure * r / (Mw / Md  + r))


@cython.ufunc
cdef T saturation_mixing_ratio(T pressure, T temperature) noexcept nogil:
    return C.saturation_mixing_ratio(pressure, temperature)


@cython.ufunc
cdef T vapor_pressure(T pressure, T mixing_ratio) noexcept nogil:
    return C.vapor_pressure(pressure, mixing_ratio)


@cython.ufunc
cdef T virtual_temperature(T temperature, T mixing_ratio) noexcept nogil:
    return C.virtual_temperature(temperature, mixing_ratio)


# 3x1
@cython.ufunc
cdef T dry_lapse(T pressure, T temperature, T reference_pressure) noexcept nogil:
    return C.dry_lapse(pressure, reference_pressure, temperature)


@cython.ufunc # theta_e
cdef T equivalent_potential_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    r"""Calculates the equivalent potential temperature.

    Parameters
    ----------
    x : array_like
        pressure (Pa) values.
    x1 : array_like
        temperature (K) values.
    x2 : array_like
        dewpoint (K) values.
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    theta_e : ndarray
        Equivalent potential temperature (K).

    Examples
    -------
    >>> import numpy as np
    >>> import nzthermo as nzt
    >>> data = np.load("tests/data.npz", allow_pickle=False)
    >>> pressure = data['P']
    >>> temperature = data['T']
    >>> dewpoint = data['Td']
    >>> assert pressure.ndim == 1 and pressure.shape != temperature.shape
    >>> mask = (pressure <= 70000.0) & (pressure >= 50000.0)
    >>> theta_e = nzt.equivalent_potential_temperature(
    ...     pressure,
    ...     temperature,
    ...     dewpoint,
    ...     where=mask, # masking values with inf will alow us to call argmin without worrying about nan
    ...     out=np.full_like(temperature, np.inf),
    ... )
    >>> theta_e.shape
    (540, 40)
    >>> theta_e.argmin(axis=1)
    array([13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 15, 21, 21, 21, 21, 21, 21,
           20, 13, 18, 14, 14, 14, 14, 20, 20, 14, 14, 16, 18, 13, 13, 13, 13,
           13, 13, 13, 13, 13, 13, 21, 21, 17, 15, 21, 18, 21, 13, 13, 13, 18,
           13, 14, 13, 16, 13, 19, 18, 18, 20, 13, 13, 15, 14, 13, 13, 13, 13,
           13, 14, 21, 18, 21, 21, 13, 21, 20, 21, 14, 13, 19, 20, 13, 16, 13,
           18, 16, 18, 21, 20, 13, 13, 14, 16, 16, 14, 13, 13, 13, 19, 21, 21,
           21, 21, 20, 17, 20, 21, 21, 13, 13, 20, 13, 14, 18, 13, 13, 13, 13,
           14, 13, 13, 13, 13, 13, 13, 13, 13, 14, 15, 19, 18, 18, 20, 19, 19,
           13, 20, 21, 13, 14, 20, 18, 18, 16, 13, 13, 13, 16, 13, 13, 13, 13,
           13, 13, 14, 13, 13, 15, 15, 15, 15, 13, 13, 16, 16, 20, 18, 15, 21,
           21, 13, 16, 16, 14, 13, 13, 13, 13, 13, 13, 13, 14, 13, 14, 15, 13,
           13, 13, 14, 21, 21, 21, 16, 14, 15, 13, 17, 18, 13, 20, 18, 18, 20,
           14, 18, 14, 13, 13, 13, 19, 18, 14, 14, 13, 15, 15, 18, 21, 20, 19,
           21, 20, 21, 21, 14, 14, 18, 20, 15, 18, 13, 16, 14, 16, 14, 16, 18,
           13, 13, 20, 13, 18, 18, 18, 16, 17, 19, 19, 18, 20, 21, 20, 18, 21,
           17, 17, 19, 18, 16, 18, 13, 13, 14, 13, 16, 16, 16, 16, 18, 16, 14,
           14, 16, 18, 18, 19, 18, 17, 18, 20, 21, 21, 20, 20, 21, 15, 19, 17,
           18, 18, 13, 15, 16, 13, 13, 16, 15, 13, 13, 14, 13, 13, 18, 18, 16,
           19, 19, 16, 16, 19, 19, 18, 20, 19, 21, 20, 18, 20, 18, 18, 13, 15,
           15, 17, 18, 16, 13, 13, 13, 13, 14, 13, 13, 16, 16, 18, 18, 16, 16,
           17, 18, 20, 19, 16, 19, 13, 14, 14, 18, 17, 16, 15, 18, 18, 13, 13,
           13, 14, 13, 13, 13, 14, 13, 16, 16, 19, 17, 14, 14, 15, 16, 17, 18,
           15, 13, 14, 13, 15, 13, 13, 18, 13, 13, 14, 14, 15, 14, 13, 13, 13,
           13, 13, 13, 14, 16, 19, 15, 18, 15, 13, 15, 15, 16, 16, 13, 13, 19,
           17, 13, 13, 13, 13, 13, 13, 15, 19, 13, 13, 13, 13, 13, 13, 13, 13,
           13, 15, 16, 16, 13, 13, 18, 16, 16, 15, 14, 13, 14, 13, 15, 17, 16,
           13, 13, 13, 16, 15, 18, 13, 13, 13, 13, 13, 13, 13, 13, 14, 16, 17,
           13, 17, 17, 16, 16, 14, 13, 13, 15, 16, 16, 15, 15, 17, 18, 13, 15,
           15, 14, 14, 13, 13, 13, 13, 13, 13, 15, 14, 15, 16, 14, 17, 17, 16,
           16, 13, 13, 14, 20, 17, 17, 14, 16, 16, 13, 13, 17, 16, 15, 14, 13,
           13, 13, 13, 13, 13, 13, 14, 20, 18, 18, 15, 17, 13, 14, 13, 13, 13,
           14, 13, 13, 13, 15, 20, 18, 13, 14, 19, 13, 16, 13])
    """
    return C.equivalent_potential_temperature(pressure, temperature, dewpoint)


@cython.ufunc # theta_w
cdef T wet_bulb_potential_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    return C.wet_bulb_potential_temperature(pressure, temperature, dewpoint)


@cython.ufunc 
cdef T wet_bulb_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    return C.wet_bulb_temperature(pressure, temperature, dewpoint)


@cython.ufunc 
cdef T lcl_pressure(T pressure, T temperature, T dewpoint) noexcept nogil:
    return C.lcl_pressure(pressure, temperature, dewpoint)
    


# 3x2
@cython.ufunc
cdef (double, double) lcl(T pressure, T temperature, T dewpoint) noexcept nogil:
    # TODO: the signature....
    # cdef (T, T) lcl(T pressure, T temperature, T dewpoint) noexcept nogil:...
    # 
    # Is unsupported by the ufunc signature. So the option are:
    # - maintain gil
    # - cast to double 
    # - write the template in C
    cdef C.lcl[T] lcl = C.lcl[T](pressure, temperature, dewpoint)

    return <double>lcl.pressure, <double>lcl.temperature

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

    Parameters
    ----------
    x : (year) array_like
        Input array.
    x1 : (month) array_like
        Input array.

    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
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



