# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

# pyright: reportGeneralTypeIssues=false

cimport nzthermo._C as C

cimport cython
cimport numpy as np

ctypedef fused T:
    float
    double

np.import_array()
np.import_ufunc()


@cython.ufunc # theta
cdef T potential_temperature(T pressure, T temperature) noexcept nogil:
    return C.potential_temperature(pressure, temperature)


@cython.ufunc # theta_e
cdef T equivalent_potential_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    return C.equivalent_potential_temperature(pressure, temperature, dewpoint)


@cython.ufunc # theta_w
cdef T wet_bulb_potential_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    return C.wet_bulb_potential_temperature(pressure, temperature, dewpoint)


@cython.ufunc 
cdef T wet_bulb_temperature(T pressure, T temperature, T dewpoint) noexcept nogil:
    cdef: # the ufunc signature doesn't support keyword arguments
        size_t max_iter = 50
        T eps  = 0.1
        T step = 1000.0

    return C.wet_bulb_temperature(pressure, temperature, dewpoint, eps, step, max_iter)


@cython.ufunc 
cdef T lcl_pressure(T pressure, T temperature, T dewpoint) noexcept nogil:
    cdef:
        size_t max_iter = 50
        T eps = 0.1

    return C.lcl_pressure(pressure, temperature, dewpoint, eps, max_iter)