# pyright: reportGeneralTypeIssues=false

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


cdef enum BroadcastMode:
    BROADCAST = 1
    MATRIX = 2
    ELEMENT_WISE = 3

cdef enum ProfileStrategy:
    SURFACE_BASED = 0
    MOST_UNSTABLE = 1
    MIXED_LAYER = 2
    EQUILIBRIUM = 3
    VIRTUAL = 4

# cdef inline T degrees(T rad) noexcept nogil:
#     return rad * 180 / pi


# cdef inline T radians(T deg) noexcept nogil:
#     return deg * pi / 180


# cdef extern from "<utility>" namespace "std" nogil:
#     cdef cppclass pair[T, U]:
#         # ctypedef T first_type
#         # ctypedef U second_type
#         T pressure    "first"
#         U temperature "second"
#         # pair() except +
#         # pair(pair&) except +
#         # pair(T&, U&) except +
#         # bint operator==(pair&, pair&)
#         # bint operator!=(pair&, pair&)
#         # bint operator<(pair&, pair&)
#         # bint operator>(pair&, pair&)
#         # bint operator<=(pair&, pair&)
#         # bint operator>=(pair&, pair&)



# cdef extern from "<cmath>" namespace "std" nogil:
#     T lerp[T](T a, T b, T t) noexcept


# cdef extern from "_core.cpp" namespace "nzt" nogil:
#     T mixing_ratio[T](const T pressure, const T vapor_pressure) noexcept
#     T saturation_vapor_pressure[T](const T temperature) noexcept
#     T mixing_ratio_from_dewpoint[T](const T pressure, const T dewpoint) noexcept
#     T saturation_mixing_ratio[T](const T pressure, const T temperature) noexcept
#     T vapor_pressure[T](const T temperature, const T mixing_ratio) noexcept
#     T virtual_temperature[T](const T temperature, const T mixing_ratio) noexcept
    
#     # cdef cppclass parcel[T]:
#     #     T pressure
#     #     T temperature
#     #     parcel() except +
#     #     parcel(parcel&) except +
#     #     parcel(T&, T&) except +
        
#     # @overload
#     T dewpoint[T](const T vapor_pressure) noexcept
#     # @overload
#     T dewpoint_from_mixing_ratio "nzt::dewpoint" [T](
#         const T pressure, const T mixing_ratio
#     ) noexcept
    
#     T c_moist_lapse "nzt::moist_lapse" [T] (
#         const T pressure, const T next_pressure, const T temperature, const  T step
#     ) noexcept

#     pair[T, T] c_lcl "nzt::lcl" [T](
#         const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
#     ) noexcept
#     T c_lcl_pressure "nzt::lcl_pressure" [T](
#         const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
#     ) noexcept

#     T c_moist_lapse "nzt::moist_lapse" [T](
#         const T pressure, const T next_pressure, const T temperature, const T step
#     ) noexcept
#     # .............................................................................................
#     # ufunc definitions
#     # .............................................................................................
#     # theta
#     T potential_temperature[T](const T pressure, const T temperature) noexcept
#     # theta_e
#     T equivalent_potential_temperature[T](const T pressure, const T temperature, const T dewpoint) noexcept 
#     # theta_w
#     T wet_bulb_potential_temperature[T](const T pressure, const T temperature, const T dewpoint) noexcept
#     T wet_bulb_temperature[T](
#         const T pressure,
#         const T temperature,
#         const T dewpoint,
#         const T eps,
#         const T step,
#         const size_t max_iters
#     ) noexcept