# pyright: reportGeneralTypeIssues=false

from libcpp.vector cimport vector

ctypedef fused Float:
    float
    double



cdef extern from "<utility>" namespace "std" nogil:
    cdef cppclass pair[T, U]:
        # ctypedef T first_type
        # ctypedef U second_type
        T pressure    "first"
        U temperature "second"



cdef extern from "_C.cpp" namespace "nzt" nogil:
    # ........................................................................................... #
    #  - constants
    # ........................................................................................... #
    # TODO:...
    # ........................................................................................... #
    #  - structure
    # ........................................................................................... #
    cdef cppclass WindComponents[T]:
        T u
        T v

    cdef cppclass LCL[T]:
        T pressure
        T temperature

    cdef cppclass Parcel[T]:
        T pressure
        T temperature
        T dewpoint

    # ........................................................................................... #
    #  - common functions
    # ........................................................................................... #
    T linear_interpolate[T](T x, T x0, T x1, T y0, T y1) noexcept
    T degrees[T](T radians) noexcept
    T radians[T](T degrees) noexcept
    T interpolate_z[T](size_t size, T x, T* xp, T* fp) noexcept
    # ........................................................................................... #
    #  - wind
    # ........................................................................................... #
    T wind_direction[T](T u, T v) noexcept
    T wind_direction[T](T u, T v, T from_) noexcept
    T wind_magnitude[T](T u, T v) noexcept
    WindComponents[T] wind_components[T](T direction, T magnitude) noexcept

    # ........................................................................................... #
    #  - thermodynamic functions
    # ........................................................................................... #
    T mixing_ratio[T](T pressure, T vapor_pressure) noexcept
    T saturation_vapor_pressure[T](T temperature) noexcept
    T mixing_ratio_from_dewpoint[T](T pressure, T dewpoint) noexcept
    T saturation_mixing_ratio[T](T pressure, T temperature) noexcept
    T vapor_pressure[T](T temperature, T mixing_ratio) noexcept
    T virtual_temperature[T](T temperature, T mixing_ratio) noexcept
    T dry_lapse[T](T pressure, T reference_pressure, T temperature) noexcept
    # @overload
    T dewpoint[T](T vapor_pressure) noexcept
    # @overload
    T dewpoint_from_mixing_ratio "nzt::dewpoint" [T](T pressure, T mixing_ratio) noexcept
    # ........................................................................................... #
    #  - adiabatic processes
    # ........................................................................................... #
    LCL[T] lcl[T](T pressure, T temperature, T dewpoint, T eps, size_t max_iters) noexcept
    T lcl_pressure[T](T pressure, T temperature, T dewpoint, T eps, size_t max_iters) noexcept
    T moist_lapse[T](T pressure, T next_pressure, T temperature, T step) noexcept
    T potential_temperature[T](T pressure, T temperature) noexcept # theta
    T equivalent_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_e
    T wet_bulb_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_w
    T wet_bulb_temperature[T](
        T pressure, T temperature, T dewpoint, T eps, T step, size_t max_iters
    ) noexcept
    
    # sharp routine's
    T wobus[T](T temperature) noexcept