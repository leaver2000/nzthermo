# pyright: reportGeneralTypeIssues=false

cdef extern from "<utility>" namespace "std" nogil:
    cdef cppclass pair[T, U]:
        # ctypedef T first_type
        # ctypedef U second_type
        T pressure    "first"
        U temperature "second"
        # pair() except +
        # pair(pair&) except +
        # pair(T&, U&) except +
        # bint operator==(pair&, pair&)
        # bint operator!=(pair&, pair&)
        # bint operator<(pair&, pair&)
        # bint operator>(pair&, pair&)
        # bint operator<=(pair&, pair&)
        # bint operator>=(pair&, pair&)

cdef extern from "_C.cpp" namespace "nzt" nogil:
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
    pair[T, T] lcl[T](T pressure, T temperature, T dewpoint, T eps, size_t max_iters) noexcept
    T lcl_pressure[T](T pressure, T temperature, T dewpoint, T eps, size_t max_iters) noexcept
    T moist_lapse[T](T pressure, T next_pressure, T temperature, T step) noexcept
    T potential_temperature[T](T pressure, T temperature) noexcept # theta
    T equivalent_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_e
    T wet_bulb_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_w
    T wet_bulb_temperature[T](
        T pressure, T temperature, T dewpoint, T eps, T step, size_t max_iters
    ) noexcept