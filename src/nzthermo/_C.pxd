# cython: boundscheck=False
# pyright: reportGeneralTypeIssues=false


cdef extern from "functional.cpp" namespace "libthermo" nogil:
    cdef cppclass point[T]:
        T x
        T y

    T linear_interpolate[T](T x, T x0, T x1, T y0, T y1) noexcept
    T degrees[T](T radians) noexcept
    T radians[T](T degrees) noexcept
    T interpolate_1d[T](T x, T* xp, T* fp, size_t size) noexcept
    point[T] intersect_1d[T](
        T* x, T* a, T* b, size_t size, bint log_x, bint increasing, bint bottom
    ) noexcept

    size_t search_sorted[T](T* x, T value, size_t size, bint inverted) noexcept


cdef extern from "wind.cpp" namespace "libthermo" nogil:
    T wind_direction[T](T, T) noexcept
    T wind_magnitude[T](T, T) noexcept

    cdef cppclass wind_components[T]:
        T u, v
        wind_components() noexcept
        wind_components(T, T) noexcept
        wind_components(wind_vector[T]) noexcept

    cdef cppclass wind_vector[T]:
        T direction, magnitude
        wind_vector() noexcept
        wind_vector(T, T) noexcept
        wind_vector(wind_components[T])  noexcept


cdef extern from "libthermo.cpp" namespace "libthermo" nogil:
    const double T0      # `(J/kg*K)` - freezing point in kelvin
    const double E0      # `(Pa)` - vapor pressure at T0
    const double Cp      # `(J/kg*K)` - specific heat of dry air
    const double Rd      # `(J/kg*K)` - gas constant for dry air
    const double Rv      # `(J/kg*K)` - gas constant for water vapor
    const double Lv      # `(J/kg)` - latent heat of vaporization
    const double P0      # `(Pa)` - standard pressure at sea level
    const double Mw      # `(g/mol)` - molecular weight of water
    const double Md      # `(g/mol)` - molecular weight of dry air
    const double epsilon # `Mw / Md` - molecular weight ratio
    const double kappa   # `Rd / Cp`  - ratio of gas constants

    cdef cppclass lcl[T]:
        T pressure
        T temperature
        lcl() noexcept
        lcl(T pressure, T temperature) noexcept
        lcl(T pressure, T temperature, T dewpoint) noexcept
        size_t index_on(T* levels, size_t size) noexcept

    # 1x1
    T saturation_vapor_pressure[T](T temperature) noexcept
    T exner_function[T](T pressure) noexcept
    T exner_function[T](T pressure, T reference_pressure) noexcept  # .. overload ..
    # 2x1
    T mixing_ratio[T](T pressure, T vapor_pressure) noexcept
    T mixing_ratio_from_dewpoint[T](T pressure, T dewpoint) noexcept
    T saturation_mixing_ratio[T](T pressure, T temperature) noexcept
    T vapor_pressure[T](T temperature, T mixing_ratio) noexcept
    # 3x1
    T virtual_temperature[T](T temperature, T mixing_ratio) noexcept
    T dry_lapse[T](T pressure, T reference_pressure, T temperature) noexcept
    # @overload
    T dewpoint[T](T vapor_pressure) noexcept
    T dewpoint[T](T pressure, T mixing_ratio) noexcept # .. overload ..
    T lcl_pressure[T](T pressure, T temperature, T dewpoint) noexcept
    T moist_lapse[T](T pressure, T next_pressure, T temperature) noexcept
    T potential_temperature[T](T pressure, T temperature) noexcept # theta
    T equivalent_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_e
    T wet_bulb_potential_temperature[T](T pressure, T temperature, T dewpoint) noexcept # theta_w
    T wet_bulb_temperature[T](T pressure, T temperature, T dewpoint) noexcept
    T wobus[T](T temperature) noexcept
