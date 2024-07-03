# pyright: reportGeneralTypeIssues=false


cdef extern from "libthermo.cpp" namespace "libthermo" nogil:
    const double g       # `(m/s^2)` - acceleration due to gravity
    const double T0      # `(K)` -  
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

    T linear_interpolate[T](T x, T x0, T x1, T y0, T  y1) noexcept
    size_t index_pressure[T](T* levels, T value, size_t size) noexcept


    cdef cppclass wind_components[T]:
        T u, v
        wind_components() noexcept
        wind_components(T, T) noexcept
        wind_components(wind_vector[T]) noexcept

    cdef cppclass wind_vector[T]:
        T direction, magnitude
        wind_vector() noexcept
        wind_vector(T, T) noexcept
        wind_vector(wind_components[T]) noexcept

    cdef cppclass lcl[T]:
        T pressure
        T temperature
        lcl() noexcept
        lcl(T pressure, T temperature) noexcept
        lcl(T pressure, T temperature, T dewpoint) noexcept
        size_t index_on(T* levels, size_t size) noexcept

    T wind_direction[T](T u, T v) noexcept
    T wind_magnitude[T](T u, T v) noexcept
    T standard_pressure[T](T height) noexcept 
    T standard_height[T](T pressure) noexcept 
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
    T dry_static_energy[T](T height, T temperature) noexcept
    T moist_static_energy[T](T height, T temperature, T specific_humidity) noexcept


    