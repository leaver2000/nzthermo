# cython: boundscheck=False
# pyright: reportGeneralTypeIssues=false

ctypedef fused Float:
    float
    double


cdef extern from "<utility>" namespace "std" nogil:
    cdef cppclass pair[T, U]:
        T pressure    "first"
        U temperature "second"


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


cdef inline size_t search_pressure(Float[:] pressure, Float value) noexcept nogil:
    cdef size_t Z = pressure.shape[0]
    if pressure[Z - 1] > value:
        return Z
    return search_sorted(&pressure[0], value, Z, True)


cdef extern from "wind.cpp" namespace "libthermo" nogil:
    cdef cppclass WindComponents[T]:
        T u
        T v

    T wind_direction[T](T u, T v) noexcept
    T wind_direction[T](T u, T v, T from_) noexcept
    T wind_magnitude[T](T u, T v) noexcept
    WindComponents[T] wind_components[T](T direction, T magnitude) noexcept


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

    # 1x1
    T saturation_vapor_pressure[T](T temperature) noexcept
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
