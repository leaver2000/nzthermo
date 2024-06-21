#include <wind.hpp>

namespace libthermo {

template <floating T>
constexpr T wind_direction(const T u, const T v) noexcept {
    return fmod(degrees(atan2(u, v)) + 180.0, 360.0);
}

template <floating T>
constexpr T wind_direction(const wind_components<T>& uv) noexcept {
    return wind_direction(uv.u, uv.v);
}

template <floating T>
constexpr T wind_magnitude(const T u, const T v) noexcept {
    return hypot(u, v);
}

template <floating T>
constexpr T wind_magnitude(const wind_components<T>& uv) noexcept {
    return hypot(uv.u, uv.v);
}

template <floating T>
constexpr wind_vector<T>::wind_vector(const wind_components<T>& uv) noexcept :
    direction(wind_direction(uv)), magnitude(wind_magnitude(uv)) {
}

template <floating T>
constexpr wind_components<T>::wind_components(const wind_vector<T>& dm) noexcept {
    const T d = radians(dm.direction);
    const T m = -dm.magnitude;
    u = m * sin(d);
    v = m * cos(d);
}

}  // namespace libthermo