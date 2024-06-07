#include <wind.hpp>

namespace libthermo {

/* ........................................{ common }........................................... */
template <floating T>
constexpr T wind_direction(const T u, const T v, const bool from) noexcept {
    T wdir = degrees(atan2(u, v));
    if (from)
        wdir = fmod(wdir - 180.0, 360.0);

    if (wdir <= 0)
        wdir = 360.0;
    if (u == 0 && v == 0)
        wdir = 0.0;

    return wdir;
}
template <floating T>
constexpr T wind_magnitude(const T u, const T v) noexcept {
    return hypot(u, v);
}

template <floating T>
constexpr WindComponents<T> wind_components(const T direction, const T magnitude) noexcept {
    if (direction < 0 || direction > 360)
        return {NAN, NAN};

    const T u = magnitude * sin(radians(direction));
    const T v = magnitude * cos(radians(direction));

    return {-u, -v};
}
}  // namespace libthermo