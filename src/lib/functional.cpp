#include "functional.hpp"
#include <omp.h>
namespace libthermo {

template <floating T>
constexpr bool monotonic(const T x[], const size_t size, direction direction) noexcept {
    if (direction == direction::increasing) {
#pragma omp simd
        for (size_t i = 1; i < size; i++)
            if (x[i] < x[i - 1])
                return false;
    } else {
#pragma omp simd
        for (size_t i = 1; i < size; i++)
            if (x[i] > x[i - 1])
                return false;
    }

    return true;
}

template <floating T>
constexpr T degrees(const T radians) noexcept {
    return radians * 180.0 / M_PI;
}

template <floating T>
constexpr T radians(const T degrees) noexcept {
    return degrees * M_PI / 180.0;
}

template <floating T>
constexpr T norm(const T x, const T x0, const T x1) noexcept {
    return (x - x0) / (x1 - x0);
}

template <floating T>
constexpr T linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1
) noexcept {
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

template <floating T>
size_t search_sorted(
  const T x[], const T value, const size_t size, const bool inverted = false
) noexcept {
    size_t i = 0;
    if (inverted) {
        while (i < size && x[i] >= value)
            i++;
    } else {
        while (i < size && x[i] <= value)
            i++;
    }

    return i;
}

template <floating T>
constexpr T interpolate_z(const size_t size, const T x, const T xp[], const T fp[]) noexcept {
    size_t i = search_sorted(xp, x, size, true);
    if (i == 0)
        return fp[0];

    return linear_interpolate(x, xp[i - 1], xp[i], fp[i - 1], fp[i]);
}

/*see: https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_internal.h.src#L426*/
template <floating T>
constexpr T heaviside(const T x, const T h0) noexcept {
    if (isnan(x))
        return NaN;
    else if (x == 0)
        return h0;
    else if (x < 0)
        return 0.0;

    return 1.0;
}

}  // namespace libthermo