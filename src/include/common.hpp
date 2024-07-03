#pragma once
#include <cmath>
#if __cplusplus >= 202002L
#include <type_traits>
template <typename T>
concept floating = std::is_floating_point_v<T>;
#else
#define floating typename
#endif

namespace libthermo {
using std::atan2;
using std::ceil;
using std::cos;
using std::exp;
using std::fabs;
using std::fmod;
using std::hypot;
using std::isnan;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;

template <floating T>
inline constexpr T limit_zero(T x) {
    return (x) ? (x) : std::numeric_limits<T>::epsilon();
}

/**
 * @brief The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan
 * inputs.
 * 
 * @tparam T 
 * @param x 
 * @return constexpr T 
 */
template <floating T>
inline constexpr T sign(const T x) noexcept {
    if (isnan(x))
        return NAN;
    else if (x == 0.)
        return 0.;
    else if (x < 0)
        return -1.;
    return 1.;
}

template <floating T>
inline constexpr T degrees(const T radians) noexcept {
    return radians * 180.0 / M_PI;
}

template <floating T>
inline constexpr T radians(const T degrees) noexcept {
    return degrees * M_PI / 180.0;
}

template <floating T>
inline constexpr T norm(const T x, const T x0, const T x1) noexcept {
    return (x - x0) / limit_zero(x1 - x0);
}

template <floating T>
inline constexpr T linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1
) noexcept {
    return y0 + (x - x0) * (y1 - y0) / limit_zero(x1 - x0);
}

/**
 * @brief The Heaviside step function, or the unit step function, usually denoted by H or θ, is a
 * mathematical function that is zero for negative arguments and one for positive arguments.
 * 
 * @tparam T (floating point)
 * @param x 
 * @param h0 
 * @return constexpr T 
 */
template <floating T>
inline constexpr T heaviside(const T x, const T h0) noexcept {
    if (isnan(x))
        return NAN;
    else if (x == 0)
        return h0;
    else if (x < 0)
        return 0.0;

    return 1.0;
}

/**
 * @author Jason Leaver - USAF 557WW/1WXG
 * 
 * @brief Runge-Kutta 2nd order method for solving ordinary differential equations.
 * 
 * @tparam T (floating point)
 * @param fn
 * @param x0
 * @param x1
 * @param y
 * @param step (1000.0 Pa) increasing the step will increase the speed of the calculation but may
 * decrease the accuracy.
 * @return constexpr T
 */
template <floating T>
inline constexpr T rk2(T (*fn)(T, T), T x0, T x1, T y, T step /* 1000.0 (Pa) */) noexcept {
    T k1, delta, abs_delta;
    size_t N = 1;

    abs_delta = fabs(delta = (x1 - x0));
    if (abs_delta > step) {
        N = (size_t)ceil(abs_delta / step);
        delta = delta / (T)N;
    }

    for (size_t i = 0; i < N; i++) {
        k1 = delta * fn(x0, y);
        y += delta * fn(x0 + delta * 0.5, y + k1 * 0.5);
        x0 += delta;
    }

    return y;
}

/**
 * @author Jason Leaver - USAF 557WW/1WXG
 * 
 * @brief A fixed point of a function is the point at which evaluation of the
 * function returns the point.
 * @ref https://docs.scipy.org/doc/scipy/tutorial/optimize.html#fixed-point-solving
 *
 * @tparam T (floating point)
 * @tparam Args (...floating point)
 * @param fn
 * @param max_iters
 * @param eps
 * @param x0
 * @param args
 *
 * @return (T) fixed point (T) (if found, else NAN)
 */
template <floating T, floating... Args>
inline constexpr T fixed_point(
  T (*fn)(T, T, Args...), const size_t max_iters, const T eps, const T x0, const Args... args
) noexcept {
    T p0, p1, p2, delta;

    p0 = x0;
    for (size_t i = 0; i < max_iters; i++) {
        p1 = fn(p0, x0, args...);
        p2 = fn(p1, x0, args...);
        delta = p2 - 2.0 * p1 + p0;

        p2 = delta ? p0 - pow(p1 - p0, 2) / delta : p2; /* delta squared */

        if ((p0 ? fabs((p2 - p0) / p0) : p2 /* absolute relative error */) < eps)
            return p2;

        p0 = p2;
    }

    return NAN;
}

}  // namespace libthermo