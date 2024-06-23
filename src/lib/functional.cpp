#include <omp.h>
#include <functional.hpp>
#include <functional>

namespace libthermo {

/**
 * @brief The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan
 * inputs.
 * 
 * @tparam T 
 * @param x 
 * @return constexpr T 
 */
template <floating T>
constexpr T sign(const T x) noexcept {
    if (isnan(x))
        return NAN;
    else if (x == 0.)
        return 0.;
    else if (x < 0)
        return -1.;
    return 1.;
}

template <floating T>
constexpr bool monotonic(const T x[], const size_t size, direction direction) noexcept {
    if (direction == direction::increasing) {
        for (size_t i = 1; i < size; i++)
            if (x[i] < x[i - 1])
                return false;
    } else {
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
    return (x - x0) / LIMIT_ZERO(x1 - x0);
}

template <floating T>
constexpr T linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1
) noexcept {
    return y0 + (x - x0) * (y1 - y0) / LIMIT_ZERO(x1 - x0);
}

/**
 * @author Kelton Halbert - NWS Storm Prediction Center/OU-CIWRO
 *
 * @brief Finds the index of the lower bound that does not satisfy element value
 *
 * Based on std::lower_bound, this iterates over an array using a binary search
 * to find the first element that does not satisfy the comparison condition. By
 * default, the comparison is std::less. Binary search expects data to be sorted
 * in ascending order -- for pressure level data, change the comparator.
 *
 * We use a custom implementation of sharp::lower_bound rather than
 * std::lower_bound for a few reasons. First, we prefer raw pointer
 * arrays over vectors for easy integration with SWIG bindings,
 * C code, and the potential future in which this runs on CUDA
 * architecture. Currently, the algorithms in the STL library are
 * not supported by the CUDA STL, but the types in <functional>
 * (i.e. std::less) are supported by the CUDA STL. Additionally, this
 * implementation of lower_bound is designed to reduce branching.
 *
 * @param   array   The array to search over
 * @param   N       The length of the array
 * @param   value   The value for lower-bound comparison
 * @param   cmp     The comparator
 *
 * @return  Index of lower bound
 */
template <floating T, typename C>
constexpr size_t lower_bound(const T array[], const int N, const T& value, const C cmp) {
    int len = N;
    size_t idx = 0;
    while (len > 1) {
        int half = len / 2;
        idx += cmp(array[idx + half - 1], value) * half;
        len -= half;
    }
    return idx;
}

template <floating T, typename C>
constexpr size_t upper_bound(const T array[], const int N, const T& value, const C cmp) {
    int len = N;
    size_t idx = 0;
    while (len > 1) {
        int half = len / 2;
        idx += !cmp(value, array[idx + half - 1]) * half;
        len -= half;
    }
    return idx;
}

template <floating T>
size_t search_sorted(const T x[], const T value, const size_t size, const bool inverted) noexcept {
    if (inverted)
        return lower_bound(x, size, value, std::greater_equal());

    return upper_bound(x, size, value, std::less_equal());
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
constexpr T heaviside(const T x, const T h0) noexcept {
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
constexpr T rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step /* 1000.0 (Pa) */) noexcept {
    T k1, delta, abs_delta;
    size_t N = 1;

    abs_delta = std::fabs(delta = (x1 - x0));
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
constexpr T fixed_point(
  const Fn<T, T, T, Args...> fn,
  const size_t max_iters,
  const T eps,
  const T x0,
  const Args... args
) noexcept {
    T p0, p1, p2, delta;

    p0 = x0;
    for (size_t i = 0; i < max_iters; i++) {
        p1 = fn(p0, x0, args...);
        p2 = fn(p1, x0, args...);
        delta = p2 - 2.0 * p1 + p0;

        p2 = delta ? p0 - std::pow(p1 - p0, 2) / delta : p2; /* delta squared */

        if ((p0 ? std::fabs((p2 - p0) / p0) : p2 /* absolute relative error */) < eps)
            return p2;

        p0 = p2;
    }

    return NAN;
}

/**
 * @brief Interpolates a 1D function using a linear interpolation.
 * 
 * @tparam T 
 * @param x 
 * @param xp 
 * @param fp 
 * @param size 
 * @return constexpr T 
 */
template <floating T>
constexpr T interpolate_1d(const T x, const T xp[], const T fp[], const size_t size) noexcept {
    const size_t i = lower_bound(xp, size, x, std::greater_equal());
    if (i == 0)
        return fp[0];

    return linear_interpolate(x, xp[i - 1], xp[i], fp[i - 1], fp[i]);
}

/**
 * @author Jason Leaver - USAF 557WW/1WXG
 * 
 * @brief Intersects two 1D functions.
 * 
 * @tparam T (floating point) 
 * @param x 
 * @param a 
 * @param b 
 * @param size 
 * @param log_x 
 * @param increasing 
 * @param bottom 
 * @return constexpr point<T>
 */
template <floating T>
constexpr point<T> intersect_1d(
  const T x[],
  const T a[],
  const T b[],
  const size_t size,
  const bool log_x,
  const bool increasing,
  const bool bottom
) noexcept {
    const T sign_check = increasing ? 1. : -1.;
    T x0, x1, a0, a1, b0, b1, delta_y0, delta_y1, x_intercept, y_intercept;
    size_t i0, i1;
    if (bottom) {
        for (i0 = 0, i1 = 1; i0 < size - 1; i0++, i1++) {
            T s0 = sign(a[i0] - b[i0]);
            T s1 = sign(a[i1] - b[i1]);

            if ((s0 != s1) && (s1 == sign_check))
                break;
        }
        if (i1 == size)
            return {NAN, NAN};
    } else {
        for (i0 = size - 1, i1 = size; i0 > 0; i0--, i1--) {
            T s0 = sign(a[i0] - b[i0]);
            T s1 = sign(a[i1] - b[i1]);

            if ((s0 != s1) && (s1 == sign_check))
                break;
        }
        if (i0 == 0)
            return {NAN, NAN};
    }

    x0 = x[i0];
    x1 = x[i1];
    if (log_x) {
        x0 = std::log(x0);
        x1 = std::log(x1);
    }

    a0 = a[i0];
    a1 = a[i1];
    b0 = b[i0];
    b1 = b[i1];

    delta_y0 = a0 - b0;
    delta_y1 = a1 - b1;

    x_intercept = (delta_y1 * x0 - delta_y0 * x1) / LIMIT_ZERO(delta_y1 - delta_y0);
    y_intercept = ((x_intercept - x0) / LIMIT_ZERO(x1 - x0)) * (a1 - a0) + a0;

    if (log_x)
        x_intercept = std::exp(x_intercept);

    return {x_intercept, y_intercept};
};

}  // namespace libthermo
