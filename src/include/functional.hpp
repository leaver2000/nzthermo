#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <functional>

#include <common.hpp>
#include <functional.hpp>
namespace libthermo {

typedef enum direction {
    increasing,
    decreasing
} direction;

template <floating T>
constexpr bool monotonic(
  const T x[], const size_t size, direction direction = direction::increasing
) noexcept;

template <floating T>
constexpr T degrees(const T radians) noexcept;

template <floating T>
constexpr T radians(const T degrees) noexcept;

template <floating T>
constexpr T norm(const T x, const T x0, const T x1) noexcept;

template <floating T>
constexpr T linear_interpolate(const T x, const T x0, const T x1, const T y0, const T y1) noexcept;

template <floating T>
size_t search_sorted(
  const T x[], const T value, const size_t size, const bool inverted = false
) noexcept;

template <floating T>
constexpr T interpolate_z(const size_t size, const T x, const T xp[], const T fp[]) noexcept;

/**
 * see: https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_internal.h.src#L426
 */

template <floating T>
constexpr T heaviside(const T x, const T h0) noexcept;

/**
 * @brief Runge-Kutta 2nd order method for solving ordinary differential equations.
 */
template <floating T>
constexpr T rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step = 1000.0) noexcept;

/**
 * @brief A fixed point of a function is the point at which evaluation of the
 * function returns the point.
 * @ref * https://docs.scipy.org/doc/scipy/tutorial/optimize.html#fixed-point-solving
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
) noexcept;

/**
 * \author Kelton Halbert - NWS Storm Prediction Center/OU-CIWRO
 *
 * \brief Finds the index of the lower bound that does not <!--
 * --> satisfy element < value
 *
 * Based on std::lower_bound, this iterates over an array using a binary search
 * to find the first element that does not satisfy the comparison condition. By
 * default, the comparison is std::less. Binary search expects data to be sorted
 * in ascending order -- for pressure level data, change the comparitor.
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
 * \param   array   The array to search over
 * \param   N       The length of the array
 * \param   value   The value for lower-bound comparison
 * \param   cmp     The comparitor
 *
 * \return  Index of lower bound
 */
template <floating T, typename C = std::less<T>>
constexpr size_t lower_bound(const T array[], const int N, const T& value, const C cmp = C{});

/**
 * \author Kelton Halbert - NWS Storm Prediction Center/OU-CIWRO
 *
 * \brief Finds the first index that satisfies value < element
 *
 * Based on std::upper_bound, this iterates over an array using a binary search
 * to find the first element that satisfies the comparison condition. By
 * default, the comparison is std::less. Binary search expects data to be in
 * ascending order -- for pressure level data, change the comparitor.
 *
 * We use a custom implementation of sharp::upper_bound rather than
 * std::upper_bound for a few reasons. First, we prefer raw pointer
 * arrays over vectors for easy integration with SWIG bindings,
 * C code, and the potential future in which this runs on CUDA
 * architecture. Currently, the algorithms in the STL library are
 * not supported by the CUDA STL, but the types in <functional>
 * (i.e. std::less) are supported by the CUDA STL. Additionally, this
 * implementation of lower_bound is designed to reduce branching.
 *
 * \param   array   The array to search over
 * \param   N       The length of the array
 * \param   value   The value for upper-bound comparison
 * \param   cmp     The comparitor
 *
 * \return  Index of the upper bound
 */
template <floating T, typename C = std::less<T>>
constexpr size_t upper_bound(const T array[], const int N, const T& value, const C cmp = C{});

template <floating T>
constexpr T trapezoidal(const Fn<T, T> fn, const T a, const T b, const T n) noexcept;

}  // namespace libthermo
