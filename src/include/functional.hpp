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
constexpr T heaviside(const T x, const T h0) noexcept;

template <floating T>
constexpr T rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step = 1000.0) noexcept;

template <floating T, floating... Args>
constexpr T fixed_point(
  const Fn<T, T, T, Args...> fn,
  const size_t max_iters,
  const T eps,
  const T x0,
  const Args... args
) noexcept;

template <floating T, typename C = std::less<T>>
constexpr size_t lower_bound(const T array[], const int N, const T& value, const C cmp = C{});

template <floating T, typename C = std::less<T>>
constexpr size_t upper_bound(const T array[], const int N, const T& value, const C cmp = C{});

template <floating T>
constexpr T interpolate_1d(const T x, const T xp[], const T fp[], const size_t size) noexcept;

template <floating T>
constexpr point<T> intersect_1d(
  const T x[],
  const T a[],
  const T b[],
  const size_t size,
  const bool log_x = false,
  const bool increasing = true,
  const bool bottom = true
) noexcept;

}  // namespace libthermo
