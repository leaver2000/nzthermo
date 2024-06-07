#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <array>

#include <functional.hpp>
#include <common.hpp>

namespace libthermo {

template <floating T>
struct WindVector {
    T speed;
    T direction;
};

template <floating T>
struct WindComponents {
    T u;
    T v;
};

template <floating T>
constexpr WindComponents<T> wind_components(const T direction, const T magnitude) noexcept;

template <floating T>
constexpr T wind_direction(const T u, const T v, const bool from = false) noexcept;

template <floating T>
constexpr T wind_magnitude(const T u, const T v) noexcept;

}  // namespace libthermo
