#pragma once

#include <cmath>

#include <common.hpp>

namespace libthermo {

template <floating T>
class wind_vector;

template <floating T>
class wind_components;

template <floating T>
constexpr T wind_direction(const T u, const T v) noexcept;
template <floating T>
constexpr T wind_magnitude(const T u, const T v) noexcept;

template <floating T>
class wind_components {
  public:
    T u, v;
    constexpr wind_components() noexcept = default;
    constexpr wind_components(const T u, const T v) noexcept : u(u), v(v){};
    constexpr explicit wind_components(const wind_vector<T>& uv) noexcept;
};

template <floating T>
class wind_vector {
  public:
    T direction, magnitude;
    constexpr wind_vector() noexcept = default;
    constexpr wind_vector(const T d, const T m) noexcept : direction(d), magnitude(m){};
    constexpr explicit wind_vector(const wind_components<T>& uv) noexcept;
};

}  // namespace libthermo
