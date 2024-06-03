#pragma once
#include <concepts>
#include <limits>
#include <vector>

namespace libthermo {

template <typename T>
concept floating = std::is_floating_point_v<T>;

template <typename R, typename... Args>
using Fn = R (*)(Args...);

#define NaN std::numeric_limits<T>::quiet_NaN() /* not a number */
#define FLOATING          \
    template <floating T> \
    constexpr T

#define VECTOR            \
    template <floating T> \
    constexpr std::vector<T>
}  // namespace libthermo