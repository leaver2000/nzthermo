#pragma once

#include <type_traits>

namespace libthermo {

#define LIMIT_ZERO(x) ((x) ? (x) : std::numeric_limits<T>::epsilon())

template <typename T>
concept floating = std::is_floating_point_v<T>;

template <typename R, typename... Args>
using Fn = R (*)(Args...);

template <floating T>
struct point {
    T x;
    T y;
};

}  // namespace libthermo