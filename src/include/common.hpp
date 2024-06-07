#pragma once

#include <type_traits>

namespace libthermo {

template <typename T>
concept floating = std::is_floating_point_v<T>;

template <typename R, typename... Args>
using Fn = R (*)(Args...);

}  // namespace libthermo