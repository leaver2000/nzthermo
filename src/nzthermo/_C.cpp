#include <cmath>
#include <functional>
#include <algorithm>

namespace nzt {
/* ................................................................................................
 - define
................................................................................................ */

#define NaN std::numeric_limits<T>::quiet_NaN()

/* ................................................................................................
 - types
................................................................................................ */

template <typename R, typename... Args>
using Fn = R (*)(Args...);

/* ................................................................................................
 - constants
................................................................................................ */

static constexpr double T0 = 273.15;  // `(J/kg*K)` - freezing point in kelvin
static constexpr double E0 = 611.21;  // `(Pa)` - vapor pressure at T0
static constexpr double Cpd = 1004.6662184201462;  // `(J/kg*K)` - specific heat of dry air
static constexpr double Rd = 287.04749097718457;  // `(J/kg*K)` - gas constant for dry air
static constexpr double Rv = 461.52311572606084;  // `(J/kg*K)` - gas constant for water vapor
static constexpr double Lv = 2501000.0;  // `(J/kg)` - latent heat of vaporization
static constexpr double epsilon = Rd / Rv;  // ratio of gas constants
static constexpr double P0 = 100000.0;  // `(Pa)` - standard pressure at sea level

/* ................................................................................................
 - common functions
................................................................................................ */

template <typename T>
    requires std::floating_point<T>
constexpr T linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1
) noexcept {
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}

/* ................................................................................................
 - thermodynamic functions
................................................................................................ */

template <typename T>
    requires std::floating_point<T>
constexpr T mixing_ratio(const T partial_press, const T total_press) noexcept {
    return epsilon * partial_press / (total_press - partial_press);
}

template <typename T>
    requires std::floating_point<T>
constexpr T mixing_ratio_from_dewpoint(const T pressure, const T dewpoint) noexcept {
    return mixing_ratio(saturation_vapor_pressure<T>(dewpoint), pressure);
}

template <typename T>
    requires std::floating_point<T>
constexpr T saturation_vapor_pressure(const T temperature) noexcept {
    return E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65));
}

template <typename T>
    requires std::floating_point<T>
constexpr T virtual_temperature(const T temperature, const T mixing_ratio) {
    return temperature * ((mixing_ratio + epsilon) / (epsilon * (1 + mixing_ratio)));
}

template <typename T>
    requires std::floating_point<T>
constexpr T saturation_mixing_ratio(const T pressure, const T temperature) noexcept {
    const T e = saturation_vapor_pressure(temperature);

    return epsilon * e / (pressure - e);
}

template <typename T>
    requires std::floating_point<T>
constexpr T vapor_pressure(const T pressure, const T mixing_ratio) noexcept {
    return pressure * mixing_ratio / (epsilon + mixing_ratio);
}

template <typename T>
    requires std::floating_point<T>
constexpr T dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept {
    return temperature * pow(pressure / reference_pressure, (Rd / Cpd));
}

template <typename T>
    requires std::floating_point<T>
constexpr T dewpoint(const T vapor_pressure) noexcept {
    const T ln = log(vapor_pressure / E0);

    return T0 + 243.5 * ln / (17.67 - ln);
}

template <typename T>
    requires std::floating_point<T>
constexpr T dewpoint(const T pressure, const T mixing_ratio) noexcept {
    return dewpoint(vapor_pressure(pressure, mixing_ratio));
}

template <typename T>
    requires std::floating_point<T>
constexpr T exner_function(const T pressure, const T reference_pressure = P0) noexcept {
    return pow(pressure / reference_pressure, Rd / Cpd);
}

/* theta */
template <typename T>
    requires std::floating_point<T>
constexpr T potential_temperature(const T pressure, const T temperature) noexcept {
    return temperature / exner_function(pressure);
}

/* theta_e */
template <typename T>
    requires std::floating_point<T>
constexpr T equivalent_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T r = saturation_mixing_ratio(pressure, dewpoint);
    const T e = saturation_vapor_pressure(dewpoint);
    const T t_l = 56 + 1.0 / (1.0 / (dewpoint - 56) + log(temperature / dewpoint) / 800.0);
    const T th_l =
      potential_temperature(pressure - e, temperature) * pow(temperature / t_l, 0.28 * r);

    return th_l * exp(r * (1 + 0.448 * r) * (3036.0 / t_l - 1.78));
}

/* theta_w */
template <typename T>
    requires std::floating_point<T>
constexpr T wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T theta_e = equivalent_potential_temperature<T>(pressure, temperature, dewpoint);
    if (theta_e <= 173.15)
        return theta_e;
    const T x = theta_e / T0;
    const T x2 = pow(x, 2);
    const T x3 = pow(x, 3);
    const T x4 = pow(x, 4);
    const T a = 7.101574 - 20.68208 * x + 16.11182 * x2 + 2.574631 * x3 - 5.205688 * x4;
    const T b = 1 - 3.552497 * x + 3.781782 * x2 - 0.6899655 * x3 - 0.5929340 * x4;
    const T theta_w = theta_e - exp(a / b);

    return theta_w;
}

/* ................................................................................................
- numerical methods
................................................................................................ */
/* RK2 */
template <typename T>
    requires std::floating_point<T>
constexpr T rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step /* = .1 */) noexcept {
    T k1, delta, abs_delta;
    size_t N = 1;

    delta = x1 - x0;
    abs_delta = fabs(delta);
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
 * @brief A fixed point of a function is the point at which evaluation of the function returns the
 *  point.
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
 * @return (T) fixed point (T) (if found, else NaN)
 */
template <typename T, typename... Args>
    requires std::floating_point<T>
constexpr T fixed_point(
  const Fn<T, T, T, Args...> fn,
  const size_t max_iters,
  const T eps,
  const T x0,
  const Args... args
) noexcept {
    T p0, p1, p2, delta, err;

    p0 = x0;
    for (size_t i = 0; i < max_iters; i++) {
        p1 = fn(p0, x0, args...);
        p2 = fn(p1, x0, args...);
        delta = p2 - 2.0 * p1 + p0;
        if (delta)
            p2 = p0 - pow(p1 - p0, 2) / delta; /* delta squared */

        err = p2;
        if (p0)
            err = fabs((p2 - p0) / p0); /* absolute relative error */

        if (err < eps)
            return p2;

        p0 = p2;
    }

    return NaN;
}

/* ................................................................................................
 - adiabatic processes
................................................................................................ */

/* using the rk2 method we can solve the ivp problem by integrating the moist lapse rate
 * equation from the initial pressure to the next pressure level */
template <typename T>
    requires std::floating_point<T>
constexpr T moist_lapse_solver(const T pressure, const T temperature) noexcept {
    const T r = saturation_mixing_ratio(pressure, temperature);

    return (Rd * temperature + Lv * r) /
      (Cpd + (Lv * Lv * r * epsilon / (Rd * temperature * temperature))) / pressure;
}

template <typename T>
    requires std::floating_point<T>
constexpr T moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step
) noexcept {
    return rk2<T>(moist_lapse_solver<T>, pressure, next_pressure, temperature, step);
}

/* LCL */

template <typename T>
    requires std::floating_point<T>
constexpr T lcl_solver(T pressure, T reference_pressure, T temperature, T mixing_ratio) noexcept {
    const T td = dewpoint(pressure, mixing_ratio);
    const T p = reference_pressure * pow(td / temperature, 1.0 / (Rd / Cpd));

    return std::isnan(p) ? pressure : p;
}

template <typename T>
    requires std::floating_point<T>
constexpr T lcl_pressure(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure);

    return fixed_point(lcl_solver<T>, max_iters, eps, pressure, temperature, r);
}

template <typename T>
    requires std::floating_point<T>
constexpr std::pair<T, T> lcl(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure);
    const T lcl_p = fixed_point(lcl_solver<T>, max_iters, eps, pressure, temperature, r);
    const T lcl_t = nzt::dewpoint(lcl_p, r);

    return std::make_pair(lcl_p, lcl_t);
}

template <typename T>
    requires std::floating_point<T>
constexpr T wet_bulb_temperature(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps,
  const T step,
  const size_t max_iters
) noexcept {
    const auto [lcl_p, lcl_t] = lcl(pressure, temperature, dewpoint, eps, max_iters);

    return moist_lapse(lcl_p, pressure, lcl_t, step);
}

}  // namespace nzt