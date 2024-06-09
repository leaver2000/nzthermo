#include <libthermo.hpp>

namespace libthermo {

template <floating T>
constexpr T mixing_ratio(const T partial_press, const T total_press) noexcept {
    return epsilon * partial_press / (total_press - partial_press);
}

template <floating T>
constexpr T saturation_vapor_pressure(const T temperature) noexcept {
    return E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65));
}

template <floating T>
constexpr T virtual_temperature(const T temperature, const T mixing_ratio) {
    return temperature * ((mixing_ratio + epsilon) / (epsilon * (1 + mixing_ratio)));
}

template <floating T>
constexpr T virtual_temperature(const T pressure, const T temperature, const T dewpoint) {
    return virtual_temperature(temperature, saturation_mixing_ratio(pressure, dewpoint));
}

template <floating T>
constexpr T saturation_mixing_ratio(const T pressure, const T temperature) noexcept {
    return mixing_ratio(saturation_vapor_pressure(temperature), pressure);
}

template <floating T>
constexpr T vapor_pressure(const T pressure, const T mixing_ratio) noexcept {
    return pressure * mixing_ratio / (epsilon + mixing_ratio);
}

template <floating T>
constexpr T dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept {
    return temperature * pow(pressure / reference_pressure, kappa);
}

template <floating T>
constexpr T dewpoint(const T vapor_pressure) noexcept {
    const T ln = log(vapor_pressure / E0);

    return T0 + 243.5 * ln / (17.67 - ln);
}

template <floating T>
constexpr T dewpoint(const T pressure, const T mixing_ratio) noexcept {
    return dewpoint(vapor_pressure(pressure, mixing_ratio));
}

template <floating T>
constexpr T exner_function(const T pressure, const T reference_pressure = P0) noexcept {
    return pow(pressure / reference_pressure, kappa);
}

template <floating T>
constexpr T potential_temperature(const T pressure, const T temperature) noexcept {
    return temperature / exner_function(pressure);
}

template <floating T>
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

template <floating T>
constexpr T wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T theta_e = equivalent_potential_temperature<T>(pressure, temperature, dewpoint);
    if (theta_e <= 173.15)
        return theta_e;
    const T x = theta_e / T0;
    const T x2 = x * x;
    const T x3 = x2 * x;
    const T x4 = x3 * x;
    const T a = 7.101574 - 20.68208 * x + 16.11182 * x2 + 2.574631 * x3 - 5.205688 * x4;
    const T b = 1 - 3.552497 * x + 3.781782 * x2 - 0.6899655 * x3 - 0.5929340 * x4;
    const T theta_w = theta_e - exp(a / b);

    return theta_w;
}

template <floating T>
constexpr T _moist_lapse(const T pressure, const T temperature) noexcept {
    const T r = saturation_mixing_ratio(pressure, temperature);

    return ((Rd * temperature + Lv * r) /
            (Cp + (Lv * Lv * r * epsilon / (Rd * temperature * temperature)))) /
      pressure;
}

template <floating T>
constexpr T moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step
) noexcept {
    return rk2<T>(_moist_lapse<T>, pressure, next_pressure, temperature, step);
}

template <floating T>
constexpr T find_lcl(T pressure, T reference_pressure, T temperature, T mixing_ratio) noexcept {
    const T td = dewpoint(pressure, mixing_ratio);
    const T p = reference_pressure * pow(td / temperature, 1.0 / (Rd / Cp));

    return std::isnan(p) ? pressure : p;
}

template <floating T>
constexpr T lcl_pressure(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure);

    return fixed_point(find_lcl<T>, max_iters, eps, pressure, temperature, r);
}
template <floating T>
constexpr LCL<T> lcl(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure);
    const T lcl_p = fixed_point(find_lcl<T>, max_iters, eps, pressure, temperature, r);
    const T lcl_t = libthermo::dewpoint(lcl_p, r);

    return {lcl_p, lcl_t};
}

template <floating T>
constexpr T wet_bulb_temperature(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps,
  const T step,
  const size_t max_iters
) noexcept {
    const LCL x = lcl(pressure, temperature, dewpoint, eps, max_iters);
    return moist_lapse(x.pressure, pressure, x.temperature, step);
}

template <floating T>
constexpr T wobus(T temperature) {
    T pol;
    const T x = temperature - T0 - 20.0;
    if (x <= 0.0) {
        pol = 1.0 +
          x *
            (-8.841660499999999e-03 +
             x *
               (1.4714143e-04 +
                x * (-9.671989000000001e-07 + x * (-3.2607217e-08 + x * (-3.8598073e-10)))));
        pol = pol * pol;
        return (15.13 / (pol * pol)) + T0;
    }
    pol = x *
      (4.9618922e-07 +
       x * (-6.1059365e-09 + x * (3.9401551e-11 + x * (-1.2588129e-13 + x * (1.6688280e-16)))));
    pol = 1.0 + x * (3.6182989e-03 + x * (-1.3603273e-05 + pol));
    pol = pol * pol;
    return (29.93 / (pol * pol) + 0.96 * x - 14.8) + T0;
}

}  // namespace libthermo