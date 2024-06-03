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
    const T e = saturation_vapor_pressure(temperature);

    return epsilon * e / (pressure - e);
}

template <floating T>
constexpr T vapor_pressure(const T pressure, const T mixing_ratio) noexcept {
    return pressure * mixing_ratio / (epsilon + mixing_ratio);
}

template <floating T>
constexpr T dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept {
    return temperature * pow(pressure / reference_pressure, (Rd / Cpd));
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
    return pow(pressure / reference_pressure, Rd / Cpd);
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

    return (Rd * temperature + Lv * r) /
      (Cpd + (Lv * Lv * r * epsilon / (Rd * temperature * temperature))) / pressure;
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
    const T p = reference_pressure * pow(td / temperature, 1.0 / (Rd / Cpd));

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
constexpr T downdraft_cape(
  const T pressure[], const T temperature[], const T dewpoint[], const size_t size
) noexcept {
    T p, t, td, delta, p_top, t_top, td_top, wb_top, trace, ln, delta_start;
    size_t start, stop;
    start = 0;
    for (size_t i = 0; i < size; i++) {
        if (pressure[i] >= 7e4) {  // start a 7000. Pa
            continue;

        } else if (start == 0) {
            start = i;
        }

        if (pressure[i] <= 5e4) {  // stop at 5000. Pa
            stop = i;
            break;
        }
    }
    p_top = pressure[stop];  //  # (N,)
    t_top = temperature[stop];  // # (N,)
    td_top = dewpoint[stop];  //];  # (N,)

    wb_top = wet_bulb_temperature(p_top, t_top, td_top, .1, 1000.0, 50);  // # (N,)

    p = pressure[start];
    t = temperature[start];
    td = dewpoint[start];

    trace = moist_lapse(p, wb_top, p_top, 1000.0);
    delta_start = virtual_temperature(p, t, td) - virtual_temperature(p, trace, trace);
    T ln_start = std::log(p);

    T dcape = 0.0;
    for (size_t i = start + 1; i < stop; i++) {
        p = pressure[i];
        t = temperature[i];
        td = dewpoint[i];
        trace = moist_lapse(p, wb_top, p_top, 1000.0);
        delta = virtual_temperature(p, t, td) - virtual_temperature(p, trace, trace);

        ln = std::log(p);
        dcape += (delta - delta_start) * (ln + ln_start) / 2.0;
        // next iter
        ln_start = ln;
        delta_start = delta;
    }
    // dcape = -(Rd * F.nantrapz(delta, logp, axis=1))
    return -Rd * dcape;
}

/* ........................................{ sharp  }...........................................


   ........................................{ sharp  }........................................... */

template <floating T>
constexpr T wobus(T temperature) {
    T pol;
    const T x = temperature - T0 - 20.0;
    if (x <= 0.0) {
        pol = 1.0f +
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