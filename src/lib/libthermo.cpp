#include <libthermo.hpp>

namespace libthermo {

/**
 * @brief given a pressure array of decreasing values, find the index of the pressure level that
 * corresponds to the given value. This function is optimized for evenly distributed pressure
 * and will typically find the index in O(1) time.
 * 
 * @tparam T 
 * @param levels        pressure levels array
 * @param value         value to find
 * @param size          size of the array
 * @return constexpr size_t 
 */
template <floating T>
constexpr size_t index_pressure(const T levels[], const T value, const size_t size) noexcept {
    const size_t N = size - 1;
    const T p1 = levels[N];
    if (isnan(p1))
        return index_pressure(levels, value, N);
    const T p0 = levels[0];
    const T step = ((p1 - p0) / N);

    size_t idx = (size_t)((value / step) - (p0 / step));

    if (idx >= N)
        return N;

    while ((idx < N) && (value < levels[idx]))
        idx++;

    return idx;
}

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

/**
 * @brief Equivalent potential temperature, `theta-e (𝜃𝑒)`, is a quantity that is conserved during
 * changes to an air parcel's pressure (that is, during vertical motions in the atmosphere), even
 * if water vapor condenses during that pressure change. It is therefore more conserved than the
 * ordinary potential temperature, which remains constant only for unsaturated vertical motions
 * (pressure changes).
 * 
 * @tparam T 
 * @param pressure          `(Pa)`
 * @param temperature       `(K)`
 * @param dewpoint          `(K)`
 * @return `theta-e` -      `(K)`
 */
template <floating T>
constexpr T equivalent_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T r = saturation_mixing_ratio(pressure, dewpoint);
    const T e = saturation_vapor_pressure(dewpoint);
    const T t_l = 56. + 1. / (1. / (dewpoint - 56.) + log(temperature / dewpoint) / 800.);
    const T th_l =
      potential_temperature(pressure - e, temperature) * pow(temperature / t_l, 0.28 * r);

    return th_l * exp(r * (1. + 0.448 * r) * (3036. / t_l - 1.78));
}

/**
 * @brief theta_w
 * 
 * @tparam T 
 * @param pressure          `(Pa)`
 * @param temperature       `(K)`
 * @param dewpoint          `(K)`
 * @return constexpr T 
 */
template <floating T>
constexpr T wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T theta_e = equivalent_potential_temperature(pressure, temperature, dewpoint);
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

    return isnan(p) ? pressure : p;
}

template <floating T>
constexpr T lcl_pressure(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint), pressure);

    return fixed_point(find_lcl<T>, max_iters, eps, pressure, temperature, r);
}

template <floating T>
constexpr lcl<T>::lcl(
  const T pressure_, const T temperature_, const T dewpoint_, const T eps, const size_t max_iters
) noexcept {
    const T r = mixing_ratio(saturation_vapor_pressure(dewpoint_), pressure_);
    pressure = fixed_point(find_lcl<T>, max_iters, eps, pressure_, temperature_, r);
    temperature = dewpoint(pressure, r);
}

template <floating T>
constexpr T lcl<T>::wet_bulb_temperature(const T pressure, const T step) noexcept {
    return moist_lapse(this->pressure, pressure, this->temperature, step);
}

template <floating T>
constexpr size_t lcl<T>::index_on(const T pressure[], const size_t size) noexcept {
    return index_pressure(pressure, this->pressure, size);
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
    return lcl<T>(pressure, temperature, dewpoint, eps, max_iters)
      .wet_bulb_temperature(pressure, step);
}

/**
 * 
 * \author John Hart - NSSFC KCMO / NWSSPC OUN
 *
 * \brief Computes the difference between the wet-bulb potential<!--
 * --> temperatures for saturated and dry air given the temperature.
 *
 * The Wobus Function (wobf) is defined as the difference between
 * the wet-bulb potential temperature for saturated air (WBPTS)
 * and the wet-bulb potential temperature for dry air (WBPTD) given
 * the same temperature in Celsius.
 *
 * WOBF(T) := WBPTS - WBPTD
 *
 * Although WBPTS and WBPTD are functions of both pressure and
 * temperature, it is assumed their difference is a function of
 * temperature only. The difference is also proportional to the
 * heat imparted to a parcel.
 *
 * This function uses a polynomial approximation to the wobus function,
 * fitted to values in Table 78 of PP.319-322 of the Smithsonian Meteorological
 * Tables by Roland List (6th Revised Edition). Herman Wobus, a mathematician
 * for the Navy Weather Research Facility in Norfolk, VA computed these
 * coefficients a very long time ago, as he was retired as of the time of
 * the documentation found on this routine written in 1981.
 *
 * It was shown by Robert Davies-Jones (2007) that the Wobus function has
 * a slight dependence on pressure, which results in errors of up to 1.2
 * degrees Kelvin in the temperature of a lifted parcel.
 *
 * @param   temperature     (degK)
 *
 * @return  wobf            (degK)
 */
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