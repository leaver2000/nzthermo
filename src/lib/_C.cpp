#include <_C.hpp>

namespace nzt {

template <typename T, typename C = std::less<>>
[[nodiscard]] constexpr int upper_bound(
  const T array[], const int N, const T& value, const C cmp
) {
    int len = N;
    int idx = 0;
    while (len > 1) {
        int half = len / 2;
        idx += !cmp(value, array[idx + half - 1]) * half;
        len -= half;  // = ceil(len / 2)
    }
    return idx;
}

/* ........................................{ common }........................................... */
template <floating T>
bool monotonically_decreasing(const T x[], const size_t size) noexcept {
    // quick check on first and last element
    if (x[0] < x[1] || x[size - 1] > x[size - 2])
        return false;
    for (size_t i = 1; i < size; i++)
        if (x[i] > x[i - 1])
            return false;

    return true;
}

FLOATING degrees(const T radians) noexcept {
    return radians * 180.0 / M_PI;
}

FLOATING radians(const T degrees) noexcept {
    return degrees * M_PI / 180.0;
}

FLOATING norm(const T x, const T x0, const T x1) noexcept {
    return (x - x0) / (x1 - x0);
}

FLOATING linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1, const bool log_x
) noexcept {
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
}
template <floating T>
size_t search_sorted(
  const T x[], const T value, const size_t size, const bool inverted = false
) noexcept {
    size_t i = 0;
    if (inverted) {
        while (i < size && x[i] >= value)
            i++;
    } else {
        while (i < size && x[i] <= value)
            i++;
    }

    return i;
}

FLOATING interpolate_z(
  const size_t size, const T x, const T xp[], const T fp[], const bool log_x
) noexcept {
    size_t i = search_sorted(xp, x, size, true);
    if (i == 0)
        return fp[0];

    return linear_interpolate(x, xp[i - 1], xp[i], fp[i - 1], fp[i], log_x);
}
/*see: https://github.com/numpy/numpy/blob/main/numpy/_core/src/npymath/npy_math_internal.h.src#L426*/
FLOATING heaviside(const T x, const T h0) noexcept {
    if (isnan(x))
        return NaN;
    else if (x == 0)
        return h0;
    else if (x < 0)
        return 0.0;

    return 1.0;
}

// ecape functions

FLOATING omega(T T0, T T1, T T2) noexcept {
    return ((T0 - T1) / (T2 - T1)) * heaviside((T0 - T1) / (T2 - T1), 1) *
      heaviside((1 - (T0 - T1) / (T2 - T1)), 1) +
      heaviside(-(1 - (T0 - T1) / (T2 - T1)), 1);
}

FLOATING domega(T T0, T T1, T T2) noexcept {
    return (heaviside(T1 - T0, 1) - heaviside(T2 - T0, 1)) / (T2 - T1);
}

/* ........................................{ winds  }........................................... */

FLOATING wind_direction(const T u, const T v, const bool from) noexcept {
    T wdir = degrees(atan2(u, v));
    if (from)
        wdir = fmod(wdir - 180.0, 360.0);

    if (wdir <= 0)
        wdir = 360.0;
    if (u == 0 && v == 0)
        wdir = 0.0;

    return wdir;
}

FLOATING wind_magnitude(const T u, const T v) noexcept {
    return hypot(u, v);
}

template <floating T>
constexpr WindComponents<T> wind_components(const T direction, const T magnitude) noexcept {
    if (direction < 0 || direction > 360)
        return {NaN, NaN};

    const T u = magnitude * sin(radians(direction));
    const T v = magnitude * cos(radians(direction));

    return {-u, -v};
}

FLOATING mixing_ratio(const T partial_press, const T total_press) noexcept {
    return epsilon * partial_press / (total_press - partial_press);
}

FLOATING saturation_vapor_pressure(const T temperature) noexcept {
    return E0 * exp(17.67 * (temperature - T0) / (temperature - 29.65));
}

FLOATING virtual_temperature(const T temperature, const T mixing_ratio) {
    return temperature * ((mixing_ratio + epsilon) / (epsilon * (1 + mixing_ratio)));
}

FLOATING saturation_mixing_ratio(const T pressure, const T temperature) noexcept {
    const T e = saturation_vapor_pressure(temperature);

    return epsilon * e / (pressure - e);
}

FLOATING vapor_pressure(const T pressure, const T mixing_ratio) noexcept {
    return pressure * mixing_ratio / (epsilon + mixing_ratio);
}

FLOATING dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept {
    return temperature * pow(pressure / reference_pressure, (Rd / Cpd));
}

FLOATING dewpoint(const T vapor_pressure) noexcept {
    const T ln = log(vapor_pressure / E0);

    return T0 + 243.5 * ln / (17.67 - ln);
}

FLOATING dewpoint(const T pressure, const T mixing_ratio) noexcept {
    return dewpoint(vapor_pressure(pressure, mixing_ratio));
}

FLOATING exner_function(const T pressure, const T reference_pressure = P0) noexcept {
    return pow(pressure / reference_pressure, Rd / Cpd);
}

FLOATING potential_temperature(const T pressure, const T temperature) noexcept {
    return temperature / exner_function(pressure);
}

FLOATING equivalent_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept {
    const T r = saturation_mixing_ratio(pressure, dewpoint);
    const T e = saturation_vapor_pressure(dewpoint);
    const T t_l = 56 + 1.0 / (1.0 / (dewpoint - 56) + log(temperature / dewpoint) / 800.0);
    const T th_l =
      potential_temperature(pressure - e, temperature) * pow(temperature / t_l, 0.28 * r);

    return th_l * exp(r * (1 + 0.448 * r) * (3036.0 / t_l - 1.78));
}

FLOATING wet_bulb_potential_temperature(
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

/* ........................................{ ode    }........................................... */

/**
 * @brief Runge-Kutta 2nd order method for solving ordinary differential equations.
 */
FLOATING rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step /* = .1 */) noexcept {
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
 * @brief A fixed point of a function is the point at which evaluation of the
 * function returns the point.
 * @ref * https://docs.scipy.org/doc/scipy/tutorial/optimize.html#fixed-point-solving
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
template <floating T, floating... Args>
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

FLOATING _moist_lapse(const T pressure, const T temperature) noexcept {
    const T r = saturation_mixing_ratio(pressure, temperature);

    return (Rd * temperature + Lv * r) /
      (Cpd + (Lv * Lv * r * epsilon / (Rd * temperature * temperature))) / pressure;
}

FLOATING moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step
) noexcept {
    return rk2<T>(_moist_lapse<T>, pressure, next_pressure, temperature, step);
}

FLOATING find_lcl(T pressure, T reference_pressure, T temperature, T mixing_ratio) noexcept {
    const T td = dewpoint(pressure, mixing_ratio);
    const T p = reference_pressure * pow(td / temperature, 1.0 / (Rd / Cpd));

    return std::isnan(p) ? pressure : p;
}

FLOATING lcl_pressure(
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
    const T lcl_t = nzt::dewpoint(lcl_p, r);

    return {lcl_p, lcl_t};
}

FLOATING wet_bulb_temperature(
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

/* ........................................{ sharp  }...........................................


   ........................................{ sharp  }........................................... */

FLOATING wobus(T temperature) {
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

}  // namespace nzt