#ifndef _NZTHERMO_H_
#define _NZTHERMO_H_

/* include */
#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <array>

#include <common.hpp>
namespace nzt {

/* ........................................{ types  }........................................... */

template <typename T>
concept floating = std::is_floating_point_v<T>;

template <typename R, typename... Args>
using Fn = R (*)(Args...);

/* ........................................{ const  }........................................... */

static constexpr double T0 = 273.15; /* `(J/kg*K)` - freezing point in kelvin */
static constexpr double E0 = 611.21;  // `(Pa)` - vapor pressure at T0
static constexpr double Cpd = 1004.6662184201462;  // `(J/kg*K)` - specific heat of dry air
static constexpr double Rd = 287.04749097718457;  // `(J/kg*K)` - gas constant for dry air
static constexpr double Rv = 461.52311572606084;  // `(J/kg*K)` - gas constant for water vapor
static constexpr double epsilon = Rd / Rv;  // `Rd/Rv` - ratio of gas constants
static constexpr double Lv = 2501000.0;  // `(J/kg)` - latent heat of vaporization
static constexpr double P0 = 100000.0;  // `(Pa)` - standard pressure at sea level

/* ........................................{ struct }........................................... */
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
struct LCL {
    T pressure;
    T temperature;
};

template <floating T>
struct Parcel {
    T pressure;
    T temperature;
    T dewpoint;
};

/* ........................................{ common }........................................... */
template <floating T>
bool monotonically_decreasing(const T x[], const size_t size) noexcept;
FLOATING norm(const T x, const T x0, const T x1) noexcept;
FLOATING radians(const T degrees) noexcept;
FLOATING degrees(const T radians) noexcept;
FLOATING linear_interpolate(
  const T x, const T x0, const T x1, const T y0, const T y1, const bool log_x = false
) noexcept;
FLOATING interpolate_z(
  const size_t size, const T x, const T xp[], const T fp[], bool log_x = false
) noexcept;
FLOATING heaviside(const T x, const T h0) noexcept;
FLOATING omega(T T0, T T1, T T2) noexcept;
FLOATING domega(T T0, T T1, T T2) noexcept;
/* ........................................{ winds  }........................................... */

FLOATING wind_direction(const T u, const T v, const bool from = true) noexcept;
FLOATING wind_magnitude(const T u, const T v) noexcept;

template <floating T>
constexpr WindComponents<T> wind_components(const T direction, const T magnitude) noexcept;

/* ........................................{ thermo }........................................... */

FLOATING mixing_ratio(const T partial_press, const T total_press) noexcept;
FLOATING saturation_vapor_pressure(const T temperature) noexcept;
FLOATING virtual_temperature(const T temperature, const T mixing_ratio);
FLOATING saturation_mixing_ratio(const T pressure, const T temperature) noexcept;
FLOATING vapor_pressure(const T pressure, const T mixing_ratio) noexcept;
FLOATING dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept;
FLOATING dewpoint(const T vapor_pressure) noexcept;
FLOATING dewpoint(const T pressure, const T mixing_ratio) noexcept;
FLOATING potential_temperature(const T pressure, const T temperature) noexcept;  // theta
FLOATING equivalent_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept;  // theta_e
FLOATING wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept;  // theta_w

/* ........................................{ ode    }........................................... */

FLOATING rk2(Fn<T, T, T> fn, T x0, T x1, T y, T step /* = .1 */) noexcept;
template <floating T, floating... Args>
constexpr T fixed_point(
  const Fn<T, T, T, Args...> fn,
  const size_t max_iters,
  const T eps,
  const T x0,
  const Args... args
) noexcept;

FLOATING moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step
) noexcept;

/* ........................................{ lcl  }........................................... */

template <floating T>
constexpr LCL<T> lcl(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept;

FLOATING lcl_pressure(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept;

FLOATING wet_bulb_temperature(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps,
  const T step,
  const size_t max_iters
) noexcept;

/* ........................................{ sharp  }........................................... */
/**
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
 * \param   temperature     (degK)
 *
 * \return  wobf            (degK)
 */
FLOATING wobus(T temperature);

/* ........................................{ ecape  }........................................... */

}  // namespace nzt
#endif  // _NZTHERMO_H_