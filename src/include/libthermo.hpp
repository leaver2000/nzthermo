#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <array>

#include <functional.hpp>
#include <common.hpp>

namespace libthermo {

/* ........................................{ const  }........................................... */

static constexpr double T0 = 273.15; /* `(J/kg*K)` - freezing point in kelvin */
static constexpr double E0 = 611.21;  // `(Pa)` - vapor pressure at T0
static constexpr double Cp = 1004.6662184201462;  // `(J/kg*K)` - specific heat of dry air
static constexpr double Rd = 287.04749097718457;  // `(J/kg*K)` - gas constant for dry air
static constexpr double Rv = 461.52311572606084;  // `(J/kg*K)` - gas constant for water vapor
static constexpr double Lv = 2501000.0;  // `(J/kg)` - latent heat of vaporization
static constexpr double P0 = 100000.0;  // `(Pa)` - standard pressure at sea level
static constexpr double Mw = 18.01528;  // `(g/mol)` - molecular weight of water
static constexpr double Md = 28.96546;  // `(g/mol)` - molecular weight of dry air
static constexpr double epsilon = Mw / Md;  // `Mw / Md` - molecular weight ratio
static constexpr double kappa = Rd / Cp;  // `Rd / Cp`  - ratio of gas constants

/* ........................................{ struct }........................................... */
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

template <floating T>
constexpr T mixing_ratio(const T partial_press, const T total_press) noexcept;

template <floating T>
constexpr T saturation_vapor_pressure(const T temperature) noexcept;

template <floating T>
constexpr T virtual_temperature(const T temperature, const T mixing_ratio);

template <floating T>
constexpr T saturation_mixing_ratio(const T pressure, const T temperature) noexcept;

template <floating T>
constexpr T vapor_pressure(const T pressure, const T mixing_ratio) noexcept;

template <floating T>
constexpr T dry_lapse(const T pressure, const T reference_pressure, const T temperature) noexcept;

template <floating T>
constexpr T dewpoint(const T vapor_pressure) noexcept;

template <floating T>
constexpr T dewpoint(const T pressure, const T mixing_ratio) noexcept;

template <floating T>
constexpr T potential_temperature(const T pressure, const T temperature) noexcept;  // theta

template <floating T>
constexpr T equivalent_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept;  // theta_e

template <floating T>
constexpr T wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept;  // theta_w

/* ........................................{ ode    }........................................... */

template <floating T>
constexpr T moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step
) noexcept;

/* ........................................{ lcl  }........................................... */

template <floating T>
constexpr LCL<T> lcl(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept;

template <floating T>
constexpr T lcl_pressure(
  const T pressure, const T temperature, const T dewpoint, const T eps, const size_t max_iters
) noexcept;

template <floating T>
constexpr T wet_bulb_temperature(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps,
  const T step,
  const size_t max_iters
) noexcept;

/* ........................................{ sharp  }........................................... */

template <floating T>
constexpr T wobus(T temperature);

/* ........................................{ ecape  }........................................... */

}  // namespace libthermo
