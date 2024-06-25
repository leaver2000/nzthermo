#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>
#include <array>

#include <functional.hpp>
#include <common.hpp>

namespace libthermo {
#define DEFAULT_STEP 1000.0  // `(Pa)` - default step for moist_lapse
#define DEFAULT_EPS 0.1      // default epsilon for lcl
#define DEFAULT_ITERS 5      // default number of iterations for lcl

static constexpr double g = 9.80665;              // `(m/s^2)` - acceleration due to gravity
static constexpr double ABSOLUTE_ZERO = -273.15;  // `(K)` - absolute zero temperature
static constexpr double T0 = 288.;                // `(K)` - standard temperature at sea level
static constexpr double P0 = 101325.0;            // `(Pa)` - standard pressure at sea level
static constexpr double E0 = 1.225;               // `(kg/m^3)` - standard air density at sea level
static constexpr double R = 8.31446261815324;     // `(J/mol*K)` - universal gas constant
static constexpr double T_0c = -ABSOLUTE_ZERO;    // `(K)` - 0C in Kelvin
static constexpr double Es_0c = 611.21;           // `(Pa)` - saturation vapor pressure @ 0C
static constexpr double Cp = 1004.6662184201462;  // `(J/kg*K)` - specific heat of dry air
static constexpr double Rd = 287.04749097718457;  // `(J/kg*K)` - gas constant for dry air
static constexpr double Rv = 461.52311572606084;  // `(J/kg*K)` - gas constant for water vapor
static constexpr double Lv = 2.50084e6;           // `(J/kg)` - latent heat of vaporization
static constexpr double Mw = 18.01528;            // `(g/mol)` - molecular weight of water
static constexpr double Md = 28.96546;            // `(g/mol)` - molecular weight of dry air
static constexpr double ELR = 6.5e-3;             // `(K/m)` - environmental lapse rate
static constexpr double epsilon = Mw / Md;        // `Mw / Md` - molecular weight ratio
static constexpr double kappa = Rd / Cp;          // `Rd / Cp`  - ratio of gas constants

template <floating T>
constexpr size_t index_pressure(const T x[], const T value, const size_t size) noexcept;

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
) noexcept;

template <floating T>
constexpr T wet_bulb_potential_temperature(
  const T pressure, const T temperature, const T dewpoint
) noexcept;

template <floating T>
constexpr T moist_lapse(
  const T pressure, const T next_pressure, const T temperature, const T step = DEFAULT_STEP
) noexcept;

template <floating T>
constexpr T lcl_pressure(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps = DEFAULT_EPS,
  const size_t max_iters = DEFAULT_ITERS
) noexcept;

template <floating T>
class lcl {
  public:
    T pressure, temperature;
    constexpr lcl() noexcept = default;
    constexpr lcl(const T pressure, const T temperature) noexcept :
        pressure(pressure), temperature(temperature){};
    constexpr lcl(
      const T pressure,
      const T temperature,
      const T dewpoint,
      const T eps = DEFAULT_EPS,
      const size_t max_iters = DEFAULT_ITERS
    ) noexcept;
    constexpr T wet_bulb_temperature(const T pressure, const T step = DEFAULT_STEP) noexcept;
    constexpr size_t index_on(const T pressure[], const size_t size) noexcept;
};

template <floating T>
constexpr T wet_bulb_temperature(
  const T pressure,
  const T temperature,
  const T dewpoint,
  const T eps = DEFAULT_EPS,
  const T step = DEFAULT_STEP,
  const size_t max_iters = DEFAULT_ITERS
) noexcept;

template <floating T>
constexpr T wobus(T temperature);

}  // namespace libthermo
