/*
               Copyright © 2008, UChicago Argonne, LLC
                       All Rights Reserved

                        WBGT, Version 1.1

			     James C. Liljegren
              Decision & Information Sciences Division

			     OPEN SOURCE LICENSE

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, 
   this list of conditions and the following disclaimer.  Software changes, 
   modifications, or derivative works, should be noted with comments and 
   the author and organization’s name.

2. Redistributions in binary form must reproduce the above copyright notice, 
   this list of conditions and the following disclaimer in the documentation 
   and/or other materials provided with the distribution.

3. Neither the names of UChicago Argonne, LLC or the Department of Energy 
   nor the names of its contributors may be used to endorse or promote products 
   derived from this software without specific prior written permission.

4. The software and the end-user documentation included with the 
   redistribution, if any, must include the following acknowledgment:

   "This product includes software produced by UChicago Argonne, LLC 
   under Contract No. DE-AC02-06CH11357 with the Department of Energy.”

***************************************************************************************************
DISCLAIMER

THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF ANY KIND.

NEITHER THE UNITED STATES GOVERNMENT, NOR THE UNITED STATES DEPARTMENT OF ENERGY, 
NOR UCHICAGO ARGONNE, LLC, NOR ANY OF THEIR EMPLOYEES, MAKES ANY WARRANTY, EXPRESS 
OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, 
COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, DATA, APPARATUS, PRODUCT, OR 
PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.

**************************************************************************************************/
#include <stdio.h>
#include <math.h>
#include "wbgt.h"
/*.............................{ thermodynamic functions }.......................................*/

/** 
 * @brief Calculate the specific heat of air, J/(kg K)
 * @param temperature temperature, K
 * @returns specific heat of air, J/(kg K)
 * @ref BSL, page 23.
 */
double viscosity(double temperature) {
  const double sigma = 3.617;
  const double eps_kappa = 97.0;
  double Tr;
  double omega;

  Tr = temperature / eps_kappa;
  omega = (Tr - 2.9) / 0.4 * (-0.034) + 1.048;
  return (2.6693E-6 * sqrt(M_AIR * temperature) / (sigma * sigma * omega));
}

/**
 * @brief Calculate the thermal conductivity of air, W/(m K)
 * @param temperature temperature, K
 * @returns thermal conductivity of air, W/(m K)
 * @ref BSL, page 257.
 */
double thermal_conductivity(double temperature) {
  return ((Cp + 1.25 * R_AIR) * viscosity(temperature));
}

/**
 * @brief saturation vapor pressure (mb) over liquid water (phase = 0) or ice (phase = 1).
 * @ref Buck's (1981) approximation (eqn 3) of Wexler's (1976) formulae.
 */
double saturation_vapor_pressure(double temperature, Phase phase) {
  double y;
  double es;
  if ((phase == LIQUID) || (phase == INFER && temperature > T0)) {
    y = (temperature - T0) / (temperature - 32.18);
    es = 6.1121 * exp(17.502 * y);
    //! es = (1.0007 + (3.46E-6 * pres)) * es  correction for moist air, if pressure is available */
  } else {  // ice
    y = (temperature - T0) / (temperature - 0.6);
    es = 6.1115 * exp(22.452 * y);
    /* es = (1.0003 + (4.18E-6 * pres)) * es  correction for moist air, if pressure is available */
  }

  // correction for moist air, if pressure is not available; for pressure > 800 mb
  es *= 1.004;
  /*    es = 1.0034 * es;  correction for moist air, if pressure is not available; for pressure down to 200 mb */

  return es;
}

/**
 * @brief calculate the dew point (phase=0) or frost point (phase=1) 
 * @returns temperature, K.
 */
double dewpoint(double vapor_pressure, Phase phase) {
  double ln;

  if ((phase == LIQUID) || (phase == INFER && vapor_pressure > E0)) {
    ln = log(vapor_pressure / (6.1121 * 1.004));
    return T0 + 240.97 * ln / (17.502 - ln);
  }
  /* frost point */
  ln = log(vapor_pressure / (6.1115 * 1.004));
  return T0 + 272.55 * ln / (22.452 - ln);
}

typedef enum {
  SPHERE,
  CYLINDER
} ShapeType;

double convective_heat_transfer(double pressure, double temperature, double speed, double type) {
  double density;
  double Re;  // Reynolds number
  double Nu;  // Nusselt number
  double diameter;
  density = pressure * 100. / (R_AIR * temperature);
  if (type == SPHERE) {
    diameter = GLOBE_DIAMETER;
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(temperature);
    Nu = 2.0 + 0.6 * sqrt(Re) * pow(Pr, 0.3333);
  } else {
    diameter = WICK_DIAMETER;
    Re = max(speed, MIN_SPEED) * density * diameter / viscosity(temperature);
    Nu = 0.281 * pow(Re, (1.0 - 0.4)) * pow(Pr, (1.0 - 0.56));
  }

  return Nu * thermal_conductivity(temperature) / diameter;
}

/**
 *  @brief calculate the diffusivity of water vapor in air, m2/s
 *  @ref BSL, page 505.
 */
double diffusivity(double pressure, double temperature) {
  const double Pcrit_air = 36.4;
  const double Pcrit_h2o = 218.;
  const double Tcrit_air = 132.;
  const double Tcrit_h2o = 647.3;
  const double a = 3.640e-4;
  const double b = 2.334;

  double Patm, Pcrit13, Tcrit512, Tcrit12, Mmix;

  Pcrit13 = pow((Pcrit_air * Pcrit_h2o), (1. / 3.));
  Tcrit512 = pow((Tcrit_air * Tcrit_h2o), (5. / 12.));
  Tcrit12 = sqrt(Tcrit_air * Tcrit_h2o);
  Mmix = sqrt(1. / M_AIR + 1. / M_H2O);
  Patm = pressure / 1013.25; /* convert pressure from mb to atmospheres */

  return a * pow((temperature / Tcrit12), b) * Pcrit13 * Tcrit512 * Mmix / Patm * 1e-4;
}

/**
 * @brief: calculate the heat of evaporation, J/(kg K), for temperature in the range 283-313 K.
 * @param temperature K
 * @returns heat of evaporation, J/(kg K)
 * @ref: Van Wylen and Sonntag, Table A.1.1
 */
double evaporation(double temperature) {
  return (313.15 - temperature) / 30. * (-71100.0) + 2.4073E6;
}

/**
 *  Purpose: calculate the atmospheric emissivity.
 *
 *  Reference: Oke (2nd edition), page 373.
 */
double emissivity(double temperature, double rh) {
  double e;

  e = rh * saturation_vapor_pressure(temperature, LIQUID);
  return (0.575 * pow(e, 0.143));
}

double schmidt_number(double pressure, double temperature) {
  double Sc;

  Sc = viscosity(temperature);
  Sc /= ((pressure * 100. / (R_AIR * temperature)) * diffusivity(pressure, temperature));
  return Sc;
}
/*...................................{ solar functions }.........................................*/

/**
 * 'daynum()' returns the sequential daynumber of a calendar date during a
 *  Gregorian calendar year (for years 1 onward).
 *  The integer arguments are the four-digit year, the month number, and
 *  the day of month number.
 *  (Jan. 1 = 01/01 = 001; Dec. 31 = 12/31 = 365 or 366.)
 *  A value of -1 is returned if the year is out of bounds.
 *

 * @author Nels Larson
 *         Pacific Northwest National Laboratory
 *         P.O. Box 999
 *         Richland, WA 99352
 *         U.S.A.
 */
int daynum(int year, int month, int day) {
  const int begmonth[13] = {0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int dnum;
  int leapyr = 0;

  // There is no year 0 in the Gregorian calendar and the leap year cycle changes for earlier years
  if (year < 1) {
    return -1;
  }

  // Leap years are divisible by 4, except for centurial years not divisible by 400.
  if (((year % 4) == 0 && (year % 100) != 0) || (year % 400) == 0) {
    leapyr = 1;
  }

  dnum = begmonth[month] + day;
  if (leapyr && (month > 2)) {
    dnum += 1;
  }

  return dnum;
}

/**
 * @paragraph{Function: solarposition} 
 *  solarposition employs the low precision formulas for the Sun's coordinates
 *  given in the "Astronomical Almanac" of 1990 to compute the Sun's apparent
 *  right ascension, apparent declination, altitude, atmospheric refraction
 *  correction applicable to the altitude, azimuth, and distance from Earth.
 *  The "Astronomical Almanac" (A. A.) states a precision of 0.01 degree for the
 *  apparent coordinates between the years 1950 and 2050, and an accuracy of
 *  0.1 arc minute for refraction at altitudes of at least 15 degrees.
 * 
 *  The following assumptions and simplifications are made:
 *  -> refraction is calculated for standard atmosphere pressure and temperature
 *     at sea level.
 *  -> diurnal parallax is ignored, resulting in 0 to 9 arc seconds error in
 *     apparent position.
 *  -> diurnal aberration is also ignored, resulting in 0 to 0.02 second error
 *     in right ascension and 0 to 0.3 arc second error in declination.
 *  -> geodetic site coordinates are used, without correction for polar motion
 *     (maximum amplitude of 0.3 arc second) and local gravity anomalies.
 *  -> local mean sidereal time is substituted for local apparent sidereal time
 *     in computing the local hour angle of the Sun, resulting in an error of
 *     about 0 to 1 second of time as determined explicitly by the equation of
 *     the equinoxes.
 *
 *  Right ascension is measured in hours from 0 to 24, and declination in
 *  degrees from 90 to -90.
 *  Altitude is measured from 0 degrees at the horizon to 90 at the zenith or
 *  -90 at the nadir. Azimuth is measured from 0 to 360 degrees starting at
 *  north and increasing toward the east at 90.
 *  The refraction correction should be added to the altitude if Earth's
 *  atmosphere is to be accounted for.
 *  Solar distance from Earth is in astronomical units, 1 a.u. representing the
 *  mean value.
 *
 *  The necessary input parameters are:
 *  -> the date, specified in one of three ways:
 *       1) year, month, day.fraction
 *       2) year, daynumber.fraction
 *       3) days.fraction elapsed since January 0, 1900.
 *  -> site geodetic (geographic) latitude and longitude.
 *
 *  Refer to the function declaration for the parameter type specifications and
 *  formats.
 *
 *  solarposition() returns -1 if an input parameter is out of bounds, or 0 if
 *  values were written to the locations specified by the output parameters.
 * 
 * @author Nels Larson
 *         Pacific Northwest National Laboratory
 *         P.O. Box 999
 *         Richland, WA 99352
 *         U.S.A.
 * @version 3.0
 * @date February 20, 1992.
 */
int solarposition(
  int year,
  int month,
  double day,
  double days_1900,
  double latitude,
  double longitude,
  double* ap_ra,
  double* ap_dec,
  double* altitude,
  double* refraction,
  double* azimuth,
  double* distance
) {
  int daynumber;  //        Sequential daynumber during a year.
  int delta_days;  //       Whole days since 2000 January 0.
  int delta_years;  //      Whole years since 2000.
  double cent_J2000;  //     Julian centuries since epoch J2000.0 at 0h UT.
  double cos_alt;  //        Cosine of the altitude of Sun.
  double cos_apdec;  //      Cosine of the apparent declination of Sun.
  double cos_az;  //         Cosine of the azimuth of Sun.
  double cos_lat;  //        Cosine of the site latitude.
  double cos_lha;  //        Cosine of the local apparent hour angle of Sun.
  double days_J2000;  //     Days since epoch J2000.0.
  double ecliptic_long;  //  Solar ecliptic longitude.
  double lmst;  //           Local mean sidereal time.
  double local_ha;  //       Local mean hour angle of Sun.
  double gmst0h,  //         Greenwich mean sidereal time at 0 hours UT.
    integral, /* Integral portion of double precision number. */
    mean_anomaly, /* Earth mean anomaly. */
    mean_longitude, /* Solar mean longitude. */
    mean_obliquity, /* Mean obliquity of the ecliptic. */
    sin_apdec, /* Sine of the apparent declination of Sun. */
    sin_az, /* Sine of the azimuth of Sun. */
    sin_lat, /* Sine of the site latitude. */
    tan_alt, /* Tangent of the altitude of Sun. */
    ut; /* UT hours since midnight. */

  /* Earth mean atmospheric temperature at sea level */
  double temp = 15.0; /*   in degrees Celsius. */
  /* Earth mean atmospheric pressure at sea level */
  double pressure = 1013.25; /*   in millibars. */
  // Check latitude and longitude for proper range before calculating dates.
  if (latitude < -90.0 || latitude > 90.0 || longitude < -180.0 || longitude > 180.0) {
    return -1;
  }

  /* If year is not zero then assume date is specified by year, month, day.
   * If year is zero then assume date is specified by days_1900.
   * Date given by {year, month, day} or {year, 0, daynumber}. */
  if (year != 0) {
    if (year < 1950 || year > 2049) {
      return -1;
    }
    if (month != 0) {
      if (month < 1 || month > 12 || day < 0.0 || day > 33.0) {
        return -1;
      }
      daynumber = daynum(year, month, (int)day);
    } else {
      if (day < 0.0 || day > 368.0) {
        return -1;
      }
      daynumber = (int)day;
    }

    // Construct Julian centuries since J2000 at 0 hours UT of date days.fraction since J2000, and UT hours.
    delta_years = year - 2000;

    // delta_days is days from 2000/01/00 (1900's are negative).
    delta_days = delta_years * 365 + delta_years / 4 + daynumber;
    if (year > 2000) {
      delta_days += 1;
    }
    // J2000 is 2000/01/01.5
    days_J2000 = delta_days - 1.5;

    cent_J2000 = days_J2000 / 36525.0;

    ut = modf(day, &integral);
    days_J2000 += ut;
    ut *= 24.0;
    /* Date given by days_1900. */
  } else {
    /* days_1900 is 18262 for 1950/01/00, and 54788 for 2049/12/32.
     * A. A. 1990, K2-K4. */
    if (days_1900 < 18262.0 || days_1900 > 54788.0) {
      return -1;
    }

    /* Construct days.fraction since J2000, UT hours, and
     * Julian centuries since J2000 at 0 hours UT of date.
     */
    /* days_1900 is 36524 for 2000/01/00. J2000 is 2000/01/01.5 */
    days_J2000 = days_1900 - 36525.5;

    ut = modf(days_1900, &integral) * 24.0;

    cent_J2000 = (integral - 36525.5) / 36525.0;
  }

  /* Compute solar position parameters.
   * A. A. 1990, C24.
   */
  mean_anomaly = (357.528 + 0.9856003 * days_J2000);
  mean_longitude = (280.460 + 0.9856474 * days_J2000);

  /* Put mean_anomaly and mean_longitude in the range 0 -> 2 pi. */
  mean_anomaly = modf(mean_anomaly / 360.0, &integral) * TWOPI;
  mean_longitude = modf(mean_longitude / 360.0, &integral) * TWOPI;

  mean_obliquity = (23.439 - 4.0e-7 * days_J2000) * DEG_RAD;
  ecliptic_long =
    ((1.915 * sin(mean_anomaly)) + (0.020 * sin(2.0 * mean_anomaly))) * DEG_RAD + mean_longitude;

  *distance = 1.00014 - 0.01671 * cos(mean_anomaly) - 0.00014 * cos(2.0 * mean_anomaly);

  // Tangent of ecliptic_long separated into sine and cosine parts for ap_ra.
  *ap_ra = atan2(cos(mean_obliquity) * sin(ecliptic_long), cos(ecliptic_long));

  // Change range of ap_ra from -pi -> pi to 0 -> 2 pi.
  if (*ap_ra < 0.0)
    *ap_ra += TWOPI;
  // Put ap_ra in the range 0 -> 24 hours.
  *ap_ra = modf(*ap_ra / TWOPI, &integral) * 24.0;

  *ap_dec = asin(sin(mean_obliquity) * sin(ecliptic_long));

  /* Calculate local mean sidereal time. A. A. 1990, B6-B7. */
  /* Horner's method of polynomial exponent expansion used for gmst0h. */
  gmst0h =
    24110.54841 + cent_J2000 * (8640184.812866 + cent_J2000 * (0.093104 - cent_J2000 * 6.2e-6));

  // Convert gmst0h from seconds to hours and put in the range 0 -> 24. */
  gmst0h = modf(gmst0h / 3600.0 / 24.0, &integral) * 24.0;
  if (gmst0h < 0.0) {
    gmst0h += 24.0;
  }

  // Ratio of lengths of mean solar day to mean sidereal day is 1.00273790934
  // in 1990. Change in sidereal day length is < 0.001 second over a century.
  // A. A. 1990, B6.
  lmst = gmst0h + (ut * 1.00273790934) + longitude / 15.0;
  /* Put lmst in the range 0 -> 24 hours. */
  lmst = modf(lmst / 24.0, &integral) * 24.0;
  if (lmst < 0.0) {
    lmst += 24.0;
  }

  /**
   * @brief Calculate local hour angle, altitude, azimuth, and refraction correction.
   * @ref A. A. 1990, B61-B62.
   */
  local_ha = lmst - *ap_ra;
  // Put hour angle in the range -12 to 12 hours.
  if (local_ha < -12.0) {
    local_ha += 24.0;
  } else if (local_ha > 12.0) {
    local_ha -= 24.0;
  }

  // Convert latitude and local_ha to radians.
  latitude *= DEG_RAD;
  local_ha = local_ha / 24.0 * TWOPI;

  cos_apdec = cos(*ap_dec);
  sin_apdec = sin(*ap_dec);
  cos_lat = cos(latitude);
  sin_lat = sin(latitude);
  cos_lha = cos(local_ha);

  *altitude = asin(sin_apdec * sin_lat + cos_apdec * cos_lha * cos_lat);

  cos_alt = cos(*altitude);

  // Avoid tangent overflow at altitudes of +-90 degrees. 1.57079615 radians is equal to 89.99999 degrees.
  if (fabs(*altitude) < 1.57079615) {
    tan_alt = tan(*altitude);
  } else {
    tan_alt = 6.0e6;
  }

  cos_az = (sin_apdec * cos_lat - cos_apdec * cos_lha * sin_lat) / cos_alt;
  sin_az = -(cos_apdec * sin(local_ha) / cos_alt);
  *azimuth = acos(cos_az);

  /* Change range of azimuth from 0 -> pi to 0 -> 2 pi. */
  if (atan2(sin_az, cos_az) < 0.0)
    *azimuth = TWOPI - *azimuth;

  /* Convert ap_dec, altitude, and azimuth to degrees. */
  *ap_dec *= RAD_DEG;
  *altitude *= RAD_DEG;
  *azimuth *= RAD_DEG;

  /* Compute refraction correction to be added to altitude to obtain actual position.
   * Refraction calculated for altitudes of -1 degree or more allows for a
   * pressure of 1040 mb and temperature of -22 C. Lower pressure and higher
   * temperature combinations yield less than 1 degree refraction.
   * 
   * NOTE:
   *  The two equations listed in the A. A. have a crossover altitude of
   *  19.225 degrees at standard temperature and pressure. This crossover point
   *  is used instead of 15 degrees altitude so that refraction is smooth over
   *  the entire range of altitudes. The maximum residual error introduced by
   *  this smoothing is 3.6 arc seconds at 15 degrees. Temperature or pressure
   *  other than standard will shift the crossover altitude and change the error.
   */
  if (*altitude < -1.0 || tan_alt == 6.0e6)
    *refraction = 0.0;
  else {
    if (*altitude < 19.225) {
      *refraction = (0.1594 + (*altitude) * (0.0196 + 0.00002 * (*altitude))) * pressure;
      *refraction /= (1.0 + (*altitude) * (0.505 + 0.0845 * (*altitude))) * (273.0 + temp);
    } else {
      *refraction = 0.00452 * (pressure / (273.0 + temp)) / tan_alt;
    }
  }
  /* To match Michalsky's sunae program, the following line was inserted
   * by JC Liljegren to add the refraction correction to the solar altitude
   */
  *altitude = *altitude + *refraction;
  return 0;
}

/*
 * Purpose: to calculate the cosine solar zenith angle and the fraction of the
 *           solar irradiance due to the direct beam.
 *    
 * Author: James C. Liljegren 
 *         Decision and Information Sciences Division
 *         Argonne National Laboratory
 */
int calc_solar_parameters(
  int year, int month, double day, double lat, double lon, double* solar, double* cza, double* fdir
) {
  double toasolar, normsolar;
  double days_1900 = 0.0, ap_ra, ap_dec, elev, refr, azim, soldist;

  if (!solarposition(
        year,
        month,
        day,
        days_1900,
        (double)lat,
        (double)lon,
        &ap_ra,
        &ap_dec,
        &elev,
        &refr,
        &azim,
        &soldist
      )) {
    return -1;
  }

  *cza = cos((90. - elev) * DEG_RAD);
  toasolar = SOLAR_CONST * max(0., *cza) / (soldist * soldist);

  // if the sun is not fully above the horizon set the maximum (top of atmosphere) solar = 0
  if (*cza < CZA_MIN) {
    toasolar = 0.0;
  }
  if (toasolar > 0.) {
    // account for any solar sensor calibration errors and make the solar irradiance consistent with normsolar
    normsolar = min(*solar / toasolar, NORMSOLAR_MAX);
    *solar = normsolar * toasolar;

    // calculate the fraction of the solar irradiance due to the direct beam
    if (normsolar > 0.) {
      *fdir = exp(3. - 1.34 * normsolar - 1.65 / normsolar);
      *fdir = max(min(*fdir, 0.9), 0.0);
    } else {
      *fdir = 0.0;
    }
  } else {
    *fdir = 0.0;
  }

  return 0;
}

/**
 * @brief estimate 2-m wind speed for all stability conditions
 * @ref EPA-454/5-99-005, 2000, section 6.2.5
 */
double est_wind_speed(double speed, double zspeed, int stability_class, int urban) {
  double urban_exp[6] = {0.15, 0.15, 0.20, 0.25, 0.30, 0.30};
  double rural_exp[6] = {0.07, 0.07, 0.10, 0.15, 0.35, 0.55};
  double exponent;
  double est_speed;

  if (urban) {
    exponent = urban_exp[stability_class - 1];
  } else {
    exponent = rural_exp[stability_class - 1];
  }

  est_speed = speed * pow(REF_HEIGHT / zspeed, exponent);
  est_speed = max(est_speed, MIN_SPEED);
  return est_speed;
}

/**
 * @brief estimate the stability class
 * @ref EPA-454/5-99-005, 2000, section 6.2.5
 */
int stab_srdt(int daytime, double speed, double solar, double dT) {
  const int lsrdt[6][8] = {
    {1, 1, 2, 4, 0, 5, 6, 0},
    {1, 2, 3, 4, 0, 5, 6, 0},
    {2, 2, 3, 4, 0, 4, 4, 0},
    {3, 3, 4, 4, 0, 0, 0, 0},
    {3, 4, 4, 4, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0}
  };

  int i;
  int j;

  if (daytime) {
    if (solar >= 925.0)
      j = 0;
    else if (solar >= 675.0)
      j = 1;
    else if (solar >= 175.0)
      j = 2;
    else
      j = 3;
    if (speed >= 6.0)
      i = 4;
    else if (speed >= 5.0)
      i = 3;
    else if (speed >= 3.0)
      i = 2;
    else if (speed >= 2.0)
      i = 1;
    else
      i = 0;
  } else {
    if (dT >= 0.0)
      j = 6;
    else
      j = 5;

    if (speed >= 2.5)
      i = 2;
    else if (speed >= 2.0)
      i = 1;
    else
      i = 0;
  }
  return lsrdt[i][j];
}

/*....................................{ wbgt functions }.........................................*/
// double h_cylinder_in_air(
//   double diameter, double length, double temperature, double Pair, double speed
// ) {
//   double density;
//   double Re;  // Reynolds number
//   double Nu;  // Nusselt number

//   density = Pair * 100.0 / (R_AIR * temperature);
//   Re = max(speed, MIN_SPEED) * density * diameter / viscosity(temperature);
//   // parameters from Bedingfield and Drew
//   Nu = 0.281 * pow(Re, (1.0 - 0.4)) * pow(Pr, (1.0 - 0.56));
//   return Nu * thermal_conductivity(temperature) / diameter;
// }

// /**
//  * @brief Calculate the convective heat transfer coefficient, W/(m2 K) for flow around a sphere.
//  * @ref Bird, Stewart, and Lightfoot (BSL), page 409.
//  */
// double h_sphere_in_air(double diameter, double temperature, double Pair, double speed) {
//   double density;
//   double Re;  // Reynolds number
//   double Nu;  // Nusselt number

//   density = Pair * 100. / (R_AIR * temperature);
//   Re = max(speed, MIN_SPEED) * density * diameter / viscosity(temperature);
//   Nu = 2.0 + 0.6 * sqrt(Re) * pow(Pr, 0.3333);
//   return Nu * thermal_conductivity(temperature) / diameter;
// }
/**
 * @brief Calculate the convective heat transfer coefficient, W/(m2 K) for flow around a sphere.
 * @ref Bird, Stewart, and Lightfoot (BSL), page 409. 
 */
double globe_heat_transfer(double temperature, double Pair, double speed) {
  double density;
  double Re;  // Reynolds number
  double Nu;  // Nusselt number

  density = Pair * 100. / (R_AIR * temperature);
  Re = max(speed, MIN_SPEED) * density * GLOBE_DIAMETER / viscosity(temperature);
  Nu = 2.0 + 0.6 * sqrt(Re) * pow(Pr, 0.3333);
  return Nu * thermal_conductivity(temperature) / GLOBE_DIAMETER;
}
double globe_temperature(
  double Pair, double temperature, double rh, double speed, double solar, double fdir, double cza
) {
  int i;
  double Tsfc;
  double Tref;
  double Tglobe_prev;
  double Tglobe_new;
  double h;

  Tsfc = temperature;
  Tglobe_prev = temperature;  // first guess is the air temperature

  for (i = 0; i < MAX_ITER; i++) {
    Tref = 0.5 * (Tglobe_prev + temperature); /* evaluate properties at the average temperature */
    // h = convective_heat_transfer(Pair, Tref, speed, SPHERE);
    h = globe_heat_transfer(Tref, Pair, speed);
    Tglobe_new = pow(
      0.5 * (emissivity(temperature, rh) * pow(temperature, 4.) + EMIS_SFC * pow(Tsfc, 4.)) -
        h / (STEFANB * EMIS_GLOBE) * (Tglobe_prev - temperature) +
        solar / (2. * STEFANB * EMIS_GLOBE) * (1. - ALB_GLOBE) *
          (fdir * (1. / (2. * cza) - 1.) + 1. + ALB_SFC),
      0.25
    );
    if (fabs(Tglobe_new - Tglobe_prev) < CONVERGENCE)
      return Tglobe_new;

    Tglobe_prev = 0.9 * Tglobe_prev + 0.1 * Tglobe_new;
  }

  return NAN;
}
/**
 * @brief the convective heat transfer coefficient in W/(m2 K) for a long cylinder in cross flow
 * @ref Bedingfield and Drew, eqn 32 
 */
double wetbulb_heat_transfer(double pressure, double temperature, double speed) {
  double density;
  double Re;  // Reynolds number
  double Nu;  // Nusselt number

  density = pressure * 100.0 / (R_AIR * temperature);
  Re = max(speed, MIN_SPEED) * density * WICK_DIAMETER / viscosity(temperature);
  Nu = 0.281 * pow(Re, (1.0 - 0.4)) * pow(Pr, (1.0 - 0.56));

  return Nu * thermal_conductivity(temperature) / WICK_DIAMETER;
}

double radiative_heating(double temperature, double emiss, double wick) {
  double x = STEFANB * WICK_EMISSIVITY * (emiss - pow(temperature, 4.));
  x += wick;
  return x;
}

double wetbulb_temperature(
  double pressure,
  double temperature,
  double rh,
  double speed,
  double solar,
  double fdir,
  double cza,
  int rad
) {
  int i;
  double Td;
  double wb0;
  double wb1;
  double sza;
  double Tmean;
  double eair;
  double ewick;  //   saturation vapor pressure at the wet bulb temperature
  double Sc;  //      Schmidt number
  double rht;  //    radiative heating term
  double emiss;  //   atmospheric emissivity
  double swick;  //   solar irradiance absorbed by the wick
  double T = temperature;

  eair = rh * saturation_vapor_pressure(temperature, LIQUID);
  Td = dewpoint(eair, LIQUID);
  emiss = (emissivity(T, rh) * pow(T, 4.0) + SURFACE_EMISSIVITY * pow(T, 4.0)) * 0.5;
  swick = (1. - WICK_ALBEDO) * solar *
    ((1. - fdir) * (1. + 0.25 * WICK_DIAMETER / WICK_LENGTH) +
     fdir * ((tan(acos(cza)) / PI) + 0.25 * WICK_DIAMETER / WICK_LENGTH) + SURFACE_ALBEDO);

  wb0 = Td;  // first guess is the dew point temperature
  for (i = 0; i < MAX_ITER; i++) {
    // evaluate properties at the average temperature
    Tmean = (wb0 + T) * 0.5;
    rht = radiative_heating(wb0, emiss, swick);
    rht /= wetbulb_heat_transfer(pressure, Tmean, speed);
    ewick = saturation_vapor_pressure(wb0, LIQUID);
    Sc = schmidt_number(pressure, Tmean);

    wb1 = T -
      evaporation(Tmean) / RATIO * (ewick - eair) / (pressure - ewick) * pow(Pr / Sc, 0.56) +
      (rht * rad);

    if (isnan(wb1)) {
      return wb0;
    }
    if (fabs(wb1 - wb0) < CONVERGENCE) {
      return wb1;
    }
    wb0 = 0.9 * wb0 + 0.1 * wb1;
  }

  return NAN;
}

int calc_wbgt(
  int year,
  int month,
  int day,
  int hour,
  int minute,
  int gmt,
  int avg,
  double lat,
  double lon,
  double solar,
  double pressure,
  double temperature,
  double relhum,
  double speed,
  double zspeed,
  double dT,
  int urban,
  double* est_speed,
  double* Tg,
  double* Tnwb,
  double* Tpsy,
  double* Twbg
) {
  int stability_class;
  double cza;  // cosine of solar zenith angle
  double fdir;  // fraction of solar irradiance due to direct beam
  // double tk;  // temperature converted to kelvin
  double rh;  // relative humidity, fraction between 0 and 1
  double hour_gmt;
  double dday;

  // convert time to GMT and center in avg period
  hour_gmt = hour - gmt + (minute - 0.5 * avg) / 60.0;
  dday = day + hour_gmt / 24.;

  // calculate the cosine of the solar zenith angle and fraction of solar irradiance
  // due to the direct beam; adjust the solar irradiance if it is out of bounds
  calc_solar_parameters(year, month, dday, lat, lon, &solar, &cza, &fdir);

  if (zspeed != REF_HEIGHT) {
    stability_class = stab_srdt((int)cza > 0., speed, solar, dT);
    *est_speed = est_wind_speed(speed, zspeed, stability_class, urban);
    speed = *est_speed;
  } else {
    *est_speed = speed;
  }

  // unit conversions
  // tk = temperature + T0;  // C -> K
  rh = 0.01 * relhum;  // % -> fraction

  // calculate globe, natural wet bulb, psychrometric wet bulb, & wet bulb globe temperatures
  *Tg = globe_temperature(pressure, temperature, rh, speed, solar, fdir, cza);
  *Tnwb = wetbulb_temperature(pressure, temperature, rh, speed, solar, fdir, cza, 1);
  *Tpsy = wetbulb_temperature(pressure, temperature, rh, speed, solar, fdir, cza, 0);

  if (*Tg == NAN || *Tnwb == NAN) {
    *Twbg = NAN;
    return -1;
  }
  *Twbg = 0.1 * temperature + 0.2 * (*Tg) + 0.7 * (*Tnwb);
  return 0;
}
