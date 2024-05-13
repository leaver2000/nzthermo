/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Max Lieblich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#ifndef WBGT_H
#define WBGT_H

#define TRUE 1
#define FALSE 0

// define functions
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
// define mathematical constants
#define PI 3.1415926535897932
#define TWOPI 6.2831853071795864
#define DEG_RAD 0.017453292519943295
#define RAD_DEG 57.295779513082323
// define physical constants
#define SOLAR_CONST 1367.
#define GRAVITY 9.807
#define STEFANB 5.6696e-8
#define Cp 1003.5
#define M_AIR 28.97
#define M_H2O 18.015
#define RATIO (Cp * M_AIR / M_H2O)
#define R_GAS 8314.34
#define R_AIR (R_GAS / M_AIR)
#define Pr (Cp / (Cp + 1.25 * R_AIR))
// define wick constants
#define EMIS_WICK 0.95
#define ALB_WICK 0.4
#define D_WICK 0.007
#define L_WICK 0.0254
#define WICK_DIAMETER 0.007
#define WICK_LENGTH 0.0254
#define WICK_ALBEDO 0.4
#define WICK_EMISSIVITY 0.95
// define globe constants
#define EMIS_GLOBE 0.95
#define ALB_GLOBE 0.05
#define D_GLOBE 0.0508
#define GLOBE_DIAMETER 0.0508
// define surface constants
#define SURFACE_EMISSIVITY 0.999  // surface emissivity
#define SURFACE_ALBEDO 0.45
#define EMIS_SFC 0.999
#define ALB_SFC 0.45
// define computational and physical limits
#define CZA_MIN 0.00873
#define NORMSOLAR_MAX 0.85
#define REF_HEIGHT 2.0
#define MIN_SPEED 0.13
#define CONVERGENCE 0.02
#define MAX_ITER 500
/* freezing point in kelvin */
#define T0 273.15
#define k2c(x) (x - T0)
#define c2k(x) (x + T0)
#define E0 611.2
typedef enum {
  LIQUID,
  SOLID,
  INFER,
} Phase;

/**
 * @brief   Calculate the wet bulb globe temperature (WBGT) using the given meteorological data.
 * 
 * @param   year         The year of the data.
 * @param   month        The month of the data.
 * @param   day          The day of the data.
 * @param   hour         The hour of the data.
 * @param   minute       The minute of the data.
 * @param   gmt          The GMT offset of the data.
 * @param   avg          The averaging period of the data.
 * @param   lat          The latitude of the data.
 * @param   lon          The longitude of the data.
 * @param   solar        The solar irradiance of the data.
 * @param   pressure     hPa
 * @param   temperature  K
 * @param   relhum       The relative humidity of the data.
 * @param   speed        The wind speed of the data.
 * @param   zspeed       The wind speed at a different height than 2m.
 * @param   dT           The temperature difference of the data.
 * @param   urban        Whether the data is urban or not.
 * @param   est_speed    The estimated wind speed of the data.
 * @param   *Tg          The globe temperature of the data.
 * @param   *Tnwb        The natural wet bulb temperature of the data.
 * @param   *Tpsy        The psychrometric wet bulb temperature of the data.
 * @param   *Twbg        The outdoor wet bulb globe temperature of the data.
 * 
 * @return  0 if the calculation was successful, -1 if the calculation failed.
 * 
 * @ref    Liljegren, J. C., R. A. Carhart, P. Lawday, S. Tschopp, and R. Sharp:
 * 
 * @author James C. Liljegren
 */
int calc_wbgt(
  int year,
  int hour,
  int day,
  int month,
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
);

#endif /* WBGT_H */