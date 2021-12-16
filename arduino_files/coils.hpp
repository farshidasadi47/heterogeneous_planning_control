/*
  This header file contains driver for 3 hbridges that run Helmholtz
  coil.
  The pin connection are based on using Arduino Uno and 
  three Pololu Dual VNH5019 Motor Driver Shield.
  Author: Farshid Asadi, farshidasadi47@yahoo.com
  Decemer 2021.
*/
#ifndef COILS_H
#define COILS_H

#include "Arduino.h"
/********* Type defs **************************************************/
// PIN definitions, 
// H bridge 1, connected to small coil, Z axis, A->B is +Z direction.
static const unsigned char EN1 = 6;
static const unsigned char INA1 = 2;
static const unsigned char INB1 = 4;
static const unsigned char PWM1 = 9;
static const unsigned char CS1 = A0;
static const float m1_correction_factor = 2*1.5/4.2;
// H bridge 2, connected to middle coil, X axis, B->A is +X direction.
static const unsigned char EN2 = 12;
static const unsigned char INA2 = 7;
static const unsigned char INB2 = 8;
static const unsigned char PWM2 = 10;
static const unsigned char CS2 = A1;
static const float m2_correction_factor = 2*2.5/4.2;
// H bridge 3, connected to big coil, Y axis, B->A is +Y direction.
static const unsigned char EN3 = 0;
static const unsigned char INA3 = 3;
static const unsigned char INB3 = 5;
static const unsigned char PWM3 = 11;
static const unsigned char CS3 = A2;
static const float m3_correction_factor = 1.0;
/********* Function declarations **************************************/
// DualVNH5019_Init
// Sets up pin modes for coils drivers.
// Input: None
// Output: None
void DualVNH5019_Init();
// set_mi_P_Q
// Sets voltage of the coil.
// Input: voltage_percentage, voltage applied in terms of max voltage.
// Output: None.
void set_m1_small_z(float voltage_percentage);
void set_m2_middle_x(float voltage_percentage);
void set_m3_big_y(float voltage_percentage);
// set_magnetic_field_cartesian
// Sets magnetic field for a given unit vector 
// and magnetic power percentage.
// Input: ex, ey, ez, magnetic field unit vector coordinates.
//    power_percentage, magnetic field power percentage w.r.t max power.
// Output: None.
void set_magnetic_field_cartesian(float ex, float ey,
                                  float ez, float power_percentage);


#endif // COILS_H