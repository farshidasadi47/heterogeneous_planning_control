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
const unsigned char EN1 = 6;
const unsigned char INA1 = 2;
const unsigned char INB1 = 4;
const unsigned char PWM1 = 9;
const unsigned char CS1 = A0;
const float m1_correction_factor = 1.5/2.5;
// H bridge 2, connected to middle coil, X axis, B->A is +X direction.
const unsigned char EN2 = 12;
const unsigned char INA2 = 7;
const unsigned char INB2 = 8;
const unsigned char PWM2 = 10;
const unsigned char CS2 = A1;
const float m2_correction_factor = 1;
// H bridge 3, connected to big coil, Y axis, B->A is +Y direction.
const unsigned char EN3 = 0;
const unsigned char INA3 = 3;
const unsigned char INB3 = 5;
const unsigned char PWM3 = 11;
const unsigned char CS3 = A2;
const float m3_correction_factor = 4.2/(2*2.5);

/********* Function declarations **************************************/
// DualVNH5019_Init
// Sets up pin modes for coils drivers.
// Input: None
// Output: None
void DualVNH5019_Init();
// set_coil
// Sets voltage of the given coil.
// Input: voltage_percentage, voltage applied in terms of max voltage.
//        Positive voltage flows from INA to INB.
// Output: None.
template <unsigned char INA, unsigned char INB, unsigned char PWMX>
void set_coil(float voltage_percentage);
// Sets magnetic field for a given unit vector 
// and magnetic power percentage.
// Input: theta, alpha. Angles of magnetic field in degrees.
//        power_percentage, magnetic field power percentage
//        w.r.t max power.
// Output: None.
void set_magnetic_field(float theta, float alpha, float power_percentage);

#endif // COILS_H