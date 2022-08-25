/*
    This header file contains driver for 3 hbridges that run Helmholtz
    coil.
    The pin connection are Arduino Uno compatible.
    The "Coil" class is designed specifically to work with the following
    drivers connected to the specified pins.
    1- Pins 2,3 to Roboteq sdc2160s at 30.6 Volts.
    2- Pins 4,5 to Roboteq sdc2160s at 43.05 Volts.
    3- Pins 8,9,10 to Pololu VNH5019 Motor Driver 24v24 at 20.2 Volts.
    Positive directions are considerred from terminal A to B of drivers.
    Author: Farshid Asadi, farshidasadi47@yahoo.com
    August 2022.
*/
#ifndef COILS_H
#define COILS_H

#include <Arduino.h>
/********* Pin defs ***************************************************/
// Pin definitions.
// Positive direction is
// H bridge 1, middle coil, X axis, A->B is +X direction.
// Roboteq sdc1260s, runs at 30.6 Volts.
const unsigned char DIR1 = 2;
const unsigned char PWM1 = 3;
const float m1_correction_factor = 1.0;
// H bridge 2, big coil, Y axis, A->B is +Y direction.
// Roboteq sdc1260s, runs at 43.05 Volts.
const unsigned char DIR2 = 4;
const unsigned char PWM2 = 5;
const float m2_correction_factor = 1.0;
// H bridge 3, small coil, Z axis, A->B is +Z direction.
// Pololu VNH5019 Motor Driver 24v24 (single), runs at 20.2 Volts.
const unsigned char INA3 = 8;
const unsigned char INB3 = 9;
const unsigned char PWM3 = 10;
const float m3_correction_factor = 0.88;
//
const float general_correction_factor = 1.0;
/********* Class defs *************************************************/
class Coils
{
public:
    void initialize()
    {
        // Sets up pin modes for coils drivers.
        // Input: none
        // Output: none
        pinMode(DIR1,OUTPUT);
        pinMode(PWM1,OUTPUT);

        pinMode(DIR2,OUTPUT);
        pinMode(PWM2,OUTPUT);
        
        pinMode(INA3,OUTPUT);
        pinMode(INB3,OUTPUT);
        pinMode(PWM3,OUTPUT);

        return;
    }

    void set_magnetic_field(int X, int Y, int Z)
    {
        // Sets magnetic field for a given percentage of each coil.
        // Overloaded function V1/2.
        // Input: X, Y, Z power percentages.
        // Output: None.
        // Apply the voltages.
        set_coil_2pin<DIR1, PWM1>      (X*general_correction_factor);
        set_coil_2pin<DIR2, PWM2>      (Y*general_correction_factor);
        set_coil_3pin<INA3, INB3, PWM3>(Z*general_correction_factor);

        return;
    }

    void set_magnetic_field(float theta, float alpha, float power_percentage)
    {
        // Sets magnetic field for a given unit vector 
        // and magnetic power percentage.
        // Overloaded function V2/2.
        // Input: theta, alpha. Angles of magnetic field in degrees.
        //        power_percentage, magnetic field power percentage
        //        w.r.t max power.
        // Output: None.

        // Convert degrees to Radians.
        theta = theta*PI/180;
        alpha = alpha*PI/180;
        // Convert Spherical to cartesian, considering magnetic correction.
        // x coordinate, m2, middle coil, +x when corrent flows A to B.
        float ex = power_percentage*cos(alpha)*cos(theta)*m1_correction_factor;
        // y coordinate, m3, big coil,    +y when current flows A to B.
        float ey = power_percentage*cos(alpha)*sin(theta)*m2_correction_factor;
        // z coordinate, m1, small coil,  +z when current flows A to B.
        float ez = power_percentage*sin(alpha)*m3_correction_factor;
        // Apply the voltages.
        set_coil_2pin<DIR1, PWM1>      (ex*general_correction_factor);
        set_coil_2pin<DIR2, PWM2>      (ey*general_correction_factor);
        set_coil_3pin<INA3, INB3, PWM3>(ez*general_correction_factor);

        return;
    }

private:
    float pwm_sat = 99;
    template <unsigned char INA, unsigned char INB, unsigned char PWMX>
    void set_coil_3pin(float voltage_percentage)
    {
        // Sets voltage of the given coil with 3 pin driver.
        // Input: voltage_percentage, voltage applied in terms of max voltage.
        // Output: None.
        float deadband = .01;
        // Set the current direction.
        if (voltage_percentage>deadband)
        {
            // Saturate percentage.
            if (voltage_percentage>pwm_sat){voltage_percentage=pwm_sat;}
            // Flow from A to B.
            digitalWrite(INA,HIGH);
            digitalWrite(INB,LOW);
        }
        else if (voltage_percentage<-deadband)
        {
            // Saturate percentage.
            if (voltage_percentage<(-pwm_sat)){voltage_percentage=(-pwm_sat);}
            // Flow from B to A.
            digitalWrite(INA,LOW);
            digitalWrite(INB,HIGH);
        }
        else
        {
            // No flow.
            digitalWrite(INA,LOW);
            digitalWrite(INB,LOW);
        }
        // Set the PWM value of driver.
        analogWrite(PWMX, round(abs(voltage_percentage)*255/100));

        return;
    }

    template <unsigned char DIR, unsigned char PWMX>
    void set_coil_2pin(float voltage_percentage)
    {
        // Sets voltage of the given coil.
        // Input: voltage_percentage, voltage applied in terms of max voltage.
        // Output: None.
        float deadband = .01;
        // Set the current direction.
        if (voltage_percentage>deadband)
        {
            // Saturate percentage.
            if (voltage_percentage>pwm_sat){voltage_percentage=pwm_sat;}
            // Flow from A to B.
            digitalWrite(DIR,LOW);
        }
        else if (voltage_percentage<-deadband)
        {
            // Saturate percentage.
            if (voltage_percentage<(-pwm_sat)){voltage_percentage=(-pwm_sat);}
            // Flow from B to A.
            digitalWrite(DIR,HIGH);
        }
        else
        {
            // No flow. Default state.
            digitalWrite(DIR,LOW);
        }
        // Set the PWM value of driver.
        analogWrite(PWMX, round(abs(voltage_percentage)*255/100));

        return;
    }

};

#endif // COILS_H
