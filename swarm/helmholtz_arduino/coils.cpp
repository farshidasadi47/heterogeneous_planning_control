#include "coils.h"
/********* Function definitions ***************************************/
void DualVNH5019_Init(){
    // Sets up pin modes for coils drivers.
    // Input: none
    // Output: none
    pinMode(INA1,OUTPUT);
    pinMode(INB1,OUTPUT);
    pinMode(PWM1,OUTPUT);
    pinMode(CS1,INPUT);
  
    pinMode(INA2,OUTPUT);
    pinMode(INB2,OUTPUT);
    pinMode(PWM2,OUTPUT);
    pinMode(CS2,INPUT);
    
    pinMode(INA3,OUTPUT);
    pinMode(INB3,OUTPUT);
    pinMode(PWM3,OUTPUT);
    pinMode(CS3,INPUT);

    return;
}

template <unsigned char INA, unsigned char INB, unsigned char PWMX>
void set_coil(float voltage_percentage){
    // Sets voltage of the given coil.
    // Input: voltage_percentage, voltage applied in terms of max voltage.
    // Output: None.

    // Set the current direction.
    if (voltage_percentage>0){
        // Saturate percentage.
        if (voltage_percentage>100){voltage_percentage=100;}
        // Flow from A to B.
        digitalWrite(INA,HIGH);
        digitalWrite(INB,LOW);
    }else if (voltage_percentage<0){
        // Saturate percentage.
        if (voltage_percentage<(-100)){voltage_percentage=(-100);}
        // Flow from B to A.
        digitalWrite(INA,LOW);
        digitalWrite(INB,HIGH);
    }else{
        // No flow.
        digitalWrite(INA,LOW);
        digitalWrite(INB,LOW);
    }
    // Set the PWM value of driver.
    analogWrite(PWMX, round(abs(voltage_percentage)*255/100));

    return;
}

void set_magnetic_field(float theta, float alpha, float power_percentage){
    // Sets magnetic field for a given unit vector 
    // and magnetic power percentage.
    // Input: theta, alpha. Angles of magnetic field in degrees.
    //        power_percentage, magnetic field power percentage
    //        w.r.t max power.
    // Output: None.

    // Convert degrees to Radians.
    theta = theta*PI/180;
    alpha = alpha*PI/180;
    // Convert Spherical to cartesian, considering magnetic correction.
    // x coordinate, m2, middle coil, +x when corrent flows B to A.
    float ex = power_percentage*cos(alpha)*cos(theta)*m2_correction_factor;
    // y coordinate, m3, big coil,    +y when current flows B to A.
    float ey = power_percentage*cos(alpha)*sin(theta)*m3_correction_factor;
    // z coordinate, m1, small coil,  +z when current flows A to B.
    float ez = power_percentage*sin(alpha)*m1_correction_factor;
    // Apply the voltages.
    set_coil<INA1, INB1, PWM1>(ez);
    set_coil<INB2, INA2, PWM2>(ex);
    set_coil<INB3, INA3, PWM3>(ey);

    return;
}
