#include "coils.hpp"
/********* Function definitions ***************************************/
void coils::DualVNH5019_Init(){
    // Sets up pin modes for coils drivers.
    // Input: none
    // Output: none
    pinMode(EN1,INPUT);
    pinMode(INA1,OUTPUT);
    pinMode(INB1,OUTPUT);
    pinMode(PWM1,OUTPUT);
    pinMode(CS1,INPUT);
  
    pinMode(INA2,OUTPUT);
    pinMode(INB2,OUTPUT);
    pinMode(PWM2,OUTPUT);
    pinMode(EN2,INPUT);
    pinMode(CS2,INPUT);
    
    pinMode(INA3,OUTPUT);
    pinMode(INB3,OUTPUT);
    pinMode(PWM3,OUTPUT);
    pinMode(EN3,INPUT);
    pinMode(CS3,INPUT);

    return;
}

void set_m1_small_z(float voltage_percentage){
    // Sets voltage of the small coil or Z axis.
    // Input: voltage_percentage, voltage applied in terms of max voltage.
    // Output: None.
    // A->B is +Z direction.

    // Correct the scaling the the field w.r.t other coils.
    voltage_percentage *= m1_correction_factor;
    // Set the current direction.
    if (voltage_percentage>0){
        // Flow from A to B.
        digitalWrite(INA1,HIGH);
        digitalWrite(INB1,LOW);
    }else if (voltage_percentage<0){
        // Flow from B to A.
        digitalWrite(INA1,LOW);
        digitalWrite(INB1,HIGH);
    }else{
        // No flow.
        digitalWrite(INA1,LOW);
        digitalWrite(INB1,LOW);
    }
    // Set the PWM value of driver.
    analogWrite(PWM1, round(abs(voltage_percentage)*255/100));

    return;
}

void set_m2_middle_x(float voltage_percentage){
    // Sets voltage of the middle coil or X axis.
    // Input: voltage_percentage, voltage applied in terms of max voltage.
    // Output: None.
    // B->A is +X direction.

    // Correct the scaling the the field w.r.t other coils.
    voltage_percentage *= m2_correction_factor;
    // Set the current direction.
    if (voltage_percentage>0){
        // Flow from B to A.
        digitalWrite(INA2,LOW);
        digitalWrite(INB2,HIGH);
    }else if (voltage_percentage<0){
        // Flow from A to B.
        digitalWrite(INA2,HIGH);
        digitalWrite(INB2,LOW);
    }else{
        // No flow.
        digitalWrite(INA2,LOW);
        digitalWrite(INB2,LOW);
    }
    // Set the PWM value of driver.
    analogWrite(PWM2, round(abs(voltage_percentage)*255/100));

    return;
}

void set_m3_big_y(float voltage_percentage){
    // Sets voltage of the big coil or Y axis.
    // Input: voltage_percentage, voltage applied in terms of max voltage.
    // Output: None.
    // B->A is +Y direction.

    // Correct the scaling the the field w.r.t other coils.
    voltage_percentage *= m3_correction_factor;
    // Set the current direction.
    if (voltage_percentage>0){
        // Flow from B to A.
        digitalWrite(INA3,LOW);
        digitalWrite(INB3,HIGH);
    }else if (voltage_percentage<0){
        // Flow from A to B.
        digitalWrite(INA3,HIGH);
        digitalWrite(INB3,LOW);
    }else{
        // No flow.
        digitalWrite(INA3,LOW);
        digitalWrite(INB3,LOW);
    }
    // Set the PWM value of driver.
    analogWrite(PWM3, round(abs(voltage_percentage)*255/100));

    return;
}

void set_magnetic_field_cartesian(float ex, float ey,
                                 float ez, float power_percentage){
    // Sets magnetic field for a given unit vector 
    // and magnetic power percentage.
    // Input: ex, ey, ez, magnetic field unit vector coordinates.
    //    power_percentage, magnetic field power percentage w.r.t max power.
    // Output: None.
    set_m1_small_z(ez*power_percentage);
    set_m2_middle_x(ex*power_percentage);
    set_m3_big_y(ey,power_percentage);

    return;
}

