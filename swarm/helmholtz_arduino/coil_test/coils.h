/*
    This header file contains driver for 3 hbridges that run Helmholtz
    coil.
    The pin connection are Arduino Uno compatible.
    The "Coil" class is designed specifically to work with the following
    drivers connected to the specified pins.
    1- Pins 2,3 to Roboteq sdc2160s at 34 Volts.
    2- Pins 4,5 to Roboteq sdc2160s at 48 Volts.
    3- Pins 8,9,10 to Pololu VNH5019 Motor Driver 24v24 at 22 Volts.
    Positive directions are considerred from terminal A to B of drivers.
    Author: Farshid Asadi, farshidasadi47@yahoo.com
    August 2022.
*/
#ifndef COILS_H
#define COILS_H

#include <Arduino.h>
#include <Regexp.h>
/********* Pin defs ***************************************************/
// Pin definitions.
// Positive direction is
// H bridge 1, middle coil, X axis, A->B is +X direction.
// Roboteq sdc1260s, runs at 34 Volts.
const unsigned char DIR1 = 2;
const unsigned char PWM1 = 3;
const float m1_correction_factor = 1.0;
// H bridge 2, big coil, Y axis, A->B is +Y direction.
// Roboteq sdc1260s, runs at 48 Volts.
const unsigned char DIR2 = 4;
const unsigned char PWM2 = 5;
const float m2_correction_factor = 1.0;
// H bridge 3, small coil, Z axis, A->B is +Z direction.
// Pololu VNH5019 Motor Driver 24v24 (single), runs at 22 Volts.
const unsigned char INA3 = 8;
const unsigned char INB3 = 9;
const unsigned char PWM3 = 10;
const float m3_correction_factor = 0.86;
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

class ProcessInput
{
public:
    ProcessInput(Coils& coil) : m_coil(coil){}
    
    void get_command_n_set_coil()
    {   // Reads user input and sets coils accordingly.
        // Input: None
        // Output: None
        static bool refresh_screen{true};
        char str[64]{0};
        int written{0};
        bool matched{false};
        bool new_data = readLine();
        //
        //clearScreen();
        if (refresh_screen)
        {
            if (m_mode == 0)
            {
                percentageIntroMsg();
            }
            else
            {
                polarIntroMsg();
            }
            refresh_screen= false;
        }
        //
        if (new_data)
        {   // If this is new data.
            bool mode_changed = isModeChanged();
            if (mode_changed) {refresh_screen= true;}
            if (!mode_changed)
            {   // If this is not mode change request.
                matched = match_n_convert();
                if (matched)
                {   
                    if (matched) {refresh_screen= true;}
                    if (m_mode == 0)
                    {   // This is coil raw power percentage.
                        m_coil.set_magnetic_field(m_cmd[0], m_cmd[1], m_cmd[2]);
                    }
                    else
                    {   // This is polar coordinate command mode.
                        m_coil.set_magnetic_field((float) m_cmd[0],
                                                (float) m_cmd[1],
                                                (float) m_cmd[2]);
                    }
                }
            }
        }
        
    }

private:
    static const unsigned m_buffer_len = 64;
    char m_msg[m_buffer_len]{0};
    int *m_inputs;
    int m_mode{0};  // 0: Coils raw power percenrage, 1: Polar input.
    int m_cmd[3]{0,0,0};
    Coils& m_coil;

    void clearScreen()
    {
        Serial.write(27);       // ESC command
        Serial.print("[2J");    // clear screen command
        Serial.write(27);
        Serial.print("[H");     // cursor to home command
    }

     void percentageIntroMsg()
    {   
        int written = 0;
        char str[73]{0};
        written += sprintf(&str[written], "Current command: ");
        for(int i = 0; i < 3; i++)
        {
            written += sprintf(&str[written],"Coil %d:%4d%%, ",i+1,m_cmd[i]);
        }
        str[written] = '\0';
        Serial.println(str);
        Serial.println("Please enter coils raw power percentage, separated by comma and/or space.");
        Serial.println("For example: 1,2,3  or  -1 -2 -3");
        Serial.println("Enter \"change\" to change input mode to polar.");
        for (int i = 0; i<72; i++){str[i] = '*';}
        str[72]= '\0';
        Serial.println(str);
        return;
    }

    void polarIntroMsg()
    {   
        int written = 0;
        char str[73]{0};
        written += sprintf(&str[written], "Current command: ");
        written+= sprintf(&str[written],"Theta:%4d deg, ",m_cmd[0]);
        written+= sprintf(&str[written],"Alpha:%4d deg, ",m_cmd[1]);
        written+=sprintf(&str[written],"Power %%:%4d%%",m_cmd[2]);
        Serial.println(str);
        Serial.println("Please integer array\"theta(deg), alpha(deg), power %%\", separated by comma and/or space.");
        Serial.println("For example: 45,-90,75  or  -50 85 100");
        Serial.println("Enter \"change\" to change input mode to coil raw power percentage.");
        for (int i = 0; i<72; i++){str[i] = '*';}
        str[72]= '\0';
        Serial.println(str);
        return;
    }
    
    bool readLine()
    {   // Reads serial buffer until it faces new line "\n" character.
        // Input: None
        // Output: bool new_data, indicating if it read a new data.
        bool new_data = false;
        unsigned char cnt = 0;
        int incoming_byte;
        if (Serial.available())
        {
            while (cnt < m_buffer_len - 1)
            {
                incoming_byte = Serial.read();
                if (incoming_byte == '\n' ){break;}
                new_data = true;
                if (incoming_byte != -1)
                {
                    m_msg[cnt] = incoming_byte;
                    cnt++;
                }
            } // End of while loop.
            if (new_data){m_msg[cnt] = '\0';}
        }
    return new_data;
    }

    bool isModeChanged()
    {   // Detects if input mode change is requested and changes mode.
        // Input: None
        // Output: bool mode_changed, indicating if mode is changed.
        bool mode_changed = false;
        MatchState ms (m_msg);
        ms.Target(m_msg);
        
        char change = ms.Match("Change");
        char change2 = ms.Match("change");
        if (change == REGEXP_MATCHED || change2 == REGEXP_MATCHED)
        {
            m_mode = m_mode ? 0 : 1;
            mode_changed = true;
            m_cmd[0] = 0;
            m_cmd[1] = 0;
            m_cmd[2] = 0;
        }
        return mode_changed;
    }

    bool match_n_convert()
    {   // Checks if the user input matches the desired pattern.
        // If matched updates m_command.
        // -1, 2, 3 or 2 3 8 is the desired pattern.
        // Input: None
        // Output: bool, matched
        const char *delimiter = ", "; //Delimiter is "," or "(space)"
        char *token;
        int i = 0;
        MatchState ms (m_msg);
        ms.Target(m_msg);
        bool matched = false;
        char result = ms.Match("[+-]?[%d]+[, ]+ *[+-]?[%d]+[, ]+ *[+-]?[%d]+");
        if (result == REGEXP_MATCHED)
        {
            matched = true;
            // Convert matched pattern to array.
            token = strtok(m_msg, delimiter);
            while(token != NULL)
            {
                m_cmd[i] = atoi(token);
                token = strtok(NULL,delimiter);
                i++;
            }
        }
        else
        {
            Serial.println("Input invalid, ignored.");
            if (m_mode == 0)
            {
                percentageIntroMsg();
            }
            else
            {
                polarIntroMsg();
            }
        }
        return matched;
    }

};

#endif // COILS_H
