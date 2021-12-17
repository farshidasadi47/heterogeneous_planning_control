/*
  This header file contains class and methods for our serial
  communication protocol.
  The protocols should match those in "arduinoserial.py" module.
  
  Author: Farshid Asadi, farshidasadi47@yahoo.com
  Decemer 2021.
*/
#ifndef ARDUINOSERIAL_H
#define ARDUINOSERIAL_H

#include "Arduino.h"
/********* Type defs **************************************************/

/********* Function declarations **************************************/
// some_func
// Description.
// Input: None
// Output: None
class Arduino{
    // Establishes serial connection with our simple protocol.
    public:
    // Instance variables.
    unsigned char delimiter_[4] = {0x41,0x42,0x43,0x44};// {0x7f,0xff,0xff,0xff};//
    unsigned long int baud_;
    // Constructor and deconstructor
    Arduino()                      : baud_(115200){}
    Arduino(unsigned long int baud): baud_(baud)  {}
    ~Arduino(){};
    // Methods
};



#endif // ARDUINOSERIAL_H
