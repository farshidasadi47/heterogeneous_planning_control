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
#include "HardwareSerial.h"
/********* Type defs and globals **************************************/
extern HardwareSerial Serial;
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
    HardwareSerial *io;
    // Constructor and deconstructor
    Arduino()                      : baud_(115200), io(&Serial){}
    Arduino(unsigned long int baud): baud_(baud)  , io(&Serial){}
    ~Arduino(){};
    // Methods
    void begin(){
      // Starts serial communication.
      io->begin(baud_);
    }
    void end(){
      // Finishes serial communication.
      io->end();
    }

    

};



#endif // ARDUINOSERIAL_H
