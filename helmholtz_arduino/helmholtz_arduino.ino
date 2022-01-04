/***********************************************************************
* This file cintains main arduino program for running helmholtz coils.
* It communicates with computer via ROS2 arduibo interface.
* Author: Farshid Asadi, farshidasadi47@yahoo.com
/********* Includes ***************************************************/
#include "arduinoserial.h"
#include "coils.h"
#include "Arduino.h"

/********* Globals ****************************************************/
const unsigned long int baud = 115200;
Arduino<250> arduino;
Buffer<int,4> buffer4;


/********* Initializing ***********************************************/
void setup(){
    // Initialize.
    //Serial.begin(baud); // opens serial port, sets data rate to 500000 bps.
    arduino.begin();
}
/********* Main loop **************************************************/
void loop() {
  // send data only when you receive data:
  float a[3]{-1256.84,3256.98,-1239.85};
  arduino.write<float>(a,3);
  
  delay(1000);
}
