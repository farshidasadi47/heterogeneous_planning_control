/***********************************************************************
* This file cintains main arduino program for running helmholtz coils.
* It communicates with computer via ROS2 arduibo interface.
* Author: Farshid Asadi, farshidasadi47@yahoo.com
/********* Includes ***************************************************/
#include <pyduino_bridge.h>
#include "coils.hpp"

/********* Globals ****************************************************/
int incomingByte = 0;  // For incoming serial data.
const int baud = 500000;

/********* Initializing ***********************************************/
void setup(){
    // Initialize.
    Serial.begin(baud); // opens serial port, sets data rate to 500000 bps.
}
/********* Main loop **************************************************/
void loop() {
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
    incomingByte = Serial.read();

    // say what you got:
    Serial.print("I received: ");
    Serial.println(incomingByte, DEC);
  }
}
