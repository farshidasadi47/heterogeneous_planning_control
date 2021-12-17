/***********************************************************************
* This file cintains main arduino program for running helmholtz coils.
* It communicates with computer via ROS2 arduibo interface.
* Author: Farshid Asadi, farshidasadi47@yahoo.com
/********* Includes ***************************************************/
#include "arduinoserial.h"
#include "coils.h"

/********* Globals ****************************************************/
const unsigned long int baud = 115200;
Arduino arduino;
char str[64];

/********* Initializing ***********************************************/
void setup(){
    // Initialize.
    Serial.begin(baud); // opens serial port, sets data rate to 500000 bps.
}
/********* Main loop **************************************************/
void loop() {
  // send data only when you receive data:
  sprintf(str,"%4s",arduino.delimiter_);
  Serial.println(str);
  sprintf(str,"% 7lu",arduino.baud_);
  Serial.println(str);
  delay(1000);
}
