/*

Run on Arduino Nano 33 BLE Sense.

LED on when walking
Switch the red LED on when running


Mbed Os version: 1.1.6
Arduino_TensorFlowLite: 1.15.0-ALPHA-precompiled
Arduino_LSM9DS1: 1.1.0

author: nakmuaycoder
date: 01/21
*/

#include "output.h"
#include "Arduino.h"

void HandleOutput(float output_value, bool off){
  const int ledRed = 22;
  const int ledBlue = 24;
  pinMode(ledRed, OUTPUT);
  pinMode(ledBlue, OUTPUT);

  if(off == false){
    if(output_value > 0.5f){
      digitalWrite(ledRed, LOW);
      digitalWrite(ledBlue, HIGH);
    }
    else{
      digitalWrite(ledBlue, LOW);
      digitalWrite(ledRed, HIGH);
    }
  }else{
    digitalWrite(ledBlue, HIGH);
    digitalWrite(ledRed, HIGH);
  }
}


void Error(){
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);
  while(1);
}
