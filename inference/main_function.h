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


#ifndef MAIN_FUNCTIONS_H_
#define MAIN_FUNCTIONS_H_

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void setup();


void loop();

#ifdef __cplusplus
}
#endif

#endif
