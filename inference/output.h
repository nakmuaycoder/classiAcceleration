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


#ifndef OUTPUT_H_
#define  OUTPUT_H_

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"


extern void HandleOutput(float output_value, bool off=false);
extern void Error();

#endif
