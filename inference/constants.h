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


#ifndef CONSTANTS_H_
#define CONSTANTS_H_


namespace constants {
    const int samples = 13;
    const float datascale = 4.;  // Ratio for scale the data
    const int input_shape[4] = {1, 10, 3, 1};
    const int output_shape[2] = {1, 1};
    const float sampling_rate = 2.5;
    const float Dlay = .39; //Delay in s
    const float min_amp = .2;
}

#endif
