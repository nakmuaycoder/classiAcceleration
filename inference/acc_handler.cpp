/*


Run on Arduino Nano 33 BLE Sense.

Mbed Os version: 1.1.6
Arduino_TensorFlowLite: 1.15.0-ALPHA-precompiled
Arduino_LSM9DS1: 1.1.0

author: nakmuaycoder
date: 01/21
*/

#include "constants.h"
#include "acc_handler.h"
#include <Arduino_LSM9DS1.h>
#include <Arduino.h>


TfLiteStatus SetupAccelerometer() {
  if (!IMU.begin()){
    return kTfLiteError;
  }else{
    IMU.end();
    return kTfLiteOk;}
  }

void SampleAcceleration(float *val, float wait){
  int i = 0;
  float x, y, z;
  IMU.begin();
  while(i < constants::samples * 3){
    while(!IMU.accelerationAvailable());  // Wait Accelerometer
      IMU.readAcceleration(x, y, z);  // Collect acceleration
      *(val + i) = x;
      *(val + i++) = y;
      *(val + i++) = z;
      i++;
      delay(wait);  // delay decrease sampling rate
  }
  IMU.end();
}

void SetTensor(float *val, TfLiteTensor *model_input){
  static int i = 0;  // Number of the batch
  int j = 0;  //
  while(j < (constants::input_shape[1] * 3)){
    model_input->data.f[j] = *(val + i + j)/constants::datascale;
    j++;
  }
  i += 3; //  Go to the next observation
  i %= (constants::samples * 3);  // if last batch, reset i
}

bool IsMoving(float *val){
  float mx = 0, mn = 0;
  float acc;  // Norm of the acceleration
  for(int j=0; j< constants::samples; j++){
    acc = sqrt(sq(*(val + 3*j)) + sq(*(val + 3*j + 1)) + sq(*(val + 3*j + 2)));
    mn = min(mn, acc);
    mx = max(mx, acc);
  }
  return mx - mn > constants::min_amp;
}


float SampleRate(int sample_size, float Dlay){
  float x, y, z;
  float X[3];
  float t;
  t = micros();
  IMU.begin();
  for(int i=0; i<sample_size; i++){
    while(!IMU.accelerationAvailable());
    IMU.readAcceleration(x, y, z);
    X[0] = x;
    X[1] = y;
    X[2] = z;
    delay(Dlay);
  }
  t = (micros() - t)/pow(10,6);  //time en s
  IMU.end();
  return sample_size/t;
}
