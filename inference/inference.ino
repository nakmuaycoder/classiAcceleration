/*

Run on Arduino Nano 33 BLE Sense.

LED on when walking
Switch the red LED on when running


Model input shape : (1, 10, 3, 1) 
Model outpput shape : (1, 1) 

Mbed Os version: 1.1.6
Arduino_TensorFlowLite: 1.15.0-ALPHA-precompiled
Arduino_LSM9DS1: 1.1.0

author: nakmuaycoder
date: 01/21
*/

#define DEBUG 1

// Dependancies
#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "main_function.h"
#include "model.h"
#include "acc_handler.h"
#include "output.h"
#include "constants.h"

#if DEBUG
  #include "arrays_test.h"
#endif

namespace {
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *model_input = nullptr;
  TfLiteTensor *model_output = nullptr;
  constexpr int tensorArenaSize = 100 * 1024;
  uint8_t tensorArena[tensorArenaSize];
  float Dlay = 0, s = 0;
}

void setup(){

  TfLiteStatus status;

  #if DEBUG
    Serial.begin(115200);
    while (!Serial);
    Serial.println("==================\n||Starting Debug||\n==================\n");
  #endif

  // switch LEDs OFF
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(mdl);
  #if DEBUG
    Serial.println("Model loaded");
  #endif
  
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    #if DEBUG
      Serial.println("Model schema mismatch!");
    #endif
    Error();
    }
  
  #if DEBUG
    Serial.println("Model ok");
  #endif
  

  static tflite::MicroMutableOpResolver micro_op_resolver;
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D, tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D, tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED, tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_LOGISTIC, tflite::ops::micro::Register_LOGISTIC());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RELU, tflite::ops::micro::Register_RELU());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE, tflite::ops::micro::Register_RESHAPE());

  #if DEBUG
    Serial.println("TensorFlow operations loaded!");
  #endif

  // Create interpreter
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensorArena, tensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  #if DEBUG
    Serial.println("Interpreter created");
  #endif

  status = interpreter->AllocateTensors();
  
  #if DEBUG
    if(status == kTfLiteOk){Serial.println("Interpreter allocation ok");
    }else{
      Serial.println("Interpreter allocation ko");
      Error();}
  #endif
  
  if(status != kTfLiteOk){
    Error();
  }

  
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Test input tensor
  if(model_input->dims->size != ARRAY_SIZE(constants::input_shape) || // len(input.shape)
  model_input->dims->data[0] != constants::input_shape[0] || // model batch_size
  model_input->dims->data[1] != constants::input_shape[1] || // number of points
  model_input->dims->data[2] != constants::input_shape[2] || // # canal
  model_input->dims->data[3] != constants::input_shape[3] ||
  model_input->type != kTfLiteFloat32
  ){
    
    #if DEBUG
      Serial.println("Invalid Input");
      Serial.println("Length of the input shape: Expected: " + String(ARRAY_SIZE(constants::input_shape)) + " Got: " + String(model_input->dims->size));
      Serial.println("Batch size: Expected: " + String(constants::input_shape[0]) + " Got: " + String(model_input->dims->data[0]));
      Serial.println("Sequence length: Expected:" + String(constants::input_shape[1]) + " Got: " + String( model_input->dims->data[1]));
      Serial.println("Cannal Expected:" + String(constants::input_shape[2]) + " Got: " + String(model_input->dims->data[2]));
      Serial.println("Last dim Expected:"); Serial.print(constants::input_shape[3]); Serial.print(" Got: "); Serial.println(model_input->dims->data[3]);
      Serial.println("Input data type: Expected: " + String(kTfLiteFloat32) + " Got " + String(model_input->type));
    #endif
    Error();
  }

  // Test output tensor
  if(model_output->dims->size != ARRAY_SIZE(constants::output_shape) || // len(input.shape)
  model_output->dims->data[0] != constants::output_shape[0] || // model batch_size
  model_output->dims->data[1] != constants::output_shape[1] || // # canal
  model_output->type != kTfLiteFloat32
  ){
    #if DEBUG
      Serial.println("Invalid Ouput");
      Serial.println("Length of the output shape: Expected: " + String(ARRAY_SIZE(constants::output_shape)) + " Got " + String(model_output->dims->size));
      Serial.println("Batch size: Expected: " + String(constants::output_shape[0]) + " Got " + String(model_output->dims->data[0]));
      Serial.println("Sequence length: Expected: " + String(constants::output_shape[1]) + " Got " + String(model_output->dims->data[1]));
    #endif
    Error();
  }
  
  // Test accelerometer
  status = SetupAccelerometer();
  if (status != kTfLiteOk) {
    #if DEBUG
      Serial.println("Fail to Setup IMU!");
    #endif
    Error();
  }

  #if DEBUG
    s = SampleRate(100, 0.);
    Serial.println("Sampling rate: " + String(s) + "Hz. Expected Sampling rate: " + String(constants::sampling_rate) + "Hz.");
    Serial.println("Dlay must be set to: " + String(1/constants::sampling_rate - 1/s) + "s");
  #endif
}

void loop(){

  TfLiteStatus status;
  int i = 0;
  float input_values[constants::samples * 3]; // Array acceleration
  float output = 0;

  #if DEBUG
    static int l = 0;

    if(l == 2){
      if(ARRAY_SIZE(nothing) != constants::samples * 3){
        Serial.println("Expected array size for nothing: " + String(constants::samples * 3) + " Got " + String(ARRAY_SIZE(nothing)));
        Error();
      }
      for(int z=0; z< constants::samples * 3; z++){
        input_values[z] = nothing[z];
      }
      Serial.println("Run with array nothing");
    }else if (l == 1){
      if(ARRAY_SIZE(walk) != constants::samples * 3){
        Serial.println("Expected array size for walk: " + String(constants::samples * 3) + " Got " + String(ARRAY_SIZE(walk)));
        Error();
      }
      for(int z=0; z< constants::samples * 3; z++){
        input_values[z] = walk[z];
      }
      Serial.println("Run with array walk");
    }else if (l == 0){
      if(ARRAY_SIZE(run) != constants::samples * 3){
        Serial.println("Expected array size for run: " + String(constants::samples * 3) + " Got " + String(ARRAY_SIZE(run)));
        Error();
      }
      for(int z=0; z< constants::samples * 3; z++){
        input_values[z] = run[z];
      }
      Serial.println("Run with array run");
    }else{Serial.println("Run inference on accelerometer data");
    SampleAcceleration(input_values, constants::Dlay); // Record accelerations
    Serial.println("acc sampled");
    }
    l++;
  #else
    SampleAcceleration(input_values, constants::Dlay); // Record accelerations
    Serial.println("acc sampled");
  #endif

    while(!IsMoving(input_values)){
    // Go out of this loop if detect any motion
    delay(3000);  // Wait 3s
    SampleAcceleration(input_values, constants::Dlay); // Record accelerations
    HandleOutput(output, true);

    #if DEBUG
      Serial.println("Resample acceleration");
    #endif
    }

  #if DEBUG
    Serial.println("Acceleration detected!");
  #endif

  while(i <= constants::samples - constants::input_shape[1]){
    SetTensor(input_values, model_input); // Set input tensor
  
 
    status = interpreter->Invoke();  // Run inference
    if(status != kTfLiteOk){
      #if DEBUG
        Serial.println("Fail to run inference!");
      #endif
      Error();
    }
    i++;
    output += model_output->data.f[0]/(1 + constants::samples - constants::input_shape[1]);  // Average of the output
  }
  #if DEBUG
    Serial.println("Output value: " + String(output)); 
  #endif

  HandleOutput(output, false);  // LED ON
  delay(3000);  // Wait 3s
}
