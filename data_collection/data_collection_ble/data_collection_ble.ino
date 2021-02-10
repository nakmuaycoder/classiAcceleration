/*

Stream the acceleration to an android mobile phone using BLE.
Run on Arduino Nano 33 BLE Sense.

Mbed Os version: 1.1.6
ArduinoBLE: 1.1.3
Arduino_LSM9DS1: 1.1.0

Stream start when user write the value 1 to the record Characteristic (2106).

author: nakmuaycoder
date: 01/21
*/


#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>

#define DEBUG 1

// Create service
BLEService AccService("FA01");  // Create streaming service
BLEFloatCharacteristic AccCharacteristic_x("2102", BLENotify | BLEIndicate);
BLEFloatCharacteristic AccCharacteristic_y("2103", BLENotify | BLEIndicate);
BLEFloatCharacteristic AccCharacteristic_z("2104", BLENotify | BLEIndicate);
BLEIntCharacteristic Val("2105", BLEWrite | BLERead);  // Label of the data
BLEIntCharacteristic Record("2106", BLEWrite);  // Record Yes/No


// Create descriptor for caracteristic
BLEDescriptor AccDes_x("2103", "acceleration");
BLEDescriptor RecDes("2105", "Record");
BLEDescriptor ValDes("2106", "Record");


void setup(){
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  
  #if DEBUG
  Serial.begin(9600);
  while (!Serial);
  #endif

  if(!IMU.begin()) {
    #if DEBUG
    Serial.println("Failed to initialize IMU!");
    #endif
    digitalWrite(LED_BUILTIN, HIGH);
    while (1);
  }

  if(!BLE.begin()){
    #if DEBUG
    Serial.println("Failed to initialize BLE");
    #endif
    digitalWrite(LED_BUILTIN, HIGH);
    while(1);
  }

  // Add Characteristic Descriptor
  AccCharacteristic_x.addDescriptor(AccDes_x);
  AccCharacteristic_y.addDescriptor(AccDes_x);
  AccCharacteristic_z.addDescriptor(AccDes_x);
  Record.addDescriptor(RecDes);
  Val.addDescriptor(ValDes);

  AccCharacteristic_x.broadcast();
  AccCharacteristic_y.broadcast();
  AccCharacteristic_z.broadcast();
  
  BLE.setDeviceName("Arduino");
  BLE.setAppearance(384); 
  
  BLE.setLocalName("Acceleration Streamer");
  BLE.setAdvertisedService(AccService);
  AccService.addCharacteristic(AccCharacteristic_x);
  AccService.addCharacteristic(AccCharacteristic_y);
  AccService.addCharacteristic(AccCharacteristic_z); 
  AccService.addCharacteristic(Record);
  AccService.addCharacteristic(Val);

  BLE.addService(AccService);
  AccCharacteristic_x.writeValue(0);
  AccCharacteristic_y.writeValue(0);
  AccCharacteristic_z.writeValue(0);
  Record.writeValue(0);
  Val.writeValue(0);
  BLE.advertise();
}



void loop(){
  float x, y, z;
  
  BLEDevice central = BLE.central();
  
  if(central && IMU.accelerationAvailable()){
    // Case both client and Accelerometer available
    while(central.connected()){
      if(Record.value() == 1){
        IMU.readAcceleration(x, y, z);

        #if DEBUG
        Serial.print("Acceleration: \n\t x= ");
        Serial.print(x); Serial.print("\n\ty= ");
        Serial.print(y);Serial.print("\n\tz= ");
        Serial.println(z);
        #endif

        AccCharacteristic_x.writeValue(x);
        AccCharacteristic_y.writeValue(y);
        AccCharacteristic_z.writeValue(z);
      }
    }
  }
}
