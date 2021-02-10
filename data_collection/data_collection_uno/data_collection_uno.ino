/*

data_collection.ino:
Record acceleration using Arduino, SD shield and mpu6050.
Run on Arduino UNO.


Adafruit_MPU6050:
SD: 

author: nakmuaycoder
date: 01/21
 
 */


#include <SD.h>
#include <Adafruit_MPU6050.h>
Adafruit_MPU6050 mpu;

bool record = false;

File datafile;

// Digital pin
const int button_record = 4;
const int button_mode = 3;
const int cs = 10;
int mode = 0;

#define DEBUG 1


void setup(){

  #if DEBUG
  Serial.begin(9600);
  while(!Serial);
  #endif

  // Button
  pinMode(button_record, INPUT);
  pinMode(button_record, INPUT);

  // LED 
  for(int i=5; i<7; i++){
    pinMode(i, OUTPUT);
    digitalWrite(i, LOW);
  }

  digitalWrite(5, HIGH);

  
 pinMode(LED_BUILTIN, OUTPUT);
 digitalWrite(LED_BUILTIN, HIGH);
  
  if (!SD.begin(cs)) {
    #if DEBUG
    Serial.println("Fail to initialize SD");
    #endif

    digitalWrite(LED_BUILTIN, HIGH);
    while (1);
  }

  #if DEBUG
  Serial.println("SD Found!");
  #endif

  if (!mpu.begin()){
    #if DEBUG
    Serial.println("Failed to find MPU6050 chip");
    #endif

    digitalWrite(LED_BUILTIN, HIGH);
    while (1);
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);//  2, 4, 8, 16
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);// 260, 184, 94, 44, 21 10, 5
}



void loop() {

  if(digitalRead(button_mode) == HIGH && !record){
    digitalWrite(mode + 5, LOW);
    mode++;
    mode %= 3;
    digitalWrite(mode + 5, HIGH);

    #if DEBUG
    Serial.print("Set mode to : ");
    Serial.println(mode);
    #endif

    delay(500);
  }

  
  if(digitalRead(button_record) == HIGH){
    record = !record;
    if(record){
    digitalWrite(LED_BUILTIN, HIGH);
    }else{
      digitalWrite(LED_BUILTIN, LOW);
    };
    delay(500);
  }

  if (!record && datafile){
    //Case stop recording
    datafile.close();
    #if DEBUG
    Serial.println("Close the file!");
    #endif
    }
    
  if(record && !datafile){
    // Case start recording & no file
    datafile = SD.open("acc.txt", FILE_WRITE);
    #if DEBUG
    Serial.println("Creation of the log");
    #endif
  }

  if(record && datafile){
    //Write a value on the file
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);
    datafile.print(mode);
    datafile.print("; ");    
    datafile.print(a.acceleration.x);
    datafile.print("; ");
    datafile.print(a.acceleration.y);
    datafile.print("; ");
    datafile.println(a.acceleration.z);
  
    #if DEBUG
    // Show the reult in serial print
    Serial.print(mode);
    Serial.print("; ");    
    Serial.print(a.acceleration.x);
    Serial.print("; ");
    Serial.print(a.acceleration.y);
    Serial.print("; ");
    Serial.println(a.acceleration.z);
    #endif
  }  
}
