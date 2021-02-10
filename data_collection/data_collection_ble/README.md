# data_collection_ble

Use Arduino Nano 33 ble sense to record acceleraions.<br>
This board have an builtin Inertial Measurement Unit (IMU), and a Bluetooth Low Energy (BLE) chip.<br>
This sketch set the Arduino as a BLE peripheral and stream the records via BLE.

<img width="250" alt="arduino nano 33 ble sense" src="../../img_doc/arduino_ble.JPG">

## Collect the accelerations using LightBlue for android

Instruction:

- 1 Upload data_collection_ble.ino using Arduino IDE.

- 2 Connect the phone to the **Acceleration Streamer** service.

<img width="250" alt="paired the phone and arduino" src="../../img_doc/Screenshot_20201120-213157_LightBlue.jpg">

- 3 Subscribe to the characteristic of UUIDs **2102**, **2103**, **2104** and **2105** for get a notification after every update.

<img width="250" alt="select the characteristic" src="../../img_doc/Screenshot_20201120-213230_LightBlue.jpg">

<img width="250" alt="subscribe to all of the characteristique" src="../../img_doc/Screenshot_20201120-213238_LightBlue.jpg">

- 5 Set the label: Click on the caracteristic of **UUID 2105** and write the label.

|value|label|
|:---:|:----:|
|0|walk|
|1|run|

<img src="../../img_doc/Screenshot_20201120-213300_LightBlue.jpg" width="250" alt="">

- 6 Set the characteristic of UUID **2106** to 1 to start recording.

- 5 Click on the 3 dots then log and then save the log.

<img src="../../img_doc/Screenshot_20201120-213354_LightBlue.jpg" width="250" alt="">

<img src="../../img_doc/Screenshot_20201120-213403_LightBlue.jpg" width="250" alt="">
