# classiAcceleration

The goal of this project is creating a fitness tracker using accelerations to recognize a walk from a run.<br>
and deploy it on Arduino.

Steps of this project :

- [x] Create an acceleration recording system with an arduino.
- [x] Create a TensorFlow Lite classifier.
- [x] Deploy on Arduino nano 33.
- [x] test on real life.

## Recording system : collect the data

Two differents recording systems have been prototyped:

- [Using Arduino Uno with a SD card reader and a MPU6050 accelerometer](data_collection/data_collection_uno/README.md)
- [Using Arduino Nano 33 BLE Sense and an Android mobile phone with LightBlue app](data_collection/data_collection_ble/README.md)

This project have been done with an Arduino Nano 33 BLE Sense sending the acceleration through bluetooth.

|value|label|
|:---:|:----:|
|0|walk|
|1|run|
|2|nothing|

## [Preprocessing of the data (/data_preprocessing)](/data_preprocessing/README.md)

This folder is a python package containing tools for preprocess the data. It first transform the log made by the sniffer into a Dataframe.<br>
It also include tools for create a data generator that do data augmentation during the training of the model.

## [Data analytics (folder /analysis)](/analysis/README.md)

This folder contains differents jupyter notebook, with the differents analysis made with the data collected:

- [exploration.iypnb](analysis/exploration.ipynb) Where I check some basic statistics of my sample of data.
- [freq_analysis.iypnb](analysis/freq_analysis.ipynb) Where I compare the frequencies that compose each signal.
- [global_acc_cnn.ipynb](analysis/global_acc_cnn.ipynb) Where I train a model using the norm of the acceleration, and convert it to a TensorFlow lite model.
- [analysis/3d_acc_cnn.ipynb](analysis/3d_acc_cnn.ipynb) Where I train a model using 3d accelerations and convert it to a TensorFlow lite model.
- [detect_motion.iypnb](analysis/detect_motion.ipynb) Where I find the value when I consider that the arduino is moving.

## [inference](/inference/README.md)

Deploy the model on Arduino Nano BLE Sense board.<br>
Switch the red LED on when detecting a run, and the blue when detected a walk.
