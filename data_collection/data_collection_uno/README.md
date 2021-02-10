# Record acceleration

Record the acceleration during a running or walking session.

## Hardware

* Arduino Uno
* MPU6050
* SD card writer
* 3 x LEDs
* 2 x Buttons
* 2 x 10k<img src="https://render.githubusercontent.com/render/math?math=\Omega"> resistor
* 1 x 320<img src="https://render.githubusercontent.com/render/math?math=\Omega"> resistor

## Wiring

### Accelerometer MPU6050

![full set up](../../img_doc/uno_acc.JPG)

This sensor record the acceleration on X, Y, Z axis.

### SD card writer and record system

![full set up](../../img_doc/uno_rec.JPG)

Save on a SD card the value of the acceleration.
Press the button for start/stop the acquisition.

### Mode selection

![full set up](../../img_doc/uno_mode.JPG)
Create the label of the dataset.

## Instruction

* Chose the good label by pressing the Select button
* Press the Record button to start the recording the acceleration.
* Run or walk
* Press again the Record button for stop.
