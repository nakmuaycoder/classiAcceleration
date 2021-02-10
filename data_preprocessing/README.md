
# data_preprocessing package

This python package contains tools for convert the log into a dataset.

## Convert log into csv with log_preprocessing.py

Convert a log of that type:

```txt
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002102-0000-1000-8000-00805f9b34fb changed | value: 00 E0 EE BE
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002103-0000-1000-8000-00805f9b34fb changed | value: 00 40 D3 3E
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002104-0000-1000-8000-00805f9b34fb changed | value: 00 08 55 BF
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002105-0000-1000-8000-00805f9b34fb changed | value: 00 00 00 00
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002102-0000-1000-8000-00805f9b34fb changed | value: 00 B0 F2 BE
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002103-0000-1000-8000-00805f9b34fb changed | value: 00 D0 D1 3E
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002104-0000-1000-8000-00805f9b34fb changed | value: 00 00 55 BF
Mon Nov 16 17:25:11 GMT+01:00 2020: Characteristic 00002105-0000-1000-8000-00805f9b34fb changed | value: 00 00 00 00
```

to a csv file:

|date|label|x|y|z|
|:----:|:----:|:----:|:----:|:----:|
2020-11-16 17:25:11|0|-0.466552734375|0.41259765625|-0.8321533203125
2020-11-16 17:25:11|0|-0.4739990234375|0.4097900390625|-0.83203125

From a terminal

````sh
$cd data_preprocessing/
$python log_preprocessing.py path_log.txt path_output.csv path_uuid.json mode
````

From a python interpreter

```py
>>>from data_preprocessing.log_preprocessing import LogParser

>>>header = {"2102": "x", "2103": "y", "date": "date", "2104": "z", "label": "2105" }

>>>log1 = 'data/LBX_LOGS_2020-11-16_17-47_walk.txt'
>>>log2 = 'data/LBX_LOGS_2020-11-16_18-05_run.txt'
>>>csv1 = 'data/out1.txt'
>>>csv2 = 'data/out2.txt'

>>>parser = LogParser(header)(log1)  # Parse the first file
>>>parser.write_file(csv1)  # Save as csv the first file

>>>parser(log2)  # append inpt and inpt2 records
>>>parser.write_file(csv2)  # Save as csv the first file
```

## Create the data generator

Create a data generator for the training.
Use next funct-ion to return a single batch of data.
2 Generator are implemented:

- DatasetGlobalAcc : Return a single batch with the norm of the acceleration.
- Dataset3dAcc : Return a single batch of 3d acceleration.

### Global acceleration

```py
>>>from data_preprocessing.create_dataset import DatasetGlobalAcc

>>>sequence_size = 10
>>>batch_size = 1
>>>xtrain = DatasetGlobalAcc(sequence_length, batch_size, amplitude=1, scale_batch=False)
>>>path_data = "data/clean/"

>>>xtrain.add_folder(path_data)  # Import all the csv file of this folder

>>>xtrain.add_dataframe(df1, df2,...,dfn)  # Add pandas dataFrame

>>>xval, yval = xtrain.get_validation_data(test_size=0.2)  # generate the validation sample

>>>xtrain.to_dataframe(add_label=True, x=0, y=1, z=2)  # Export the column 0 as x, 1 as y and 2 as z

>>>next(xtrain)  # Sample a random single batch of training data and compute random rotation to each element

>>>xtrain.set_scale_batch(True)  # Can be use for scale the records (divide by the max absolute value of the record)
```

### 3d data points

Create a data generator for the training.
This object import a CSV and return a batch of 3d acceleration.<br>
During the training phase of the model, the generator randomly perform a rotation of the <img src="https://render.githubusercontent.com/render/math?math=(O, \vec{x}, \vec{y}, \vec{z})"> base in order to make the model working without any impact of the orientation of the sensor.

```py
>>>from data_preprocessing.create_dataset import Dataset3dAcc

>>>sequence_size = 10
>>>batch_size = 1024

>>>xtrain = Dataset3dAcc(sequence_length, batch_size, amplitude=1, scale_batch=False)
>>>path_data = "data/clean/"

>>>next(xtrain)
```

## create tests array

Create 3 C-arrays with a batch of 3d accelerations.

```sh
$python generate_test_array.py nothing.txt walk.txt run.txt arrays_test.h batch_size
```

or from a python interpreter:

```py
>>>from data_preprocessing.generate_test_array import generate_test

>>>generate_test(nothing_path="nothing.txt", walk_path="walk.txt", run_path="run.txt", output_path="array_test.h", observations=3)
```

Produce a array_test.h for testing the inference on Arduino:

```cpp
float nothing[] = {
	0.0169677734375,-0.028564453125,0.95166015625,
	0.0172119140625,-0.0283203125,0.9512939453125,
	0.0169677734375,-0.0281982421875,0.95263671875
};

float walk[] = {
	1.2152099609375,0.5880126953125,0.343505859375,
	0.5302734375,-0.24169921875,-0.0330810546875,
	0.963134765625,0.362060546875,0.1365966796875
};

float run[] = {
	0.7203369140625,-1.0694580078125,-0.9503173828125,
	-0.8419189453125,-2.3570556640625,-1.5958251953125,
	-1.375244140625,-2.66064453125,0.457763671875
};
```
