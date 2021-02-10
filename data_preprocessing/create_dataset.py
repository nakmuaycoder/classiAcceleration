"""
module for dataset creation
Create a dataset from csv files

This object is a generator that compute random rotation to each row of the batch of data

author: nakmuaycoder
date: 01/2021
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from math import floor


class Dataset3dAcc(object):
    """
    Dataset3dAcc : Generator for batch of 3d acceleration
    The generator rotate the base in order to simulate the rotation of the sensor.
    """

    def __init__(self, sequence_length, batch_size, amplitude=1, scale_batch=False):
        """
        Create data table with all the sequences
        :param sequence_length:
        :param batch_size:
        :param amplitude: amplitude of the data, used for for normalisation
        """

        self.set_amplitude(amplitude)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.set_scale_batch(scale_batch)

    def set_amplitude(self, amplitude):
        """
        Set the amplitude
        :param amplitude:
        """

        if not isinstance(amplitude, int):
            raise TypeError("amplitude must be int type")

        if amplitude == 0.:
            raise ValueError("Cant divide by 0")

        self.amplitude = amplitude

    def get_validation_data(self, test_size):
        """
        Remove the validation sample from the full dataset
        :param test_size:
        :return: xval, yval
        """
        if not hasattr(self, "x") or not hasattr(self, "y"):
            print("Data must be added before create test split")
        else:
            self.x, xval, self.y, yval = train_test_split(self.x, self.y, test_size=test_size)
            return np.expand_dims(xval, axis=-1), yval.reshape((-1, 1))

    def set_scale_batch(self, scale_batch):
        """
        Scale each row of the data of the generator (divide by max)
        :param scale_batch:
        """
        if not isinstance(scale_batch, bool):
            raise TypeError("scale_batch must be boolean")

        self.scale_batch = scale_batch

    def set_batchsize(self, batch_size):
        """
        Set the batchsize
        :param batch_size: int
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be int")

        if batch_size == 0.:
            raise ValueError("batch_size can'ty be 0")

        self.batch_size = batch_size

    def add_dataframe(self, *df):
        """
        Create the data table from pd.dataframe
        :param df:
        """
        sequence_length = self.sequence_length
        t = None
        for f in df:
            f = f[["label", "x", "y", "z"]].values
            t = [f[lag: lag - sequence_length, 1:].reshape((-1, 1, 3)) for lag in range(sequence_length)]
            t = np.concatenate(t, axis=1) / self.amplitude

        if not hasattr(self, "x") or not hasattr(self, "y") or t is None:
            self.x = t
            self.y = f[:- sequence_length, 0]
        else:
            self.x = np.concatenate([self.x, t], axis=0)
            self.y = np.concatenate([self.y, f[:- sequence_length, 0]], axis=0)

    def add_folder(self, *path):
        """
        Add the content of a folder
        :param path:
        :return:
        """
        for folder in path:
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    data = pd.read_csv(os.path.join(folder, file))
                    self.add_dataframe(data)
            else:
                print(folder, "is not a valid path")

    def __iter__(self):
        return self

    def __next__(self):
        x, y = self.rotate(self.batch_size)

        if self.scale_batch:
            # Case scale each record
            m = np.max(np.abs(x), axis=(1, 2))  # Max of each record
            for _ in range(self.sequence_length):
                # scale each row
                x[:, _, 0] /= m
                x[:, _, 1] /= m
                x[:, _, 2] /= m

        return x, y

    def to_dataframe(self, add_label=True, **kwargs):
        """
        Get a data frame with x, y , z
        :param add_label: return the label column
        :param kwargs: couple column_name=column_index
        :return:
        """
        if hasattr(self, "x") and hasattr(self, "y"):
            x = self.x.reshape((self.x.shape[0], -1))

            if len(kwargs.keys()) != 0:
                x = x[:, list(kwargs.values())]
                columns_label = list(kwargs.keys())
            else:
                columns_label = [str(i) for i in range(x.shape[1])]

            if add_label is True:
                columns_label = ["label"] + columns_label
                x = np.concatenate([self.y.reshape((-1, 1)), x], axis=1)

            return pd.DataFrame(x, columns=columns_label)
        else:
            raise AttributeError("Data must be added before export")

    def rotate(self, batch_size):
        """Compute random rotation per row
        """

        # batch sampling
        index = np.random.randint(0, self.x.shape[0], batch_size)
        acc = self.x[index]
        lbl = self.y[index]

        # create a rotation axis
        # Generate random rotation axis ||u|| = 1
        u = np.random.rand(batch_size, 3)
        t = np.sqrt(np.sum(np.square(u), axis=1))
        u[:, 0] /= t
        u[:, 1] /= t
        u[:, 2] /= t

        # Generate random rotation angle
        a = np.random.rand(batch_size)  # Angle
        s = np.sin(a)
        c = np.cos(a)

        # Generate Rotation matrix

        x = 0
        y = 1
        z = 2

        R = np.zeros((batch_size, 3, 3))  # Create R
        R[:, 0, 0] = np.square(u[:, x]) * (1 - c) + c
        R[:, 0, 1] = u[:, x] * u[:, y] * (1 - c) - s * u[:, z]
        R[:, 0, 2] = u[:, x] * u[:, z] * (1 - c) + s * u[:, y]
        R[:, 1, 0] = u[:, x] * u[:, y] * (1 - c) + s * u[:, z]
        R[:, 1, 1] = np.square(u[:, y]) * (1 - c) + c
        R[:, 1, 2] = u[:, y] * u[:, z] * (1 - c) - s * u[:, x]
        R[:, 2, 0] = u[:, x] * u[:, z] * (1 - c) - s * u[:, y]
        R[:, 2, 1] = u[:, y] * u[:, z] * (1 - c) + s * u[:, x]
        R[:, 2, 2] = np.square(u[:, z]) * (1 - c) + c

        # Perform rotation
        sequences_length = acc.shape[1]

        for i in range(sequences_length):
            acc[:, i] = tf.keras.backend.batch_dot(acc[:, i], R).numpy()

        return np.expand_dims(acc, axis=-1), lbl.reshape((-1, 1))

    def data_augmentation(self, ratio):
        """
        method for data augmentation.
        :param ratio: ratio for data augmentation (>1)
        :return: numpy array with augmented rows
        """
        if ratio <= 1:
            raise ValueError("Value must be > 1")

        X, Y = np.expand_dims(self.x, axis=-1), np.expand_dims(self.y, axis=-1)
        while ratio >= 2:
            x, y = self.rotate(self.x.shape[0])
            X, Y = np.concatenate([x, X], axis=0), np.concatenate([y, Y], axis=0)
            ratio -= 1

        x, y = self.rotate(floor((ratio - 1) * self.x.shape[0]))
        X, Y = np.concatenate([x, X], axis=0), np.concatenate([y, Y], axis=0)

        return X, Y


class DatasetGlobalAcc(Dataset3dAcc):
    """
    DatasetGlobalAcc : Generator for batch of global acceleration
    Compute the norm of the acceleration: sqrt(x²+ y²+ z²)
    """

    def __init__(self, sequence_length, batch_size, amplitude=1, scale_batch=False):
        Dataset3dAcc.__init__(self, sequence_length, batch_size, amplitude, scale_batch)

    def __iter__(self):
        return self

    def __next__(self):
        """
        Create a random batch of acceleration
        """
        index = np.random.randint(0, self.x.shape[0], self.batch_size)
        x, y = self.x[index], self.y[index]
        
        if self.scale_batch:
            # Case scale each record
            m = np.max(x, axis=1)  # Max of each record
            for _ in range(self.sequence_length):
                x[:, _] /= m
        
        return np.expand_dims(x, axis=(1, -1)), y

    def add_dataframe(self, *df):
        """
        Create the data table from pd.dataframe
        :param df: pandas dataframe
        """
        sequence_length = self.sequence_length
        t = None
        for f in df:
            f = f[["label", "x", "y", "z"]].values
            t = [f[lag: lag - sequence_length, 1:].reshape((-1, 1, 3)) for lag in range(sequence_length)]
            t = np.concatenate(t, axis=1) / self.amplitude

        t = np.sqrt(np.sum(np.square(t), axis=2))  # Acceleration
        
        if not hasattr(self, "x") or not hasattr(self, "y") or t is None:
            self.x = t
            self.y = f[:- sequence_length, 0]
        else:
            self.x = np.concatenate([self.x, t], axis=0)
            self.y = np.concatenate([self.y, f[:- sequence_length, 0]], axis=0)
