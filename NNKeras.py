import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential


class NNKeras:
    def __init__(self, path: str, file: str):
        self._path = path
        self._file = file
        self._num_cols = 64

    def read_split_data(self):
        df = pd.read_csv(os.path.join(self._path, self._file), header=None)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        return train_test_split(X, y, test_size=0.33)


train = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset.csv")
print(train.read_split_data())
