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

    def prepare(self):
        dataset = pd.read_csv(os.path.join(self._path, self._file), header=None)

        X = dataset[:, 1, 3]
        y = dataset[:, 1]

        print(X)
        print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # print(dataset.head(2000)

        model = Sequential()
        model.add(Dense(1, activation='relu', input_shape=(1,)))


train = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset")
train.prepare()
