import os
import numpy as np
import pandas as pd

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential


class NNKeras:
    def __init__(self, path: str, file: str):
        self._path = path
        self._file = file

    def prepare(self):
        lables = ['R', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        dataset = pd.read_csv(os.path.join(self._path, self._file), names=lables)

        dataset.describe()
        print(dataset.head(2000))
        num_classes = 10

        model = Sequential()
        # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # model.add(Conv2D(64, (5, 5), activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(1000, activation='relu'))
        # model.add(Dense(num_classes, activation='softmax'))


train = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset")
