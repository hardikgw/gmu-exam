import os
import numpy as np
import pandas as pd

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential


class Database:
    def __init__(self, path: str, directory: str):
        self._path = path
        self._directory = directory
        self.__filepath = "dataset"

    def files(self, path: str) -> [str, str]:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield self._path, file

    def create(self, path: str, filename: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return open(os.path.join(path, filename), 'w')

    def join(self):
        with self.create(os.path.join(self._path, self._directory), self.__filepath) as join_file:
            for path, file in self.files(os.path.join(self._path, "split")):
                with open(os.path.join(path, "split", file), 'r') as data_file:
                    for line in data_file:
                        join_file.writelines("%s,%s\n" % (file, ','.join(line.split())))

    def split(self):
        split_len = 100
        for path, file in self.files(self._path):
            new_file = self.create(os.path.join(path, "split"), file)
            ctr = 0
            with open(os.path.join(path, file), "r") as fp:
                for line in fp:
                    new_file.write(line)
                    ctr += 1
                    if split_len < ctr:
                        break

    def train(self):
        path_ = "/Users/hp/workbench/projects/gmu/neural-network-poc/data/"
        file_name = 'dataset'
        lables = ['R', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        dataset = pd.read_csv(os.path.join(path_, file_name, file_name), names=lables)

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


db = Database("/Users/hp/workbench/projects/gmu/neural-network-poc/data/", "dataset")

db.join()
