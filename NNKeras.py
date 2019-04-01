import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential
from keras.utils import np_utils
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential


class NNKeras:
    def __init__(self, path: str, file: str):
        self._path = path
        self._file = file
        self._num_cols = 64

    def read_data(self):
        df = pd.read_csv(os.path.join(self._path, self._file), header=None)
        X = df.iloc[:, 1:].astype(float)
        classes = df.iloc[:, 0]
        unique_classes = pd.DataFrame(sorted([c.upper() for c in classes.unique()]))
        rows = X.shape[0]
        unique_classes['indices'] = range(1, len(unique_classes) + 1)
        y = np.zeros((rows, len(unique_classes)), np.bool)
        for i in range(rows):
            col_idx = np.where(unique_classes.loc[:, 0] == classes[i].upper())
            y[i, col_idx] = True
        return X, y, unique_classes

    def base_model(self):
        model = Sequential()
        # Add the first hidden layer
        model.add(Dense(32, activation='relu', input_dim=64))
        # Add the second hidden layer
        model.add(Dense(16, activation='relu'))
        # Add the output layer
        model.add(Dense(31, activation='sigmoid'))
        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # Train the model for 200 epochs
        return model

    def train(self):
        X, y, unique_classes = self.read_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        model = self.base_model()
        model.fit(X_train, y_train, epochs=20)


nn = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset64.csv")
nn.train()
