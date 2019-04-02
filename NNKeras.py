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
    def __init__(self, url: str):
        self._url = url
        self._num_cols = 64

    def read_data(self):
        df = pd.read_csv(self._url, header=None)
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

    def base_model(self, nodes):
        model = Sequential()

        for prev_node, node in zip(nodes[:-1], nodes[1:]):
            model.add(Dense(node, activation='relu', input_dim=prev_node))  # Add the first hidden layer

        model.add(Dense(31, activation='sigmoid'))  # Add the output layer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)

        for num_nodes in range(2, 100):
            nodes = [64, num_nodes]
            model = self.base_model(nodes)
            model.fit(X_train, y_train, epochs=1, verbose=0)
            loss, accuracy = model.evaluate(X_test, y_test)
            print(num_nodes, accuracy)


nn = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset/dataset.csv")
X, y, unique_classes = nn.read_data()
nn.train(X, y)
