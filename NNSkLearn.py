import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as pp

from sklearn.neural_network import MLPClassifier


class NNSkLearn:
    def __init__(self, path: str, file: str):
        self._path = path
        self._file = file
        self._num_cols = 64

    def read_data(self):
        df = pd.read_csv(os.path.join(self._path, self._file), header=None)
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]
        return X, y

    def train(self, X, y):
        le = pp.LabelEncoder
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        y = y.apply(le.fit_transform)
        print(y.unique())
        print(y_test.unique())
        print(y_train.unique())


nn = NNSkLearn("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset64.csv")
X, y = nn.read_data()
nn.train(X, y)
