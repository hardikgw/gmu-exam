import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


class NNSkLearn:
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

    def train(self):
        X, y, unique_classes = self.read_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=3, random_state=1)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score)
        return score


nn = NNSkLearn("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset", "dataset64.csv")
nn.train()
