import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from utils import TrainingPlot
from utils import TimeSummary
from utils import plot_training_summary
from keras.models import load_model
import warnings

warnings.filterwarnings('ignore')


class NNKeras:
    def __init__(self, url: str):
        self._url = url
        self._base_path = "/Users/hp/workbench/projects/gmu/neural-network-poc/"
        self._num_cols = 64
        self._num_classes = 31
        self._model = "../models/model.h5"
        self._call_back = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True,
                                      write_grads=False, write_images=False, embeddings_freq=0,
                                      embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None,
                                      update_freq='epoch')
        self._call_back_model = ModelCheckpoint('../logs/model.ckpt', monitor='val_loss', verbose=0,
                                                save_best_only=False, save_weights_only=False, mode='auto', period=1)

    def read_data(self):
        df = pd.read_csv(self._url, header=None)
        X = df.iloc[:, 1:].astype(float)
        classes = df.iloc[:, 0]
        unique_classes = pd.DataFrame(classes.unique())
        rows = X.shape[0]
        unique_classes['indices'] = range(1, len(unique_classes) + 1)
        y = np.zeros((rows, len(unique_classes)), np.bool)
        for i in range(rows):
            col_idx = np.where(unique_classes.loc[:, 0] == classes[i].upper())
            y[i, col_idx] = True
        return X, y, unique_classes

    def base_model(self, nodes, num_output=31):
        model = Sequential()

        for prev_node, node in zip(nodes[:-1], nodes[1:]):
            model.add(Dense(node, activation='relu', input_dim=prev_node))  # Add the first hidden layer

        model.add(Dense(num_output, activation='sigmoid'))  # Add the output layer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, node_range, show_summary: bool = False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=5)
        for num_nodes in node_range:
            nodes = [64, num_nodes]
            model = self.base_model(nodes)
            summary = model.fit(X_train, y_train, epochs=10, verbose=0)
            score = model.evaluate(X_test, y_test)
            if show_summary:
                model.summary()
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    def train_with_callback(self, X, y, model_file: str = None):
        if model_file is None:
            model_file = self._model
        plot_losses = TrainingPlot()
        time_summary = TimeSummary()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=5)
        for num_nodes in range(31, 32):
            nodes = [64, num_nodes]
            model = self.base_model(nodes)
            callbacks = [self._call_back, time_summary, plot_losses, self._call_back_model]
            summary = model.fit(X_train, y_train, epochs=200, verbose=0, validation_data=(X_test, y_test),
                                callbacks=callbacks)
            score = model.evaluate(X_test, y_test)
            plot_training_summary(summary, time_summary)
            score = model.evaluate(X_test, y_test)
            model.save(model_file)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    def train_network_3(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)
        P = 6
        for num_layers in range(1, 10):
            nodes = [64] + [int(P / 2)] * num_layers
            model = self.base_model(nodes)
            summary = model.fit(X_train, y_train, epochs=10, verbose=0)
            score = model.evaluate(X_test, y_test)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    def train_network_4(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)
        for P in range(2, 12):
            for num_layers in range(1, 5):
                nodes = [64] + [int(P / 2)] * num_layers
                model = self.base_model(nodes)
                summary = model.fit(X_train, y_train, epochs=10, verbose=0)
                score = model.evaluate(X_test, y_test)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])
                print('Nodes:', P)

    def single_output_score(self, X, y):
        total_score = 0
        for y_i in range(self._num_classes - 1):
            X_train, X_test, y_train, y_test = train_test_split(X, y[:, y_i], test_size=0.05, random_state=5)
            num_nodes = 6
            nodes = [64, num_nodes]
            model = self.base_model(nodes, 1)
            summary = model.fit(X_train, y_train, epochs=10, verbose=0)
            score = model.evaluate(X_test, y_test)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])
            total_score += score[1]
        return total_score / self._num_classes

    def predict(self, file_path: str, classes: [], num_lines: int = 30, model_file: str = None):
        if model_file is None:
            model_file = self._model
        model = load_model(model_file)
        with open(file_path, "r") as fp:
            for i, line in enumerate(fp):
                predict_vector = np.array(line.split(",")).astype(float).reshape(1, 64)
                prediction = model.predict_classes(predict_vector)
                print(classes[0].values[prediction])
                if i > num_lines:
                    break


#
#
nn = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset/dataset.csv")
#
X, y, classes = nn.read_data()
nn.train_with_callback(X, y)
# nn.predict("/Users/hp/workbench/projects/gmu/neural-network-poc/data/fix/AE002161.csv", classes)
#
# # Use binary classification method
#
# # avg_score = nn.single_output_score(X, y)
# # print("Average accuracy:", avg_score)
