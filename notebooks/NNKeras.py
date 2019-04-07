import numpy as np
import pandas as pd
import os
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from utils import TrainingPlot
from utils import TimeSummary
from utils import plot_training_summary
from keras.models import load_model
from keras.regularizers import l2
import time


class NNKeras:
    def __init__(self, url: str):
        self._url = url
        self._base_path = "/Users/hp/workbench/projects/gmu/neural-network-poc/"
        self._num_cols = 64
        self._num_classes = 31
        self._model = "../models/model.h5"
        self._log_dir = "../logs"

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

    def base_model(self, nodes, num_output=31, kernel_regularizer=None, layer_id=None):
        model = Sequential()
        for prev_node, node in zip(nodes[:-1], nodes[1:]):
            layer_name = None
            if layer_id is not None:
                layer_name = "{}-in-{}-n-{}".format(layer_id, str(prev_node), str(prev_node))
            model.add(Dense(node, activation='relu', kernel_regularizer=kernel_regularizer,
                            input_dim=prev_node, name=layer_name))
        model.add(Dense(num_output, activation='sigmoid', name='features'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y, show_summary: bool = False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=5)
        nodes = [64, 6]
        model = self.base_model(nodes)
        summary = model.fit(X_train, y_train, epochs=10, verbose=0)
        score = model.evaluate(X_test, y_test)
        if show_summary:
            model.summary()
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def train_with_callback(self, X, y, model_file: str = None, generate_plots=True):
        if model_file is None:
            model_file = self._model
        plot_losses = TrainingPlot()
        time_summary = TimeSummary()
        layer_names = ['features']
        num_nodes = 6
        for layer_id in range(6, 7):
            nodes = [64, num_nodes]
            for prev_node, node in zip(nodes[:-1], nodes[1:]):
                layer_name = None
                if layer_id is not None:
                    layer_name = "{}-in-{}-n-{}".format(layer_id, str(prev_node), str(prev_node))
                    layer_names.append(layer_name)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=5)
        with open(os.path.join(self._log_dir, 'metadata.tsv'), 'w') as f:
            np.savetxt(f, y_test)
        tensorboard = TensorBoard(log_dir=self._log_dir, histogram_freq=2, batch_size=32,
                                  write_graph=True,
                                  write_grads=False, write_images=True, embeddings_freq=1,
                                  embeddings_layer_names=['features'],
                                  embeddings_data=X_test,
                                  update_freq='epoch')
        callbacks = [tensorboard, time_summary]
        if generate_plots:
            callbacks.append(plot_losses)
        for layer in range(6, 7):
            nodes = [64, num_nodes]
            model = self.base_model(nodes, layer_id="l-{}".format(layer))
            summary = model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test),
                                callbacks=callbacks)
            score = model.evaluate(X_test, y_test)
            if generate_plots:
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

    def train_network_2(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=5)
        for P in range(2, 12):
            for num_layers in range(1, 2):
                nodes = [64] + [int(P / 2)] * num_layers
                print(nodes)
                model = self.base_model(nodes)
                summary = model.fit(X_train, y_train, epochs=10, verbose=0)
                score = model.evaluate(X_test, y_test)
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])

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


nn = NNKeras("/Users/hp/workbench/projects/gmu/neural-network-poc/data/dataset/dataset1.csv")
X, y, classes = nn.read_data()
nn.train_with_callback(X, y, "../logs/model.ckpt", False)
# nn.predict("/Users/hp/workbench/projects/gmu/neural-network-poc/data/fix/AE002161.csv", classes)
# # avg_score = nn.single_output_score(X, y)
# # print("Average accuracy:", avg_score)
