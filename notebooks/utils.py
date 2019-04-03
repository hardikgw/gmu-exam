import time
from datetime import timedelta
from itertools import groupby
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from keras.callbacks import Callback


class TrainingPlot(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        if len(self.losses) > 1:
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            plt.style.use("seaborn")
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.show()


class TimeSummary(Callback):
    def on_train_begin(self, logs={}):
        self.epoch_times = []
        self.training_time = time.process_time()

    def on_train_end(self, logs={}):
        self.training_time = time.process_time() - self.training_time

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.process_time()

    def on_epoch_end(self, batch, logs={}):
        self.epoch_times.append(time.process_time() - self.epoch_time_start)


def plot_training_summary(training_summary, time_summary=None, text=""):
    if time_summary:
        print('Training time: ' + str(timedelta(seconds=time_summary.training_time)))
        print('Epoch time avg: ' + str(timedelta(seconds=mean(time_summary.epoch_times))))
    hist = sorted(training_summary.history.items(),
                  key=lambda x: (x[0].replace('val_', ''), x[0]))

    epochs = [e + 1 for e in training_summary.epoch]
    for metric, values in groupby(hist,
                                  key=lambda x: x[0].replace('val_', '')):
        if 'val_loss' in training_summary.history:
            val0, val1 = tuple(values)
            plt.plot(epochs, val0[1], epochs, val1[1], '--', marker='o')
        else:
            val0 = tuple(values)[0]
            plt.plot(epochs, val0[1], '--', marker='o')
        plt.xlabel('epoch'), plt.ylabel(val0[0])
        plt.legend(('Train set', 'Validation set'))
        plt.text(.15, .1, text, va='center', size=12)
        plt.show()
