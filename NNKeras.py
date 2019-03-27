import os
import numpy as np
import pandas as pd

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras import Sequential


def create_dataset():
    path_ = '/Users/hp/workbench/projects/gmu/gmu-exam/data'
    split_len = 100
    split_dir = 'split'

    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield path, file

    def create_file(path, dir, filename):
        new_folder = os.path.join(path, dir)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        # else:
        #     shutil.rmtree(new_folder)
        #     os.makedirs(new_folder)
        return open(os.path.join(new_folder, filename), 'w')

    def join_split_files(path):
        new_file_name = "dataset"
        new_file = create_file(path, 'dataset', new_file_name)
        for p, f in files(os.path.join(path, split_dir)):
            with open(os.path.join(p, f), "r") as fp:
                for line in fp:
                    new_file.writelines(f + ',' + ','.join(line.split()) + '\n')

    def split_files(path):
        for p, f in files(path):
            ctr = 0
            new_file = create_file(p, split_dir, f)
            with open(os.path.join(p, f), "r") as fp:
                for line in fp:
                    new_file.write(line)
                    ctr += 1
                    if split_len < ctr:
                        break

    split_files(path_)
    join_split_files(path_)


def train():
    path_ = '/Users/hp/workbench/projects/gmu/gmu-exam/data'
    file_name = 'dataset'
    lables = ['R', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    train = pd.read_csv(os.path.join(path_, file_name, file_name), names=lables)
    test = pd.read_csv(os.path.join(path_, file_name, file_name), names=lables)

    print(train.head(2000))
    num_classes = 10

    trainX = train[:, 1:].reshape(train.shape[0], 1, 11, 11).astype('float32')

    y_train = train[:, 0]

    # Reshape and normalize test data
    testX = test[:, 1:].reshape(test.shape[0], 1, 11, 11).astype('float32')

    y_test = test[:, 0]

    model = Sequential()
    # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))


def main():
    # create_dataset()
    train()


main()
