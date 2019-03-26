import tensorflow as tf
from tensorflow.keras import layers


def main():
    print('')


# iris = datasets.load_iris()
# x = iris.data
# y = iris.target
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)
#
# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

def split_files():
    path_ = '/Users/hp/workbench/projects/gmu/gmu-exam/data'
    split_len = 100

    import os

    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield path + '/' + file

    for f in files(path_):
        ctr = 0
        with open(f, "r") as fp:
            for line in fp:
                print(line)
                ctr += 1
                if split_len < ctr:
                    break


split_files()
