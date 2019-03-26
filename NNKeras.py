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
    import shutil

    path_ = '/Users/hp/workbench/projects/gmu/gmu-exam/data'
    split_len = 100
    split_dir = 'split'
    import os

    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield path, file

    def create_file(path, dir, filename):
        new_folder = path + '/' + dir
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        # else:
        #     shutil.rmtree(new_folder)
        #     os.makedirs(new_folder)
        return open(new_folder + '/' + filename, 'w')

    for p, f in files(path_):
        ctr = 0
        new_file = create_file(p, split_dir, f)
        with open(p + '/' + f, "r") as fp:
            for line in fp:
                new_file.write(line)
                ctr += 1
                if split_len < ctr:
                    break


split_files()
