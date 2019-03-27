import tensorflow as tf
import os


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

    # split_files(path_)
    join_split_files(path_)


# def train():


def main():
    create_dataset()
    # train()


main()
