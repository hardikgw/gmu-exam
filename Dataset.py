import os


class Dataset:
    def __init__(self, path: str, directory: str):
        self._path = path
        self._directory = directory
        self.__filename = "dataset"
        self.__extension = ".csv"
        self._split_len = -1
        self._dataset_percent = 5

    def files(self, path: str) -> [str, str]:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield self._path, file

    def create(self, path: str, filename: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return open(os.path.join(path, filename), 'w')

    def join(self):
        with self.create(os.path.join(self._path, self._directory), self.__filename + self.__extension) as join_file:
            for path, file in self.files(os.path.join(self._path, "split")):
                with open(os.path.join(path, "split", file), 'r') as data_file:
                    ctr = 0
                    for line in data_file:
                        if ctr % self._dataset_percent == 0:
                            join_file.writelines("%s,%s\n" % (file, ','.join(line.split())))
                        ctr += 1

    def split(self):
        for path, file in self.files(self._path):
            new_file = self.create(os.path.join(path, "split"), file)
            ctr = 0
            with open(os.path.join(path, file), "r") as fp:
                for line in fp:
                    new_file.write(line)
                    ctr += 1
                    if 0 < self._split_len < ctr:
                        break

    def split_by_percent(self):
        for path, file in self.files(self._path):
            new_file = self.create(os.path.join(path, "split"), file)
            ctr = 0
            with open(os.path.join(path, file), "r") as fp:
                for line in fp:
                    if ctr % self._dataset_percent == 0:
                        new_file.write(line)
                    ctr += 1

    def join_by_vector_size(self, vector_size: int):
        filename = self.__filename + str(vector_size) + self.__extension
        with self.create(os.path.join(self._path, self._directory), filename) as join_file:
            for path, file in self.files(os.path.join(self._path, "split")):
                with open(os.path.join(path, "split", file), 'r') as data_file:
                    ctr = 0
                    x_vector = []
                    for line in data_file:
                        x_vector += line.split()
                        if len(x_vector) >= vector_size:
                            join_file.writelines(
                                "%s,%s\n" % ((os.path.splitext(file)[0]), ','.join(x_vector[:vector_size])))
                            x_vector = x_vector[vector_size:]
                    ctr += 1
        pass


db = Dataset("/Users/hp/workbench/projects/gmu/neural-network-poc/data/", "dataset")
# db.split()
# db.split_by_percent()
# db.join()
db.join_by_vector_size(64)
