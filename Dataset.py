import os


class Dataset:
    def __init__(self, path: str, directory: str):
        self._path = path
        self._directory = directory
        self.__filepath = "dataset"

    def files(self, path: str) -> [str, str]:
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield self._path, file

    def create(self, path: str, filename: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return open(os.path.join(path, filename), 'w')

    def join(self):
        with self.create(os.path.join(self._path, self._directory), self.__filepath) as join_file:
            for path, file in self.files(os.path.join(self._path, "split")):
                with open(os.path.join(path, "split", file), 'r') as data_file:
                    for line in data_file:
                        join_file.writelines("%s,%s\n" % (file, ','.join(line.split())))

    def split(self):
        split_len = 100
        for path, file in self.files(self._path):
            new_file = self.create(os.path.join(path, "split"), file)
            ctr = 0
            with open(os.path.join(path, file), "r") as fp:
                for line in fp:
                    new_file.write(line)
                    ctr += 1
                    if split_len < ctr:
                        break


db = Dataset("/Users/hp/workbench/projects/gmu/neural-network-poc/data/", "dataset")
db.split()
db.join()
