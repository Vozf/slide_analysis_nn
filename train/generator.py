import csv

import cv2
import numpy as np
import keras


class Generator(keras.utils.Sequence):
    def __init__(self, data_path, class_mapping_path, batch_size):
        self.batch_size = batch_size

        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.data = [(row[0], row[-1])for row in reader]

        with open(class_mapping_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.names_to_label = dict(reader)

    def num_classes(self):
        # add fake unused class for crossentropy to work
        return len(self.names_to_label)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        data = self.data[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__data_generation(data)

        return X, y

    def __data_generation(self, data):
        return np.asarray([cv2.imread(path, cv2.IMREAD_COLOR) for path, _ in data]), \
               np.asarray([self._get_one_hot(label_name) for _, label_name in data])

    def _get_one_hot(self, label_name):
        # y = [self.names_to_label[label_name]] if label_name in self.names_to_label else []
        # return to_categorical(y, self.num_classes())
        arr = np.zeros(self.num_classes(), dtype=int)
        if label_name in self.names_to_label:
            arr[int(self.names_to_label[label_name])] = 1

        return arr

