import csv

import cv2
import numpy as np
import keras
from keras.utils import to_categorical


class Generator(keras.utils.Sequence):
    def __init__(self, data_path, class_mapping_path, batch_size):
        self.batch_size = batch_size

        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.data = np.asarray([(row[0], row[-1]) for row in reader])

        with open(class_mapping_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.names_to_label = dict(reader)

    def num_classes(self):
        return len(self.names_to_label)

    def __len__(self):
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        paths, label_names = self.data[index * self.batch_size:(index + 1) * self.batch_size].T

        X, y = self.__data_generation(paths, label_names)

        return X, y

    def __data_generation(self, paths, label_names):
        return np.asarray([cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]) / 255, \
               self._get_one_hot(label_names)

    def _get_one_hot(self, label_names):
        labels = [self.names_to_label[label_name] for label_name in label_names]
        return to_categorical(labels, self.num_classes())

