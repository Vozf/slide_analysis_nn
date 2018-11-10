import csv
from concurrent.futures import ThreadPoolExecutor

import cv2
import keras
import numpy as np
from keras.utils import to_categorical

from slide_analysis_nn.train.datasets_preparation import DatasetPreparation


class Generator(keras.utils.Sequence):
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size

        with open(data_path, 'r') as csvfile:
            file = list(csv.reader(csvfile))[1:]
            self.data = np.asarray(file)

        if not self.data.shape[1]:
            raise Exception('Empty csv file')

        self.labels_names_to_id = DatasetPreparation.get_label_name_to_label_id_dict()

    def num_classes(self):
        return len(self.labels_names_to_id)

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        paths, label_names = self.data[index * self.batch_size:(index + 1) * self.batch_size].T

        X, y = self.__data_generation(paths, label_names)

        return X, y

    def __data_generation(self, paths, label_names):
        with ThreadPoolExecutor() as executor:
            rgb_iter = executor.map(lambda path: cv2.imread(path, cv2.IMREAD_COLOR), paths)

        rgb0_255 = np.asarray(list(rgb_iter))

        return rgb0_255 / 255, self._get_one_hot(label_names)

    def _get_one_hot(self, label_names):
        labels = [int(label_name) for label_name in label_names]
        return to_categorical(labels, self.num_classes())

    def delete_all_samples_except(self, num_samples):
        self.data = self.data[np.random.choice(self.data.shape[0], num_samples, replace=False)]

