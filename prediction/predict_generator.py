import cProfile

import numpy as np
import keras
import time
import os

from train.datasets_preparation.preparation import DatasetPreparation
from train.datasets_preparation.settings import DEFAULT_CLASS_NAME
from utils.ASAP_xml import write_polygons_xml
from utils.constants import TILE_SIZE, TILE_STEP
from utils.slide import Slide


class PredictGenerator(keras.utils.Sequence):
    def __init__(self, slide_path, batch_size):
        self.batch_size = batch_size
        self.slide = Slide(slide_path)

        x = range(0, self.slide.slide.dimensions[0]-TILE_SIZE, TILE_STEP)
        y = range(0, self.slide.slide.dimensions[1]-TILE_SIZE, TILE_STEP)
        self.addresses = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        self.label_names_to_id = DatasetPreparation.get_label_name_to_label_id_dict()

        self.times = [time.time()]
        self.diffs = []

    def __len__(self):
        return int(np.floor(len(self.addresses) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        self.diffs.append(time.time() - self.times[-1])


        print(str(index) + ':' + str(len(self)))

        addresses = self.addresses[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data_generation(addresses)

        self.times.append(time.time())
        print(self.diffs[-5:])

        return X, np.zeros(len(X))

    def __data_generation(self, addresses):
        return np.asarray(
            [np.asarray(self.slide.cut_tile(*add))[..., :3] for add in addresses]) / 255

    def create_asap_annotations(self, predicted_labels, scores):
        xml_path = '{}_predicted.xml'.format(os.path.splitext(self.slide.slide_path)[0])
        polygons = np.asarray([
            [(x1, y1), (x1 + TILE_SIZE, y1), (x1 + TILE_SIZE, y1 + TILE_SIZE), (x1, y1 + TILE_SIZE)]
            for (x1, y1) in self.addresses])

        chosen_idx = predicted_labels == self.label_names_to_id[DEFAULT_CLASS_NAME]

        return write_polygons_xml(polygons[chosen_idx],
                                  predicted_labels=predicted_labels[chosen_idx],
                                  scores=scores[chosen_idx], xml_path=xml_path)
