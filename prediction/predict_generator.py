import datetime
import time

import keras
import numpy as np

from train.datasets_preparation import DatasetPreparation
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

        self.times = [time.time()]*5
        self.diffs = [time.time()]*5

    def __len__(self):
        return int(np.ceil(len(self.addresses) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        self.diffs.append(time.time() - self.times[-1])

        self.diffs.pop(0)
        self.times.pop(0)

        print(str(index) + ':' + str(len(self)))

        addresses = self.addresses[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data_generation(addresses)

        self.times.append(time.time())
        print(self.diffs)
        mean = sum(self.diffs) / float(len(self.diffs))
        rem_ex = len(self) - index

        print('est = {}'.format(datetime.timedelta(seconds=mean*rem_ex)))

        return X, np.zeros(len(X))

    def __data_generation(self, addresses):
        return np.asarray(
            [np.asarray(self.slide.cut_tile(*add))[..., :3] for add in addresses]) / 255
