import cProfile

import numpy as np
import keras
import time
from openslide.deepzoom import DeepZoomGenerator

from utils.constants import TILE_SIZE, TILE_STEP
from utils.slide import Slide


class PredictGenerator(keras.utils.Sequence):
    def __init__(self, slide_path, batch_size):
        self.batch_size = batch_size
        self.slide = Slide(slide_path)

        x = range(0, self.slide.slide.dimensions[0], TILE_STEP)
        y = range(0, self.slide.slide.dimensions[1], TILE_STEP)
        self.addresses = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        print(self.addresses)

        self.times = [time.time()]
        self.diffs = []

    def __len__(self):
        return int(np.floor(len(self.addresses) / self.batch_size))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        self.times.append(time.time())


        print(str(index) + ':' + str(len(self)))
        addresses = self.addresses[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data_generation(addresses)

        self.diffs.append(time.time() - self.times[-1])
        print(self.diffs)

        return X, np.zeros(len(X))

    def __data_generation(self, addresses):
        return np.asarray([np.asarray(self.slide.cut_tile(*add))[..., :3] for add in addresses])

    def create_asap_annotations(self, scores):
        pass
