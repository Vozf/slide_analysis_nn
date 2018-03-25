import numpy as np
import keras
import time
from openslide.deepzoom import DeepZoomGenerator

from utils.constants import TILE_SIZE
from utils.slide import Slide


class SlideTileGenerator(keras.utils.Sequence):
    def __init__(self, slide_path, batch_size=32):
        self.batch_size = batch_size
        self.slide = Slide(slide_path)
        self.dz = DeepZoomSlide(self.slide)
        self.addresses = self.dz.get_addresses()
        self.times = [time.time()]
        self.diffs = []

    def __len__(self):
        return int(np.floor(len(self.dz) / self.batch_size))

    def __getitem__(self, index):
        self.times.append(time.time())
        self.diffs.append(self.times[-1] - self.times[-2])
        print(self.diffs)

        print(str(index) + ':' + str(len(self)))
        addresses = self.addresses[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data_generation(addresses)
        return X, np.zeros(len(X))

    def __data_generation(self, addresses):
        return np.asarray(list(map(self.dz.get_tile, addresses)))


class DeepZoomSlide:
    def __init__(self, slide:Slide):
        self.deepzoom = DeepZoomGenerator(slide.slide, tile_size=TILE_SIZE, overlap=0)

        x = np.arange(0, self.deepzoom.level_tiles[-1][0])
        y = np.arange(0, self.deepzoom.level_tiles[-1][1])

        self.addresses = np.transpose([np.repeat(x, len(y)), np.tile(y, len(x))])
        self.last_layer = len(self.deepzoom.level_tiles) - 1

    def __len__(self):
        return len(self.addresses)

    def get_tile(self, address):
        return np.asarray(self.deepzoom.get_tile(self.last_layer, address))

    def get_addresses(self):
        return self.addresses

