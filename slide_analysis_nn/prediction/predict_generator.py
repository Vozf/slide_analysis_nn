import keras
import numpy as np
import tqdm
from keras_preprocessing.image import ImageDataGenerator

from slide_analysis_nn.prediction.settings import TILE_STEP
from slide_analysis_nn.train.datasets_preparation import DatasetPreparation
from slide_analysis_nn.tile import TILE_SIZE
from slide_analysis_nn.train.settings import NETWORK_INPUT_SHAPE
from slide_analysis_nn.utils.slide import Slide
from slide_analysis_nn.utils.types import Area_box


class PredictGenerator(keras.utils.Sequence):
    def __init__(self, slide_path: str, batch_size: int, area_to_predict: Area_box = None,
                 tqdm_enabled=True):
        self.batch_size = batch_size
        self.slide = Slide(slide_path)

        self.tqdm_enabled = tqdm_enabled
        self.tqdm = None

        self.area_to_predict = area_to_predict if area_to_predict else (
            0, 0, self.slide.slide.dimensions[0], self.slide.slide.dimensions[1])

        self.datagen_for_standartize = ImageDataGenerator(
            samplewise_center=True,
            samplewise_std_normalization=True,
        )

        x = range(self.area_to_predict[0], self.area_to_predict[2] - TILE_SIZE, TILE_STEP)
        y = range(self.area_to_predict[1], self.area_to_predict[3] - TILE_SIZE, TILE_STEP)
        self.coordinates_grid = np.stack(np.meshgrid(x, y), axis=2)

        self.all_coordinates = self.coordinates_grid.reshape(-1, 2)
        self.label_names_to_id = DatasetPreparation.get_label_name_to_label_id_dict()

    def __len__(self):
        return int(np.ceil(len(self.all_coordinates) / self.batch_size))

    def __getitem__(self, index):
        if self.tqdm_enabled:
            self.update_tqdm(index)

        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop, index.step)]

        addresses = self.all_coordinates[index * self.batch_size:(index + 1) * self.batch_size]

        X = self.__data_generation(addresses)

        return X, np.zeros(len(X))

    def update_tqdm(self, index):
        if index == 0:
            self.tqdm = tqdm.tqdm(total=len(self))

        self.tqdm.update(1)

        if index == len(self) - 1:
            self.tqdm.close()

    def __data_generation(self, addresses: np.ndarray) -> np.ndarray:
        data = np.asarray(
            [np.asarray(self.slide.cut_tile(*add).resize(NETWORK_INPUT_SHAPE[:2]).convert('RGB'))
             for add in addresses]).astype(np.float)

        return self.datagen_for_standartize.standardize(data)
