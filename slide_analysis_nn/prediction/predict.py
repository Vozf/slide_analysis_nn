import glob
import os
from pathlib import Path

import cv2
import keras
import numpy as np
import tensorflow as tf

from slide_analysis_nn.prediction import PredictGenerator
from slide_analysis_nn.prediction import PredictionResult
from slide_analysis_nn.prediction.settings import MODEL_DOWNLOAD_URL
from slide_analysis_nn.train.settings import (
    SNAPSHOTS_DIR,
    BATCH_SIZE)
from slide_analysis_nn.utils.functions import download_file
from slide_analysis_nn.utils.types import Area_box


class Predict:
    def __init__(self, download_weights: bool = True, snapshot_path: Path = SNAPSHOTS_DIR):
        self.snapshot_path = snapshot_path
        self._get_session()
        keras.backend.tensorflow_backend.set_session(self.session)
        self._load_model(download_weights)

    def _load_model(self, download_weights: bool):
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        DOWNLOAD_MODEL_PATH = \
            self.snapshot_path / f'downloaded_model{os.path.basename(MODEL_DOWNLOAD_URL)}.h5'

        if download_weights and not os.path.isfile(DOWNLOAD_MODEL_PATH):
            print('Downloading weights')
            download_file(MODEL_DOWNLOAD_URL, DOWNLOAD_MODEL_PATH)

        files = glob.iglob(str(self.snapshot_path / '**' / '*.h5'), recursive=True)
        models = sorted(files, key=os.path.getmtime)
        model_path = str(self.snapshot_path / models[-1])
        print(model_path)
        self.model = keras.models.load_model(model_path)

    def _get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def predict_slide(self, slide_path: str, area_to_predict: Area_box = None, tqdm_enabled=True):
        slide_generator = PredictGenerator(slide_path,
                                           batch_size=BATCH_SIZE,
                                           area_to_predict=area_to_predict,
                                           tqdm_enabled=tqdm_enabled)

        scores = self.model.predict_generator(slide_generator)

        return PredictionResult(slide_path,
                                scores=scores,
                                tile_coordinates=slide_generator.all_coordinates)

    def predict_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[..., :3]

        scores = self.model.predict_on_batch(np.expand_dims(image, axis=0))

        predicted_labels = np.argmax(scores, axis=1)
        predicted_labels_scores = scores[np.arange(scores.shape[0]), predicted_labels]

        return PredictionResult(image_path,
                                scores=predicted_labels_scores)


def main():
    predict_example = Predict()
    prediction = predict_example \
        .predict_slide(
        'D:\\projects\\slide-analysis-nn\\slide_analysis_nn\\train\datasets\\source\\slide_images\\Tumor_041.tif',
        area_to_predict=(74000, 0, 80000, 3000))
    prediction.create_map()
    prediction.save_as_asap_annotations(
        truth_xml_path='D:\\projects\\slide-analysis-nn\\slide_analysis_nn\\train\datasets\\source\\slide_images\\Tumor_041.xml')

    # prediction.create_asap_annotations()


if __name__ == '__main__':
    main()
