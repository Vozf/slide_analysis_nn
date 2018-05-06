import glob
import os
import time

import cv2
import keras
import numpy as np
import tensorflow as tf

from prediction import PredictGenerator
from prediction import PredictionResult
from train.settings import (
    SNAPSHOTS_DIR,
    BATCH_SIZE)


class Predict:
    def __init__(self):
        self.snapshot_path = SNAPSHOTS_DIR
        self._get_session()
        keras.backend.tensorflow_backend.set_session(self.session)
        self._load_model()

    def _load_model(self):
        files = glob.iglob(self.snapshot_path+'/*/*.h5')
        models = sorted(files, key=os.path.getmtime)
        model_path = os.path.join(self.snapshot_path, models[-1])
        print(model_path)
        self.model = keras.models.load_model(model_path)

    def _get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def predict_slide(self, slide_path, area_to_predict=None):
        slide_generator = PredictGenerator(slide_path, batch_size=BATCH_SIZE,
                                           area_to_predict=area_to_predict)
        print('predict')
        start = time.time()
        scores = self.model.predict_generator(slide_generator)
        print(time.time() - start, 'sup')

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
    prediction = predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/hidden/Tumor_015.tif', area_to_predict=((74000, 74000), (80000, 80000)))
    prediction.create_map()
    # prediction.save_as_asap_annotations(truth_xml_path='/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/hidden/Tumor_015true.xml')

    # prediction.create_asap_annotations()

if __name__ == '__main__':
    main()
