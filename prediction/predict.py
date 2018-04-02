import glob
import os
import time
import keras
import numpy as np
import tensorflow as tf

from train.settings import (
    SNAPSHOTS_DIR,
    BATCH_SIZE)
from prediction.predict_generator import PredictGenerator


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
        self.model = keras.models.load_model(model_path)

    def _get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def predict_slide(self, slide_path):
        slide_generator = PredictGenerator(slide_path, batch_size=BATCH_SIZE)
        print('predict')
        start = time.time()
        scores = self.model.predict_generator(slide_generator)
        print(time.time() - start, 'sup')

        predicted_labels = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), predicted_labels]

        slide_generator.create_asap_annotations(predicted_labels, scores)

        return predicted_labels, scores

    @staticmethod
    def filter_models(filename):
        if '.h5' in filename:
            return filename


def main():
    predict_example = Predict()
    predicted_results = predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/small_with_tumor_images/Tumor_044.tif_62818:129066:70031:138983.tif')
    # predicted_results = predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/small_with_tumor_images/Tumor_044.tif_67170:143266:69380:146408.tif')


if __name__ == '__main__':
    main()
