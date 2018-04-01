import glob
import os
import time

import cv2
import keras
# TODO: solve error with python3-tk library
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prediction.errors import ModelError, ImageError, PredictionResultError

from train.settings import (
    SNAPSHOTS_DIR,
    BATCH_SIZE)
from prediction.settings import (
    LABELS,
    BRG_IMAGE_FORMAT,
)
from utils.result import Result
from prediction.predict_generator import PredictGenerator


class Predict:
    def __init__(self):
        self.snapshot_path = SNAPSHOTS_DIR
        self.labels = LABELS
        self._get_session()
        keras.backend.tensorflow_backend.set_session(self.session)
        self._load_model()

    def _load_model(self):
        try:
            files = glob.iglob(self.snapshot_path+'/**', recursive=True)
            models = sorted(filter(lambda x: Predict.filter_models(x), files), key=os.path.getmtime)
            model_path = os.path.join(self.snapshot_path, models[- 1])
            self.model = keras.models.load_model(model_path)
        except (IndexError, OSError) as error:
            print('Cannot load model: {0}'.format(error))
            raise ModelError

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

    def visualise(self, predicted_result, image_format=BRG_IMAGE_FORMAT):
        if not isinstance(predicted_result, Result):
            print('Cannot visualise results: incorrect type of prediction results')
            raise PredictionResultError

        draw = predicted_result.image.copy()
        # if image_format == BRG_IMAGE_FORMAT:
        #     draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        labels = np.argmax(predicted_result.detections[:, 4:], axis=1)
        scores = predicted_result.detections[
            np.arange(predicted_result.detections.shape[0]), 4 + labels]

        for idx, (label, score) in enumerate(zip(labels, scores)):
            if score < 0.5:
                continue

            b = predicted_result.detections[idx, :4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            caption = "{} {:.3f}".format(labels[label], score)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        return draw

    # TODO: solve error with python3-tk library
    @staticmethod
    def show(image):
        if isinstance(image, np.ndarray):
            plt.figure(figsize=(15, 15))
            plt.axis('off')
            plt.imshow(image)
            plt.show()

    @staticmethod
    def filter_models(filename):
        if '.h5' in filename:
            return filename


def main():
    predict_example = Predict()
    # predicted_results = predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/Tumor_016.tif')
    predicted_results = predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/small_with_tumor_images/Tumor_044.tif_67170:143266:69380:146408.tif')
    # image = read_image_bgr('path/to/image')
    # predicted_result = predict_example.predict_tile(image)
    # final_images = [predict_example.visualise(predicted_result, BRG_IMAGE_FORMAT) for predicted_result in predicted_results]
    # final_image = predict_example.visualise(predicted_result, BRG_IMAGE_FORMAT)
    # TODO: solve error with python3-tk library
    # list(map(Predict.show, final_images))


if __name__ == '__main__':
    main()
