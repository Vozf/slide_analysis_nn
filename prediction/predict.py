import glob
import logging
import os
import time

import cv2
import keras
# TODO: solve error with python3-tk library
# import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from prediction.errors import ModelError, ImageError, PredictionResultError
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import preprocess_image, resize_image, read_image_bgr

from train.settings import (
    SNAPSHOTS_DIR,
    BATCH_SIZE)
from prediction.settings import (
    LABELS,
    BRG_IMAGE_FORMAT,
)
from utils.result import Result
from utils.slide import Slide
from utils.slide_tile_generator import SlideTileGenerator


class Predict():
    def __init__(self):
        self.snapshot_path = SNAPSHOTS_DIR
        self.labels = LABELS
        self._get_session()
        keras.backend.tensorflow_backend.set_session(self.session)
        self._load_model()

    def _load_model(self):
        try:
            files = glob.iglob(self.snapshot_path+'/**', recursive=True)
            models = sorted(filter(lambda x: Predict.filter_models(x), files))
            model_path = os.path.join(self.snapshot_path, models[len(models) - 1])
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
        except (IndexError, OSError) as error:
            print('Cannot load model: {0}'.format(error))
            raise ModelError

    def _get_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

    def predict_slide(self, slide_path):
        slide_generator = SlideTileGenerator(slide_path, batch_size=BATCH_SIZE)
        print('predict')
        start = time.time()
        _, _, detections = self.model.predict_generator(slide_generator, steps=4)
        print(time.time() - start, 'sup')


    def predict_tile(self, image):
        if not isinstance(image, np.ndarray):
            print('Image is wrong')
            raise ImageError

        initial_image = image

        image = preprocess_image(image)
        image, scale = resize_image(image)

        start = time.time()
        _, _, detections = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
        scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]
        detections[0, :, :4] /= scale

        return Result(image=initial_image, predicted_labels=predicted_labels, scores=scores,
                      detections=detections)

    def visualise(self, predicted_result, image_format=BRG_IMAGE_FORMAT):
        if not isinstance(predicted_result, Result):
            print('Cannot visualise results: incorrect type of prediction results')
            raise PredictionResultError

        draw = predicted_result.image.copy()
        if image_format == BRG_IMAGE_FORMAT:
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        for idx, (label, score) in enumerate(zip(predicted_result.predicted_labels, predicted_result.scores)):
            if score < 0.5:
                continue
            b = predicted_result.detections[0, idx, :4].astype(int)
            cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            caption = "{} {:.3f}".format(predicted_result.predicted_labels[label], score)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
            cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
        return draw

    # TODO: solve error with python3-tk library
    # @staticmethod
    # def show(image):
    #     if isinstance(image, np.ndarray):
    #         plt.figure(figsize=(15, 15))
    #         plt.axis('off')
    #         plt.imshow(image)
    #         plt.show()

    @staticmethod
    def filter_models(filename):
        if '.h5' in filename:
            return filename


def main():
    predict_example = Predict()
    predict_example.predict_slide('/home/vozman/projects/slides/slide-analysis-nn/train/datasets/source/slide_images/Tumor_001.tif')
    # image = read_image_bgr('path/to/image')
    # predicted_result = predict_example.predict_tile(image)
    # final_image = predict_example.visualise(predicted_result, BRG_IMAGE_FORMAT)
    # TODO: solve error with python3-tk library
    # Predict.show(final_image)


if __name__ == '__main__':
    main()
