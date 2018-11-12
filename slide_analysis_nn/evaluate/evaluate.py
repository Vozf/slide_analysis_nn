import glob
import os

import keras
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, accuracy_score

from slide_analysis_nn.train.callbacks import auc_roc
from slide_analysis_nn.train.datasets_preparation.settings import TEST_DIR_NAME, TRAIN_DIR_NAME
from slide_analysis_nn.train.settings import SNAPSHOTS_DIR, NETWORK_INPUT_SHAPE, BATCH_SIZE


class Evaluate:
    def __init__(self, model_path=None):
        if not model_path:
            files = glob.iglob(str(SNAPSHOTS_DIR / '**' / '*.h5'))
            models = sorted(files, key=os.path.getmtime)
            model_path = SNAPSHOTS_DIR / models[-1]

        self.model = keras.models.load_model(str(model_path), custom_objects={'auc_roc': auc_roc})

    def evaluate(self, images_path):
        generator = self._get_generator(images_path)
        y_true = generator.classes
        y_pred = self.model.predict_generator(generator, steps=len(generator))
        y_pred = np.argmax(y_pred, axis=1)

        print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
        print(f'ROC_AUC: {roc_auc_score(y_true, y_pred)}')
        print(f'F1: {f1_score(y_true, y_pred)}')
        print(f'Confusion Matrix: \n {confusion_matrix(y_true, y_pred)}')

    def _get_generator(self, images_path):
        datagen = ImageDataGenerator(
            rescale=1. / 255,
        )
        generator = datagen.flow_from_directory(
            images_path,
            target_size=NETWORK_INPUT_SHAPE[:2],
            shuffle=False,
            batch_size=BATCH_SIZE,
            class_mode='categorical',

        )
        return generator


if __name__ == '__main__':
    evaluate = Evaluate()

    print('-' * 50)
    print('Train')
    print('-' * 50)
    evaluate.evaluate(TRAIN_DIR_NAME)
    print('-' * 50)
    print('Train')
    print('-' * 50)
    evaluate.evaluate(TEST_DIR_NAME)
