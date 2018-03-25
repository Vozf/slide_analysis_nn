import keras
from keras import backend as K
import numpy as np
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet import losses
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from train.datasets_preparation.settings import (
    TEST_DATASET_FILE_PATH,
    CLASS_MAPPING_FILE_PATH
)
from train.losses.custom import custom_loss

from train.settings import BATCH_SIZE


def evaluate_focal(model_path, loss=losses.focal(), csv_path=TEST_DATASET_FILE_PATH):
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    generator = CSVGenerator(
        csv_path,
        CLASS_MAPPING_FILE_PATH,
        batch_size=BATCH_SIZE
    )
    print(generator.size())

    generator_next = generator.next()
    y_true = generator_next[1][1]
    a, y_pred, detections = model.predict_on_batch(generator_next[0])
    np.zeros(1)

    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)

    return K.eval(loss(y_true, y_pred))


def evaluate_custom(model_path, loss=custom_loss, csv_path=TEST_DATASET_FILE_PATH):
    model = keras.models.load_model(model_path, custom_objects=custom_objects)

    generator = CSVGenerator(
        csv_path,
        CLASS_MAPPING_FILE_PATH,
        BATCH_SIZE
    )

    generator_next = generator.next()
    y_true = generator_next[1][1]
    a, y_pred, detections = model.predict_on_batch([generator_next[0]])
    np.zeros(1)

    print(y_true.shape)
    print(y_pred.shape)
    y_true = y_true[:, :, 0:4]
    # y_pred = np.concatenate((y_pred, b), axis=2)

    print(y_true.shape)
    print(y_pred.shape)

    y_pred = K.variable(y_pred)
    y_true = K.variable(y_true)

    return K.eval(loss(y_true, y_pred))


loss = evaluate_focal(
    '/home/vozman/projects/CPGLogoRecognition/train/snapshots/logo_recognition_04.h5')

print(loss)
