import logging
import os
import re
import subprocess

import keras
import tensorflow
from keras.utils import multi_gpu_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from keras_retinanet import layers
from keras_retinanet import losses
from keras_retinanet.callbacks import RedirectModel
from keras_retinanet.callbacks.eval import Evaluate
from keras_retinanet.models.resnet import resnet50_retinanet
from keras_retinanet.preprocessing.csv_generator import CSVGenerator
from keras_retinanet.utils.keras_version import check_keras_version
from keras_retinanet.utils.transform import random_transform_generator

from train.datasets_preparation.preparation import DatasetPreparation
from train.callbacks import (
    TensorBoardRetinaCallback,
    TensorGraphConverter,
    BestModelCheckpoint,
)
from train.datasets_preparation.settings import (
    CLASS_MAPPING_FILE_PATH,
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH
)
from train.settings import (
    SNAPSHOTS_DIR,
    EPOCHS,
    BATCH_SIZE,
    TRAIN_STEPS,
    VALIDATION_STEPS,
    TF_BOARD_LOGS_DIR,
    MIN_DELTA,
    PATIENCE
)
from utils.constants import TILE_SIZE
from utils.mixins import GPUSupportMixin


class Train(GPUSupportMixin):
    def __init__(self):
        check_keras_version()

        self.log = logging.getLogger('train')

        self.gpu_ids = self._get_available_gpus()
        self.set_up_gpu(self.gpu_ids)

        self.train_annotations = TRAIN_DATASET_FILE_PATH
        self.test_annotations = TEST_DATASET_FILE_PATH
        self.classes = CLASS_MAPPING_FILE_PATH
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE if len(self.gpu_ids) < 2 else len(self.gpu_ids)

        if len(self.gpu_ids) > 1 and self.batch_size < len(self.gpu_ids):
            raise ValueError(
                "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(
                    self.batch_size,
                    len(self.gpu_ids))
            )

        self.snapshot_path = os.path.join(
            SNAPSHOTS_DIR, 'train_{}'.format(len(os.listdir(SNAPSHOTS_DIR)))
        )
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)

    def _get_session(self):
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        return tensorflow.Session(config=config)

    def _create_generators(self):
        transform_generator = random_transform_generator(flip_x_chance=0.5)

        train_generator = CSVGenerator(
            self.train_annotations,
            self.classes,
            transform_generator=transform_generator,
            batch_size=self.batch_size,
            image_min_side=TILE_SIZE
        )

        validation_generator = CSVGenerator(
            self.test_annotations,
            self.classes,
            batch_size=self.batch_size,
            image_min_side=TILE_SIZE
        )

        return train_generator, validation_generator

    def _create_model(self, num_classes, weights='imagenet'):
        if len(self.gpu_ids) > 1:
            with tensorflow.device('/cpu:0'):
                model = resnet50_retinanet(num_classes, weights=weights, nms=False)
            training_model = multi_gpu_model(model, gpus=len(self.gpu_ids))

            # append NMS for prediction only
            classification = model.outputs[1]
            detections = model.outputs[2]
            boxes = keras.layers.Lambda(lambda x: x[:, :, :4])(detections)
            detections = layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])
            prediction_model = keras.models.Model(inputs=model.inputs, outputs=model.outputs[:2] + [detections])
        else:
            model = resnet50_retinanet(num_classes, weights=weights, nms=True)
            training_model = model
            prediction_model = model

        # compile model
        training_model.compile(
            loss={
                'regression': losses.smooth_l1(),
                'classification': losses.focal(),
            },
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        )

        return model, training_model, prediction_model

    def _create_callbacks(self, prediction_model, validation_generator):
        callbacks = []

        # save the prediction model
        checkpoint = BestModelCheckpoint(
            os.path.join(self.snapshot_path, 'logo_recognition_{epoch:02d}_{val_loss:.2f}.h5'),
            verbose=1, monitor='val_loss', save_best_only=True, mode='min'
        )
        checkpoint = RedirectModel(checkpoint, prediction_model)
        callbacks.append(checkpoint)

        # Save the prediction model as tf_graph
        converter = TensorGraphConverter(
            prediction_model,
            os.path.join(self.snapshot_path),
        )
        callbacks.append(converter)

        evaluation = Evaluate(validation_generator)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA,
                                                       patience=PATIENCE)
        callbacks.append(early_stopping)

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto',
            epsilon=0.0001, cooldown=0, min_lr=0
        )
        callbacks.append(lr_scheduler)

        tensor_board = TensorBoardRetinaCallback(
            write_batch_performance=True,
            log_dir=os.path.join(
                TF_BOARD_LOGS_DIR,
                'train_{}'.format(len(os.listdir(TF_BOARD_LOGS_DIR)))
            )
        )
        callbacks.append(tensor_board)

        return callbacks

    def start_training(self):
        keras.backend.tensorflow_backend.set_session(self._get_session())

        train_generator, validation_generator = self._create_generators()

        train_steps = TRAIN_STEPS if TRAIN_STEPS is not None\
            else train_generator.size() // self.batch_size
        validation_steps = VALIDATION_STEPS if VALIDATION_STEPS is not None\
            else validation_generator.size() // self.batch_size

        model, training_model, prediction_model = self._create_model(num_classes=train_generator.num_classes())

        self.log.info(model.summary())

        callbacks = self._create_callbacks(prediction_model, validation_generator)

        # Counting CSV-entries
        self.steps = 0
        for v in train_generator.image_data.values():
            self.steps += len(v) if v else 1
        self.steps = self.steps // self.batch_size

        # start training
        training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            verbose=1,
            callbacks=callbacks,
        )


def main():
    # dataset_preparation = DatasetPreparation()
    # dataset_preparation.populate_prepared_datasets()
    #
    train = Train()
    train.start_training()


if __name__ == '__main__':
    main()
