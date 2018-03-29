import logging
import os
import re
import subprocess

import keras
import tensorflow
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from train.generator import Generator
from keras.utils import multi_gpu_model
from tensorflow.python.framework.graph_util import convert_variables_to_constants


from train.datasets_preparation.preparation import DatasetPreparation
from train.callbacks import (
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
from utils.constants import TILE_SIZE, TILE_SHAPE
from utils.mixins import GPUSupportMixin


class Train(GPUSupportMixin):
    def __init__(self):
        self.log = logging.getLogger('train')

        self.gpu_ids = self._get_available_gpus()
        self.set_up_gpu(self.gpu_ids)

        self.train_annotations = TRAIN_DATASET_FILE_PATH
        self.test_annotations = TEST_DATASET_FILE_PATH
        self.classes = CLASS_MAPPING_FILE_PATH
        self.epochs = EPOCHS
        self.batch_size = BATCH_SIZE

        if len(self.gpu_ids) > 1 and self.batch_size < len(self.gpu_ids):
            raise ValueError(
                "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(
                    self.batch_size,
                    len(self.gpu_ids))
            )

        self.snapshot_path = os.path.join(
            SNAPSHOTS_DIR, 'simple_train_{}'.format(len(os.listdir(SNAPSHOTS_DIR)))
        )
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)

    def _get_session(self):
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        return tensorflow.Session(config=config)

    def _create_generators(self):
        train_generator = Generator(
            self.train_annotations,
            self.classes,
            batch_size=self.batch_size,
        )

        validation_generator = Generator(
            self.test_annotations,
            self.classes,
            batch_size=self.batch_size,
        )

        return train_generator, validation_generator

    def _create_model(self, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                         input_shape=TILE_SHAPE))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        training_model = model
        prediction_model = model

        # compile model
        training_model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
        )

        return model, training_model, prediction_model

    def _create_callbacks(self, prediction_model, validation_generator):
        callbacks = []

        # save the prediction model
        checkpoint = BestModelCheckpoint(
            os.path.join(self.snapshot_path, 'slide_analysis_{epoch:02d}_{val_loss:.2f}.h5'),
            verbose=1, monitor='val_loss', save_best_only=True, mode='min'
        )
        callbacks.append(checkpoint)

        # Save the prediction model as tf_graph
        converter = TensorGraphConverter(
            prediction_model,
            os.path.join(self.snapshot_path),
        )
        callbacks.append(converter)

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA,
                                                       patience=PATIENCE)
        callbacks.append(early_stopping)

        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto',
            epsilon=0.0001, cooldown=0, min_lr=0
        )
        callbacks.append(lr_scheduler)

        tensor_board = TensorBoard(
            log_dir=os.path.join(TF_BOARD_LOGS_DIR,
                                 'train_{}'.format(len(os.listdir(TF_BOARD_LOGS_DIR)))
                                 )
        )
        callbacks.append(tensor_board)

        return callbacks

    def start_training(self):
        keras.backend.tensorflow_backend.set_session(self._get_session())

        train_generator, validation_generator = self._create_generators()

        train_steps = TRAIN_STEPS if TRAIN_STEPS is not None else len(train_generator)
        val_steps = VALIDATION_STEPS if VALIDATION_STEPS is not None else len(validation_generator)

        model, training_model, prediction_model = self._create_model(
            num_classes=train_generator.num_classes())

        self.log.info(model.summary())

        callbacks = self._create_callbacks(prediction_model, validation_generator)

        # start training
        training_model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
        )


def main():
    dataset_preparation = DatasetPreparation()
    dataset_preparation.populate_prepared_datasets()

    # train = Train()
    # train.start_training()


if __name__ == '__main__':
    main()
