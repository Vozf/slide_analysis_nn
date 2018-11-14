import glob
import logging
import os

import keras
import tensorflow
from keras import Model

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import TensorBoard
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras_preprocessing.image import ImageDataGenerator

from slide_analysis_nn.train import Generator
from slide_analysis_nn.train.callbacks import BestModelCheckpoint
from slide_analysis_nn.train.callbacks import TB
from slide_analysis_nn.train.datasets_preparation import DatasetPreparation
from slide_analysis_nn.train.datasets_preparation.settings import (
    TRAIN_DIR_NAME,
    TEST_DIR_NAME
)
from slide_analysis_nn.train.settings import (
    SNAPSHOTS_DIR,
    EPOCHS,
    BATCH_SIZE,
    TRAIN_STEPS,
    VALIDATION_STEPS,
    TF_BOARD_LOGS_DIR,
    NETWORK_INPUT_SHAPE,
)
from slide_analysis_nn.utils.mixins import GPUSupportMixin


class Train(GPUSupportMixin):
    def __init__(self):
        self.log = logging.getLogger('train')

        self.gpu_ids = self._get_available_gpus()
        self.set_up_gpu(self.gpu_ids)

        if len(self.gpu_ids) > 1 and BATCH_SIZE < len(self.gpu_ids):
            raise ValueError(
                "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(
                    BATCH_SIZE,
                    len(self.gpu_ids))
            )

        self.snapshot_path = SNAPSHOTS_DIR / 'train_{}'.format(len(os.listdir(SNAPSHOTS_DIR)))

        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)

    def _get_session(self):
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        return tensorflow.Session(config=config)

    def _create_generators(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            TRAIN_DIR_NAME,
            target_size=NETWORK_INPUT_SHAPE[:2],
            shuffle=True,
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            TEST_DIR_NAME,
            target_size=NETWORK_INPUT_SHAPE[:2],
            shuffle=True,
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        return train_generator, validation_generator

    def _create_model(self, num_classes):
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=NETWORK_INPUT_SHAPE)

        # for layer in base_model.layers:
        #     layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # compile model
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=keras.optimizers.adam()
        )

        return model

    def _create_callbacks(self, prediction_model, validation_generator):
        callbacks = []

        # save the prediction model
        checkpoint = BestModelCheckpoint(
            str(self.snapshot_path / 'slide_analysis_{epoch:02d}_{val_acc:.2f}.h5'),
            verbose=1, monitor='val_acc', save_best_only=True, mode='max'
        )
        callbacks.append(checkpoint)

        # Save the prediction model as tf_graph
        # converter = TensorGraphConverter(
        #     prediction_model,
        #     self.snapshot_path,
        # )
        # callbacks.append(converter)

        # early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=MIN_DELTA,
        #                                                patience=PATIENCE)
        # callbacks.append(early_stopping)

        # lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        #     monitor='loss', factor=0.1, patience=2, verbose=1, mode='auto',
        #     epsilon=0.0001, cooldown=0, min_lr=0
        # )
        # callbacks.append(lr_scheduler)

        tensor_board = TensorBoard(log_dir=str(TF_BOARD_LOGS_DIR /
                                               f'train_{len(os.listdir(TF_BOARD_LOGS_DIR))}'))

        callbacks.append(tensor_board)

        return callbacks

    def _load_model(self):
        files = glob.iglob(str(self.snapshot_path / '**' / '*.h5'))
        models = sorted(files, key=os.path.getmtime)
        model_path = self.snapshot_path / models[-1]
        return keras.models.load_model(model_path)

    def start_training(self, continue_train=False):
        keras.backend.tensorflow_backend.set_session(self._get_session())

        train_generator, validation_generator = self._create_generators()

        train_steps = TRAIN_STEPS if TRAIN_STEPS is not None else len(train_generator)
        val_steps = VALIDATION_STEPS if VALIDATION_STEPS is not None else len(validation_generator)

        model = self._load_model() if continue_train else self._create_model(num_classes=2)

        self.log.info(model.summary())

        callbacks = self._create_callbacks(model, validation_generator)

        # start training
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
        )


def main():
    # dataset_preparation = DatasetPreparation()
    # dataset_preparation.populate_prepared_datasets()

    train = Train()
    train.start_training()


if __name__ == '__main__':
    main()
