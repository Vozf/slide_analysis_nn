import glob
import logging
import os

from comet_ml import Experiment
import keras
import tensorflow
from keras.applications.inception_v3 import InceptionV3

from train.datasets_preparation.settings import (
    TRAIN_DATASET_FILE_PATH,
    TEST_DATASET_FILE_PATH,
)
from train.generator import Generator
from train.settings import (
    SNAPSHOTS_DIR,
    EPOCHS,
    BATCH_SIZE,
    TRAIN_STEPS,
    VALIDATION_STEPS,
    COMET_ML_API_KEY,
)
from utils.mixins import GPUSupportMixin


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
        train_generator = Generator(
            TRAIN_DATASET_FILE_PATH,
            batch_size=BATCH_SIZE,
        )

        validation_generator = Generator(
            TEST_DATASET_FILE_PATH,
            batch_size=BATCH_SIZE,
        )

        return train_generator, validation_generator

    def _create_model(self, num_classes):
        model = InceptionV3(include_top=True, weights=None, classes=num_classes)

        # compile model
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer=keras.optimizers.adam(lr=1e-4, clipnorm=0.001)
        )

        return model

    def _create_callbacks(self, prediction_model, validation_generator):
        callbacks = []

        # save the prediction model
        checkpoint = BestModelCheckpoint(
            os.path.join(self.snapshot_path, 'slide_analysis_{epoch:02d}_{val_loss:.2f}.h5'),
            verbose=1,
            # monitor='val_loss', save_best_only=True, mode='min'
        )
        callbacks.append(checkpoint)

        # Save the prediction model as tf_graph
        # converter = TensorGraphConverter(
        #     prediction_model,
        #     os.path.join(self.snapshot_path),
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

        return callbacks

    def _connect_to_comet_ml(self):
        Experiment(api_key=COMET_ML_API_KEY)

    def _load_model(self):
        files = glob.iglob(self.snapshot_path + '/../*/*.h5')
        models = sorted(files, key=os.path.getmtime)
        model_path = os.path.join(self.snapshot_path, models[-1])
        return keras.models.load_model(model_path)

    def start_training(self, continue_train=False):
        keras.backend.tensorflow_backend.set_session(self._get_session())

        self._connect_to_comet_ml()

        train_generator, validation_generator = self._create_generators()

        train_steps = TRAIN_STEPS if TRAIN_STEPS is not None else len(train_generator)
        val_steps = VALIDATION_STEPS if VALIDATION_STEPS is not None else len(validation_generator)

        model = self._load_model() if continue_train else self._create_model(
            num_classes=train_generator.num_classes())

        self.log.info(model.summary())

        callbacks = self._create_callbacks(model, validation_generator)

        # start training
        model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            epochs=EPOCHS,
            class_weight={0: 2, 1: 1},
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
        )


def main():
    dataset_preparation = DatasetPreparation()
    dataset_preparation.populate_prepared_datasets()
    #
    train = Train()
    train.start_training()


if __name__ == '__main__':
    main()
