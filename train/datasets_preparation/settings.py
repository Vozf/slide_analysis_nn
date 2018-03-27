import os

from train.settings import PROJECT_PATH

AUGMENTATION_PERCENT = 0.5
AUGMENTED_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'augmented_images')

LABELED_IMAGES_TRAIN_PERCENT = 0.76
LABELED_IMAGES_TEST_PERCENT = 1 - LABELED_IMAGES_TRAIN_PERCENT
LABELED_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'labeled_images')

IMAGES_WITHOUT_BRAND_PERCENT = 0.2

UNLABELED_IMAGES_TRAIN_PERCENT = 0.7
UNLABELED_IMAGES_TEST_PERCENT = 1 - UNLABELED_IMAGES_TRAIN_PERCENT
UNLABELED_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'unlabeled_images')

SLIDE_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'slide_images')
SMALL_WITH_TUMOR_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'small_with_tumor_images')


TEST_DATASET_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'test.csv')
TRAIN_DATASET_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'train.csv')
CLASS_MAPPING_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'class_mapping.csv')

DEFAULT_CLASS_NAME = 'Tumor'
