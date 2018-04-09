import os

from train.settings import PROJECT_PATH

LABELED_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'labeled_images')

TRAIN_DATASET_PERCENT = 0.9

UNLABELED_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'unlabeled_images')

SLIDE_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'slide_images')
SMALL_WITH_TUMOR_IMAGES_DIR = os.path.join(PROJECT_PATH, 'datasets', 'source', 'small_with_tumor_images')


TEST_DATASET_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'test.csv')
TRAIN_DATASET_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'train.csv')
CLASS_MAPPING_FILE_PATH = os.path.join(PROJECT_PATH, 'datasets', 'prepared_datasets', 'class_mapping.csv')

BACKGROUND_CLASS_NAME = 'Background'
DEFAULT_CLASS_NAME = 'Tumor'
