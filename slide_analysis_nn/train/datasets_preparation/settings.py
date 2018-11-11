from slide_analysis_nn.train.settings import PROJECT_PATH

SOURCE_PATH = PROJECT_PATH / 'datasets' / 'source'

LABELED_IMAGES_DIR = SOURCE_PATH / 'labeled_images'
UNLABELED_IMAGES_DIR = SOURCE_PATH / 'unlabeled_images'

TRAIN_DATASET_PERCENT = 0.9

SLIDE_IMAGES_DIR = SOURCE_PATH / 'slide_images'
SMALL_WITH_TUMOR_IMAGES_DIR = SOURCE_PATH / 'small_with_tumor_images'

TEST_DATASET_FILE_PATH = PROJECT_PATH / 'datasets' / 'prepared_datasets' / 'test.csv'
TRAIN_DATASET_FILE_PATH = PROJECT_PATH / 'datasets' / 'prepared_datasets' / 'train.csv'
CLASS_MAPPING_FILE_PATH = PROJECT_PATH / 'datasets' / 'prepared_datasets' / 'class_mapping.csv'

BACKGROUND_CLASS_NAME = 'Background'
DEFAULT_CLASS_NAME = 'Tumor'
