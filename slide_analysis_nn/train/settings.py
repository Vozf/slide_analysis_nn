import os
from pathlib import Path

PROJECT_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

TRAIN_TEST_DATASET_PERCENT = 0.7
AREA_PROCESSING_MULTIPLIER = 2
AREA_TO_INTERSECT_MULTIPLIER = 0.5
MAX_TILES_PER_TUMOR = 2000

EPOCHS = 30
BATCH_SIZE = 128
TRAIN_STEPS = None
VALIDATION_STEPS = None
MIN_DELTA = 1e-4
PATIENCE = 5

NETWORK_INPUT_SHAPE = (224, 224, 3)

SNAPSHOTS_DIR = PROJECT_PATH / 'snapshots'
TF_BOARD_LOGS_DIR = PROJECT_PATH / 'logs'
