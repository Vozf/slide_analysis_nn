import os
from pathlib import Path

PROJECT_PATH = Path(os.path.abspath(os.path.dirname(__file__)))

EPOCHS = 30
BATCH_SIZE = 4
TRAIN_STEPS = None
VALIDATION_STEPS = None
MIN_DELTA = 1e-4
PATIENCE = 5
SNAPSHOTS_DIR = PROJECT_PATH / 'snapshots'
TF_BOARD_LOGS_DIR = PROJECT_PATH / 'logs'
