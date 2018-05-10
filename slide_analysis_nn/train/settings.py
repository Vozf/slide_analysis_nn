import os

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

EPOCHS = 200
BATCH_SIZE = 2
TRAIN_STEPS = 3
VALIDATION_STEPS = None
MIN_DELTA = 1e-4
PATIENCE = 5
SNAPSHOTS_DIR = os.path.join(PROJECT_PATH, 'snapshots')
TF_BOARD_LOGS_DIR = os.path.join(PROJECT_PATH, 'logs')
