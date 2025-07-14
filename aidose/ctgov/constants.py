from aidose import CORPORA_ROOT_DIR

import os

CTGOV_DATASET_PATH = os.path.join(CORPORA_ROOT_DIR, "CTGOV")
CTGOV_DATASET_RAW_PATH = os.path.join(CTGOV_DATASET_PATH, "raw")
CTGOV_DATASET_PROCESSED_PATH = os.path.join(CTGOV_DATASET_PATH, "processed")

os.makedirs(CTGOV_DATASET_RAW_PATH, exist_ok=True)
os.makedirs(CTGOV_DATASET_PROCESSED_PATH, exist_ok=True)
