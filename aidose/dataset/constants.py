from aidose import RESOURCES_DIR, DATASETS_ROOT
from aidose.meddra import MEDDRA_CREATED_ARTIFACTS_DIR
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH

from datetime import datetime
import os

DATASET_NAME = "CT-DOSING-ERRORS"
DATASET_VERSION = "0.2.2"
ENUM_FIELDS_EXPANSION = False  # Whether to expand categorical fields as one-hot or multi-hot encoding.

if DATASETS_ROOT is not None and os.path.exists(DATASETS_ROOT):
    END_POINT_HF_DATASET_PATH = os.path.join(DATASETS_ROOT, DATASET_NAME, DATASET_VERSION)
    os.makedirs(END_POINT_HF_DATASET_PATH, exist_ok=True)
else:
    END_POINT_HF_DATASET_PATH = os.path.join(RESOURCES_DIR, DATASET_NAME, DATASET_VERSION)
    os.makedirs(END_POINT_HF_DATASET_PATH, exist_ok=True)

MEDDRA_ADE_LABELS_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")
MEDDRA_HLGT_CODES_LITERAL = "[('HLGT', '10079145'), ('HLGT', '10079159')]"

CTGOV_NCTIDS_LIST_FILTERED_PATH = os.path.join(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH),
                                               "ctgov_nctids_list_filtered.txt")

ADE_ANALYSIS_RESULTS_PATH = os.path.join(RESOURCES_DIR, "ade_analysis_results.json")

CTGOV_KNOWLEDGE_CUTOFF_DATE: datetime | None = datetime(year=2025, month=9, day=2)

ALPHA_WILSON = 0.05

WILSON_PROBA_THRESHOLD = 0.0001

TRAINING_SIZE = 0.7
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
