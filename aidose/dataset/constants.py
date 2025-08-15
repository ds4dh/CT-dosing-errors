from aidose import RESOURCES_DIR
from aidose.meddra import MEDDRA_DATASET_PATH, MEDDRA_CREATED_ARTIFACTS_DIR
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH

import os

MEDDRA_LABELS_JSON_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")
MEDDRA_HLGT_CODES_LITERAL = "[('HLGT', '10079145'), ('HLGT', '10079159')]"

CTGOV_NCTIDS_LIST_FILTERED_PATH = os.path.join(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH),
                                               "ctgov_nctids_list_filtered.txt")

ADE_ANALYSIS_RESULTS_PATH = os.path.join(RESOURCES_DIR, "ade_analysis_results.json")
END_POINT_HF_DATASET_PATH = os.path.join(RESOURCES_DIR, "dataset")
