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


# TODO: discuss this list -> leadSponsorName?
# LIST OF FEATURES TO DROP IN THE FINAL DATASET
LIST_OF_FEATURES_TO_DROP =["nctId", "studyType", "leadSponsorName", "overallStatus", "isJJ", "hasProtocol", "hasSap", "hasIcf", "num_ct_level_ade_terms", "ct_level_ade_population", "num_positive_terms_matched", "label", "completionDate", "startDate"]


ALPHA_WILSON = 0.05

WILSON_PROBA_THRESHOLD = 0.0001

TRAINING_SIZE = 0.7
VALIDATION_SIZE=0.15
TEST_SIZE = 0.15

