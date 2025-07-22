from aidose import CORPORA_ROOT_DIR, RESOURCES_DIR

import os

CTGOV_DATASET_PATH = os.path.join(CORPORA_ROOT_DIR, "CTGOV")
os.makedirs(CTGOV_DATASET_PATH, exist_ok=True)

CTGOV_DATASET_RAW_PATH = os.path.join(CTGOV_DATASET_PATH, "raw")

CTGOV_API_DOWNLOAD_BASE_URL = "https://clinicaltrials.gov/api/v2"

CTGOV_NCTIDS_LIST_ALL_PATH = os.path.join(RESOURCES_DIR, "CTGOV", "ctgov_nctids_list_all.txt")
