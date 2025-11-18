from aidose import DATASETS_ROOT, RESOURCES_DIR

import os

if DATASETS_ROOT is not None and os.path.exists(DATASETS_ROOT):
    CTGOV_DATASET_PATH = os.path.join(DATASETS_ROOT, "CTGOV")
    os.makedirs(CTGOV_DATASET_PATH, exist_ok=True)
else:
    CTGOV_DATASET_PATH = os.path.join(RESOURCES_DIR, "CTGOV")
    os.makedirs(CTGOV_DATASET_PATH, exist_ok=True)

CTGOV_DATASET_RAW_PATH = os.path.join(CTGOV_DATASET_PATH, "raw")  # TODO: Rename
CTGOV_DATASET_EXTENSIONS_PATH = os.path.join(CTGOV_DATASET_PATH, "extensions")  # TODO: Rename
CTGOV_EXTRACTED_PDFS_DATASET_PATH = os.path.join(CTGOV_DATASET_EXTENSIONS_PATH, "EXTRACTED_CTGOV_PDFS")

EXTRACT_PDFS_USING_DEEPSEEK_OCR = False

CTGOV_API_DOWNLOAD_BASE_URL = "https://clinicaltrials.gov/api/v2"

CTGOV_NCTIDS_LIST_ALL_PATH = os.path.join(RESOURCES_DIR, "CTGOV", "ctgov_nctids_list_all.txt")
CTGOV_PROTOCOL_PDF_LINKS_PATH = os.path.join(RESOURCES_DIR, "CTGOV","ctgov_protocol_pdfs_links.json")
