from aidose import RESOURCES_DIR

import os

MEDDRA_VERSION = "27_1_English"

MEDDRA_DATASET_PATH = os.path.join(RESOURCES_DIR, "MEDDRA", "MedDRA_{}".format(MEDDRA_VERSION), "MedAscii")

if not os.path.exists(MEDDRA_DATASET_PATH):
    raise FileNotFoundError(
        "Please download the MedDRA dataset from their website and place it under {}. "
        "This requires you to create an account and agree to their terms.".format(
            os.path.dirname(MEDDRA_DATASET_PATH)))

MEDDRA_CREATED_ARTIFACTS_DIR = os.path.join(RESOURCES_DIR, "MEDDRA", "created_artifacts")
os.makedirs(MEDDRA_CREATED_ARTIFACTS_DIR, exist_ok=True)
