import os

REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

RESOURCES_DIR = os.path.join(REPO_DIR, "resources")

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", None)

if DATASETS_ROOT is not None:
    os.makedirs(DATASETS_ROOT, exist_ok=True)

PACKAGE_NAME = "aidose"

REPO_HOMEPAGE_URL = "https://github.com/ds4dh/CT-dosing-errors"
