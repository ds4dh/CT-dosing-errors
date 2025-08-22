import os

REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

RESOURCES_DIR = os.path.join(REPO_DIR, "resources")

DATASETS_ROOT = os.environ.get("DATASETS_ROOT", None)
