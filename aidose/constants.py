import os
import socket

REPO_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]

RESOURCES_DIR = os.path.join(REPO_DIR, "resources")

datasets_root_dirs_dict = {
    "unige-poc": "/data/",
    "ordnance": "/data/bases/corpora/",
    "sf-macbook-pro.local": "/Users/sssohrab/data/corpora",
}

# TODO: Clean this mess by using environment variables.

hostname = socket.gethostname().lower()
if hostname in datasets_root_dirs_dict.keys():
    CORPORA_ROOT_DIR = datasets_root_dirs_dict[hostname]
else:
    CORPORA_ROOT_DIR = RESOURCES_DIR

if not os.path.exists(CORPORA_ROOT_DIR):
    os.makedirs(CORPORA_ROOT_DIR)
