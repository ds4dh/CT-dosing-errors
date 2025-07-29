from aidose.dataset.utils import include_trial_after_sequential_filtering

from aidose.ctgov.structures import Study

from typing import List

import os
import json
import tqdm

from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
import aidose.ctgov.api_download as api_download

CTGOV_NCTIDS_LIST_FILTERED_PATH = os.path.join(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH),
                                               "ctgov_nctids_list_filtered.txt")


def filter_trials_list(ids_list: list[str]) -> list[str]:
    ids_list_filtered: List[str] = []

    for nct_id in tqdm.tqdm(ids_list, desc="Parsing trials and filtering them."):
        json_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nct_id}.json")

        with open(json_path, 'r') as f:
            study_data = json.load(f)

        study = Study(**study_data)

        if include_trial_after_sequential_filtering(study):
            ids_list_filtered.append(nct_id)

    return ids_list_filtered


if __name__ == '__main__':
    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
        api_download.main()
    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]

        nctids_list_filtered = filter_trials_list(nctids_list_all)

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'w') as f:
            for nctid in nctids_list_filtered:
                f.write(f"{nctid}\n")
