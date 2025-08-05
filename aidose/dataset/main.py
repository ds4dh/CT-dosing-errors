from aidose.dataset.ade import process_study_for_ade_risks
from aidose.dataset.meddra import MEDDRA_LABELS_JSON_PATH
from aidose.dataset.utils import include_trial_after_sequential_filtering, has_protocol

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
    with open(MEDDRA_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
        meddra_labels = json.load(f).get("terms")

    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
        api_download.main()
    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]

        nctids_list_filtered = filter_trials_list(nctids_list_all)

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'w') as f:
            for nctid in nctids_list_filtered:
                f.write(f"{nctid}\n")

    else:
        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, "r", encoding="utf-8") as f:
            nctids_list_filtered = [line.strip() for line in f.readlines()]

    positive_trials = []
    negative_trials = []
    errors = {}

    for nctid in tqdm.tqdm(nctids_list_filtered, desc="Getting positive and negative trials..."):
        file_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nctid}.json")

        with open(file_path, "r", encoding="utf-8") as f:
            study = Study.model_validate_json(f.read())

        try:
            result, error = process_study_for_ade_risks(study, meddra_labels)

            if error:
                errors[error] = errors.get(error, 0) + 1
            elif result["positive_terms"]:
                positive_trials.append(result)
            else:
                negative_trials.append(result)

        except Exception as e:
            errors["File Load or Validation"] = errors.get("File Load or Validation", 0) + 1
            continue

    # TODO: What to do with these? Do we save them to a file?
    print(len(positive_trials))
    print(len(negative_trials))
    print(errors)

    # TODO: We don't need this other than for stats, if I correctly understand the original code:
    positive_trials_with_protocol = [study for study in positive_trials if has_protocol(study)]
    negative_trials_with_protocol = [study for study in negative_trials if has_protocol(study)]
