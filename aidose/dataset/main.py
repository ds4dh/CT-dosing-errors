from aidose.ctgov.structures import Study

from aidose.meddra import MEDDRA_DATASET_PATH, MEDDRA_CREATED_ARTIFACTS_DIR
from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import parse_hlgt_codes_literal
from aidose.meddra.extraction import build_meddra_descendants

from aidose.dataset.ade import process_study_for_ade_risks
from aidose.dataset.utils import include_trial_after_sequential_filtering, has_protocol

from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
import aidose.ctgov.api_download as api_download

from typing import List

import os
import json
import tqdm

MEDDRA_LABELS_JSON_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")

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
    # -----------------------------
    # 0) Load MedDRA positive terms
    # -----------------------------
    if not os.path.exists(MEDDRA_LABELS_JSON_PATH):
        meddra = MedDRA()
        meddra.load_data(MEDDRA_DATASET_PATH)

        codes = parse_hlgt_codes_literal("[('HLGT', '10079145'), ('HLGT', '10079159')]")
        result = build_meddra_descendants(meddra, codes)
        meddra_labels = sorted(list(result.terms))
        with open(MEDDRA_LABELS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"terms": meddra_labels}, f, ensure_ascii=False, indent=4)

    else:
        with open(MEDDRA_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            meddra_labels = json.load(f).get("terms")

    # labels_header, labels_rows = meddra_labels_to_csv_rows(meddra_labels)
    # paths_header, paths_rows = meddra_paths_to_csv_rows(result.paths)

    # -----------------------------------
    # 1) Ensure dataset exists / is ready
    # -----------------------------------
    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
        api_download.main()

    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r', encoding='utf-8') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]
        nctids_list_filtered = filter_trials_list(nctids_list_all)
        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'w', encoding='utf-8') as f:
            for nctid in nctids_list_filtered:
                f.write(f"{nctid}\n")
    else:
        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, "r", encoding="utf-8") as f:
            nctids_list_filtered = [line.strip() for line in f if line.strip()]

    # -------------------------------------------------
    # 2) Per-study ADE processing + split pos / neg
    #    result dict structure (by our earlier design):
    #    {
    #      "study": Study,
    #      "ade_by_group": Dict[str, ADEGroupAggregate-like],
    #      "ade_clinical": Dict[str, {"numAffected": int, "numAtRisk": int}],
    #      "positive_terms": Dict[str, {"stats": {...}, "matches": [...]}}
    #    }
    # -------------------------------------------------
    positive_trials: list[dict] = []
    negative_trials: list[dict] = []
    errors: dict[str, int] = {}

    for nctid in tqdm.tqdm(nctids_list_filtered, desc="ADE matching per study"):
        file_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nctid}.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                study = Study.model_validate_json(f.read())
        except Exception:
            errors["File Load or Validation"] = errors.get("File Load or Validation", 0) + 1
            continue

        result, error = process_study_for_ade_risks(study, meddra_labels)

        if error:
            errors[error] = errors.get(error, 0) + 1
            continue

        if result.get("positive_terms"):
            positive_trials.append(result)
        else:
            negative_trials.append(result)

    print(f"Positive trials: {len(positive_trials)}")
    print(f"Negative trials: {len(negative_trials)}")
    print(f"Errors: {errors}")
