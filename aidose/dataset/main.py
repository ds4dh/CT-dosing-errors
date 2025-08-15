from aidose.meddra import MEDDRA_DATASET_PATH
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
from aidose.dataset import (
    MEDDRA_LABELS_JSON_PATH,
    CTGOV_NCTIDS_LIST_FILTERED_PATH,
    ADE_ANALYSIS_RESULTS_PATH,
    END_POINT_HF_DATASET_PATH
)

from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import parse_hlgt_codes_literal
from aidose.meddra.extraction import build_meddra_descendants

from aidose.ctgov.structures import Study
import aidose.ctgov.api_download as api_download

from aidose.dataset.utils import include_trial_after_sequential_filtering

from aidose.dataset.ade import process_study_for_ade_risks
from aidose.dataset.ade import ADEAnalysisResultForStudy

from aidose.dataset.ade_labeling import canonical_labels_from_positive_terms

from aidose.dataset.feature_extraction import FeaturesList, extract_features_for_study

from datasets import Dataset, Features, Value

from typing import List, Dict

import os
import json
import tqdm


def main():
    # -----------------------------
    # 0) MedDRA positive terms
    # -----------------------------
    meddra_labels: List[str] = []
    if not os.path.exists(MEDDRA_LABELS_JSON_PATH):
        meddra = MedDRA()
        meddra.load_data(MEDDRA_DATASET_PATH)

        codes = parse_hlgt_codes_literal("[('HLGT', '10079145'), ('HLGT', '10079159')]")  # TODO: What was this?
        ade_analysis_result = build_meddra_descendants(meddra, codes)
        meddra_labels = sorted(list(ade_analysis_result.terms))
        with open(MEDDRA_LABELS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"terms": meddra_labels}, f, ensure_ascii=False, indent=4)

    else:
        with open(MEDDRA_LABELS_JSON_PATH, "r", encoding="utf-8") as f:
            meddra_labels = json.load(f).get("terms")

    # -----------------------------------
    # 1) CTGov download and filtering
    # -----------------------------------
    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
        api_download.main()

    nctids_list_filtered: List[str] = []
    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r', encoding='utf-8') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]

        for nct_id in tqdm.tqdm(nctids_list_all, desc="Parsing trials and filtering them."):
            json_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nct_id}.json")

            with open(json_path, 'r') as f:
                study_data = json.load(f)

            study = Study(**study_data)

            if include_trial_after_sequential_filtering(study):
                nctids_list_filtered.append(nct_id)

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'w', encoding='utf-8') as f:
            for nctid in nctids_list_filtered:
                f.write(f"{nctid}\n")
    else:
        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, "r", encoding="utf-8") as f:
            nctids_list_filtered = [line.strip() for line in f if line.strip()]

    # -------------------------------------------------
    # 2) Per-study ADE processing + split pos / neg
    #    ADEAnalysisResultForStudy:
    #      - nctId: str
    #      - ade_by_group: Dict[str, ADEGroupAggregate]
    #      - ade_clinical: Dict[str, ADEClinicalTermStats]
    #      - positive_terms: Dict[str, PositiveTermMatch]  # or {} if none
    # -------------------------------------------------

    positive_trials_ade: List[ADEAnalysisResultForStudy] = []
    negative_trials_ade: List[ADEAnalysisResultForStudy] = []

    if not os.path.exists(ADE_ANALYSIS_RESULTS_PATH):

        normalized_ade_processing_errors: Dict[str, int] = {}

        for nctid in tqdm.tqdm(nctids_list_filtered, desc="ADE matching per study"):
            study_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nctid}.json")
            with open(study_path, "r", encoding="utf-8") as f:
                study = Study.model_validate_json(f.read())

            ade_analysis_result, ade_error = process_study_for_ade_risks(study, meddra_labels)

            if ade_error:
                normalized_ade_processing_errors[ade_error] = normalized_ade_processing_errors.get(ade_error, 0) + 1
                continue

            if ade_analysis_result.positive_terms:
                positive_trials_ade.append(ade_analysis_result)
            else:
                negative_trials_ade.append(ade_analysis_result)

        positive_trials_ade.sort(key=lambda x: x.nctid, reverse=False)
        negative_trials_ade.sort(key=lambda x: x.nctid, reverse=False)

        with open(ADE_ANALYSIS_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "positive_trials": [item.model_dump() for item in positive_trials_ade],
                    "negative_trials": [item.model_dump() for item in negative_trials_ade],
                    "normalized_ade_processing_errors": normalized_ade_processing_errors,
                },
                f,
                indent=2,
            )

    else:
        with open(ADE_ANALYSIS_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            positive_trials_ade = [ADEAnalysisResultForStudy.model_validate(item) for item in data["positive_trials"]]
            negative_trials_ade = [ADEAnalysisResultForStudy.model_validate(item) for item in data["negative_trials"]]

    # -------------------------------------------------
    # 3) Build global canonical label columns
    #    (best-match label per term within each positive study)
    # -------------------------------------------------
    canonical_label_set: set[str] = set()

    for ade_analysis in positive_trials_ade:  # List[ADEAnalysisResultForStudy]
        canonical_label_set.update(
            canonical_labels_from_positive_terms(ade_analysis.positive_terms)
        )

    canonical_label_cols = sorted(canonical_label_set)

    # -------------------------------------------------
    # 4) Feature extraction (per study, using ADE enrichment)
    # -------------------------------------------------
    dataset_features: List[FeaturesList] = []
    for ade_analysis in tqdm.tqdm(positive_trials_ade + negative_trials_ade, desc="Extracting features"):
        study_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{ade_analysis.nctid}.json")
        with open(study_path, "r", encoding="utf-8") as f:
            study = Study.model_validate_json(f.read())

        features = extract_features_for_study(
            study,
            canonical_label_cols=canonical_label_cols,
            ade_analysis_results_for_study=ade_analysis,
        )

        features = features.expand_enums()
        dataset_features.append(features)

    # -------------------------------------------------
    # 5)  Dataset creation
    # -----------------------------------

    def hf_type_map(t: type) -> str:
        if t is str:
            return "string"
        if t is int:
            return "int64"
        if t is float:
            return "float64"
        if t is bool:
            return "bool"
        else:
            raise NotImplementedError

    first = dataset_features[0]
    names = first.get_names()
    types = first.get_types()

    schema = Features({n: Value(hf_type_map(t)) for n, t in zip(names, types)})

    hf_dataset = Dataset.from_list(
        [dict(zip(fl.get_names(), fl.get_values())) for fl in dataset_features],
        features=schema,
    )

    # -------------------------------------------------
    # 6) Saving
    # -------------------------------------------------
    # TODO: Add versioning
    hf_dataset.save_to_disk(END_POINT_HF_DATASET_PATH)


if __name__ == '__main__':
    if not os.path.exists(END_POINT_HF_DATASET_PATH):
        main()
