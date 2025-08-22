import pandas as pd

from aidose.meddra import MEDDRA_VERSION, MEDDRA_DATASET_PATH
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH, CTGOV_DATASET_PATH
from aidose.dataset import (
    MEDDRA_LABELS_JSON_PATH,
    MEDDRA_HLGT_CODES_LITERAL,
    CTGOV_NCTIDS_LIST_FILTERED_PATH,
    ADE_ANALYSIS_RESULTS_PATH,
    END_POINT_HF_DATASET_PATH
)

from aidose.dataset.constants import LIST_OF_FEATURES_TO_DROP, WILSON_PROBA_THRESHOLD, ALPHA_WILSON, TRAINING_SIZE, \
    VALIDATION_SIZE, TEST_SIZE
from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import parse_hlgt_codes_literal
from aidose.meddra.extraction import build_meddra_descendants

from aidose.ctgov.structures import Study
import aidose.ctgov.api_download as api_download

from aidose.dataset.utils import include_trial_after_sequential_filtering

from aidose.dataset.ade import process_study_for_ade_risks
from aidose.dataset.ade import ADEAnalysisResultForStudy

from aidose.dataset.ade_labeling import canonical_labels_from_positive_terms

from aidose.dataset.feature import FeaturesList
from aidose.dataset.feature_extraction import (extract_features_for_training_from_study,
                                               extract_metadata_from_study,
                                               extract_labels_from_study)

from aidose.dataset.split import DatasetSplit
from aidose.dataset.final_processing import add_sum_dosing_error, add_dosing_error_rate, add_wilson_label, \
    dataset_spliting

from datasets import Dataset, Features, Value, DatasetInfo, DatasetDict, Version

from typing import List, Dict

import os
import json
import tqdm
from datetime import datetime


def main():
    # -----------------------------
    # 0) MedDRA positive terms
    # -----------------------------
    meddra_labels: List[str] = []
    if not os.path.exists(MEDDRA_LABELS_JSON_PATH):
        meddra = MedDRA()
        meddra.load_data(MEDDRA_DATASET_PATH)

        codes = parse_hlgt_codes_literal(MEDDRA_HLGT_CODES_LITERAL)
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
    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and
            os.path.exists(CTGOV_DATASET_RAW_PATH) and
            os.path.exists(os.path.join(CTGOV_DATASET_PATH, "download-time-tag.txt"))
    ):
        api_download.main()
    with open(os.path.join(CTGOV_DATASET_PATH, "download-time-tag.txt"), "r", encoding="utf-8") as f:
        try:
            ctgov_download_timestamp = datetime.strptime(
                f.readlines()[0].split("Download time (UTC):", 1)[1].strip(), "%Y-%m-%dT%H:%MZ")
        except ValueError:
            raise RuntimeError("Did not manage to parse the CTGov download timestamp.")

    nctids_list_filtered: List[str] = []
    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r', encoding='utf-8') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]

        for nct_id in tqdm.tqdm(nctids_list_all, desc="Parsing trials and filtering them."):
            json_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nct_id}.json")

            # specify encoding type to be compatible with windows machine
            with open(json_path, 'r', encoding='utf-8') as f:
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
    # 4) Features, metadata and label extraction (per study, using ADE enrichment)
    # -------------------------------------------------
    dataset_features: List[FeaturesList] = []
    dataset_metadata: List[FeaturesList] = []
    dataset_labels: List[FeaturesList] = []

    for ade_analysis in tqdm.tqdm(positive_trials_ade + negative_trials_ade, desc="Extracting features"):
        study_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{ade_analysis.nctid}.json")
        with open(study_path, "r", encoding="utf-8") as f:
            study = Study.model_validate_json(f.read())

        features = extract_features_for_training_from_study(study)
        metadata = extract_metadata_from_study(study)
        labels = extract_labels_from_study(
            canonical_label_cols=canonical_label_cols,
            ade_analysis_results_for_study=ade_analysis,
            wilson_proba_threshold=WILSON_PROBA_THRESHOLD,
            alpha_wilson=ALPHA_WILSON,
        )

        features = features.expand_enums()
        metadata = metadata.expand_enums()
        labels = labels.expand_enums()

        dataset_features.append(features)
        dataset_metadata.append(metadata)
        dataset_labels.append(labels)

    # TODO: Maybe doing the splitting here and before the `dataset` creation?

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
        if t is datetime:
            return "date32"
        else:
            raise NotImplementedError

    def build_struct_schema(names, types):
        return {n: Value(hf_type_map(t)) for n, t in zip(names, types)}

    # --- derive sub-schemas ---
    feat_names, feat_types = dataset_features[0].get_names(), dataset_features[0].get_types()
    meta_names, meta_types = dataset_metadata[0].get_names(), dataset_metadata[0].get_types()
    label_names, label_types = dataset_labels[0].get_names(), dataset_labels[0].get_types()

    features_schema = Features(build_struct_schema(feat_names, feat_types))
    metadata_schema = Features(build_struct_schema(meta_names, meta_types))
    labels_schema = Features(build_struct_schema(label_names, label_types))

    schema = Features({
        "features": features_schema,
        "metadata": metadata_schema,
        "labels": labels_schema,
    })

    hf_dataset = Dataset.from_dict(
        {
            "features": [dict(zip(fl.get_names(), fl.get_values())) for fl in dataset_features],
            "metadata": [dict(zip(fl.get_names(), fl.get_values())) for fl in dataset_metadata],
            "labels": [dict(zip(fl.get_names(), fl.get_values())) for fl in dataset_labels],
        },

        features=schema,
        info=DatasetInfo(features=schema,
                         description="""A dataset to study the ADE risks in clinical trials. 
        
        Based on the studies from `www.clinicaltrials.gov`, downloaded at {}, and the medical dictionary of 
        `www.meddra.org`, with version {}.""".format(
                             ctgov_download_timestamp.strftime("%Y-%m-%dT%HZ"), MEDDRA_VERSION), )
    )

    # TODO: Add versioning

    # -------------------------------------------------
    # 6) Dataset splitting
    # -------------------------------------------------

    dataset_split_dict = DatasetSplit(
        train_ratio=TRAINING_SIZE,
        valid_ratio=VALIDATION_SIZE,
        test_ratio=TEST_SIZE).split_chronologically(hf_dataset, date_path="metadata.completionDate")

    # -------------------------------------------------
    # 6) Saving
    # -------------------------------------------------

    dataset_split_dict.save_to_disk(END_POINT_HF_DATASET_PATH)


if __name__ == '__main__':
    if not os.path.exists(END_POINT_HF_DATASET_PATH):
        main()
