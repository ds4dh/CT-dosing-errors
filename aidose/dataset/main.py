from aidose import PACKAGE_NAME

from aidose.dataset.constants import (WILSON_PROBA_THRESHOLD,
                                      ALPHA_WILSON,
                                      TRAINING_SIZE,
                                      VALIDATION_SIZE,
                                      TEST_SIZE,
                                      ENUM_FIELDS_EXPANSION)
from aidose.dataset import (
    MEDDRA_ADE_LABELS_PATH,
    MEDDRA_HLGT_CODES_LITERAL,
    CTGOV_NCTIDS_LIST_FILTERED_PATH,
    ADE_ANALYSIS_RESULTS_PATH,
    END_POINT_HF_DATASET_PATH,
    CTGOV_KNOWLEDGE_CUTOFF_DATE,
    DATASET_NAME,
    DATASET_VERSION
)

from aidose.dataset.utils import (include_trial_after_sequential_filtering,
                                  make_dataset_info,
                                  build_struct_schema_from_attributes,
                                  serialize_attributes_for_hf)

from aidose.dataset.ade import process_study_for_ade_risks
from aidose.dataset.ade import ADEAnalysisResultForStudy
from aidose.dataset.ade_labeling import canonical_labels_from_positive_terms
from aidose.dataset.ade_manual_filtering import filter_ade_terms_to_focus_on_dosing_errors
from aidose.dataset.attribute import AttributesList
from aidose.dataset.feature_extraction import extract_attributes_from_study, ATTRIBS_METADATA_PREFIX
from aidose.dataset.split import ListSplitter

from aidose.meddra import MEDDRA_VERSION, MEDDRA_DATASET_PATH
from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import parse_hlgt_codes_literal
from aidose.meddra.extraction import build_meddra_descendants

from aidose.ctgov import (CTGOV_NCTIDS_LIST_ALL_PATH,
                          CTGOV_DATASET_RAW_PATH,
                          CTGOV_DATASET_PATH,
                          CTGOV_EXTRACTED_PDFS_DATASET_PATH)

from aidose.ctgov.structures import Study
from aidose.ctgov import (download_registry_from_api,
                          download_pdfs_for_all_trials_with_available_documents,
                          extract_text_incrementally_from_downloaded_document_pdfs)
from aidose.ctgov.utils_download import get_study_path_by_nctid_and_raw_dir

from datasets import Dataset, Features, DatasetDict, load_from_disk

from typing import List, Dict

import os
import json
import tqdm
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)  # TODO: Perhaps write the logs to disk.


def parse_study_by_nctid_from_json_path(nctid: str) -> Study:
    json_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
    with open(json_path, "r", encoding="utf-8") as f:
        study = Study.model_validate_json(f.read())
    return study


def main():
    # -----------------------------
    # 0) MedDRA ADE-related positive terms + dosing-related manual filtering
    # -----------------------------
    meddra_labels: List[str] = []
    if not os.path.exists(MEDDRA_ADE_LABELS_PATH):
        logger.info(f"The list of ADE-related MedDRA terms does not exist at {MEDDRA_ADE_LABELS_PATH}, "
                    f"so creating one by traversing MedDRA...")
        meddra = MedDRA()
        meddra.load_data(MEDDRA_DATASET_PATH)

        codes = parse_hlgt_codes_literal(MEDDRA_HLGT_CODES_LITERAL)
        ade_analysis_result = build_meddra_descendants(meddra, codes)
        meddra_labels = sorted(list(ade_analysis_result.terms))
        with open(MEDDRA_ADE_LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump({"terms": meddra_labels}, f, ensure_ascii=False, indent=4)

    else:
        with open(MEDDRA_ADE_LABELS_PATH, "r", encoding="utf-8") as f:
            meddra_labels = json.load(f).get("terms")
        logger.info(f"The list of ADE-related MedDRA terms exists at {MEDDRA_ADE_LABELS_PATH}, so loading it ...")

    logger.info("Manually filtering the ADE-related MedDRA terms to focus on dosing errors ...")
    meddra_labels = filter_ade_terms_to_focus_on_dosing_errors(meddra_labels)
    logger.info(f"After manual filtering, {len(meddra_labels)} ADE-related MedDRA terms remain.")

    # -----------------------------------
    # 1) CTGov download and filtering
    # -----------------------------------
    if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and
            os.path.exists(CTGOV_DATASET_RAW_PATH) and
            os.path.exists(os.path.join(CTGOV_DATASET_PATH, "download-time-tag.txt"))
    ):
        logger.info("Downloading CTGov registry from API ...")
        download_registry_from_api(CTGOV_KNOWLEDGE_CUTOFF_DATE)
    with open(os.path.join(CTGOV_DATASET_PATH, "download-time-tag.txt"), "r", encoding="utf-8") as f:
        try:
            ctgov_download_timestamp = datetime.strptime(
                f.readlines()[0].split("Download time (UTC):", 1)[1].strip(), "%Y-%m-%dT%H:%MZ")
        except ValueError:
            raise RuntimeError("Did not manage to parse the CTGov download timestamp.")

    nctids_list_filtered: List[str] = []
    if not os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH):
        logger.info("Filtering CTGov trials for inclusion ...")
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r', encoding='utf-8') as f:
            nctids_list_all = [line.strip() for line in f if line.strip()]

        for nct_id in tqdm.tqdm(nctids_list_all, desc="Parsing trials and filtering them."):
            study = parse_study_by_nctid_from_json_path(nct_id)

            if include_trial_after_sequential_filtering(study, CTGOV_KNOWLEDGE_CUTOFF_DATE):
                nctids_list_filtered.append(nct_id)

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'w', encoding='utf-8') as f:
            for nctid in nctids_list_filtered:
                f.write(f"{nctid}\n")
        logger.info("Included {} studies after sequential filtering of the {} CTGov studies.".format(
            len(nctids_list_filtered), len(nctids_list_all)))
    else:
        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, "r", encoding="utf-8") as f:
            nctids_list_filtered = [line.strip() for line in f if line.strip()]

        logger.info("Loaded (from {}) the list of {} CTGov trials included to the dataset....".format(
            CTGOV_NCTIDS_LIST_FILTERED_PATH, len(nctids_list_filtered)))

    if not os.path.exists(CTGOV_EXTRACTED_PDFS_DATASET_PATH):
        logger.info("Didn't find an existing dataset of extracted PDF texts, so attempting to create one ...")
        download_pdfs_for_all_trials_with_available_documents()
        extract_text_incrementally_from_downloaded_document_pdfs()
    else:
        logger.info("Found an existing dataset of extracted PDF texts, so re-using it ...")

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
        logger.info("Didn't find an existing analysis of the ADE's for positive/negative trials, so creating one ...")

        normalized_ade_processing_errors: Dict[str, int] = {}

        for nctid in tqdm.tqdm(nctids_list_filtered, desc="ADE matching per study"):
            study = parse_study_by_nctid_from_json_path(nctid)

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
        logger.info("Finalized with the ADE analysis of the trials and wrote them to disk for future re-use.")

    else:
        with open(ADE_ANALYSIS_RESULTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            positive_trials_ade = [ADEAnalysisResultForStudy.model_validate(item) for item in data["positive_trials"]]
            negative_trials_ade = [ADEAnalysisResultForStudy.model_validate(item) for item in data["negative_trials"]]

        logger.info("Loaded the ADE analysis of the positive/negative trials from {} ...".format(
            ADE_ANALYSIS_RESULTS_PATH))

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
    logger.info(f"Extracted {len(canonical_label_cols)} unique canonical ADE labels from positive terms ...")

    # -------------------------------------------------
    # 4) Load extracted PDF texts dataset
    # -------------------------------------------------
    extracted_pdf_texts_dataset = load_from_disk(CTGOV_EXTRACTED_PDFS_DATASET_PATH)
    logger.info("Loaded the dataset of extracted PDF texts from disk ...")
    map_of_nctid_to_extracted_texts_from_pdfs = dict(zip(extracted_pdf_texts_dataset["nctid"],
                                                         extracted_pdf_texts_dataset["extracted_text"]))

    # -------------------------------------------------
    # 5) Features, metadata and label extraction (per study, using ADE enrichment)
    # -------------------------------------------------
    dataset_attribs: List[AttributesList] = []
    for ade_analysis in tqdm.tqdm(positive_trials_ade + negative_trials_ade, desc="Extracting features"):
        study = parse_study_by_nctid_from_json_path(ade_analysis.nctid)

        attribs = extract_attributes_from_study(
            study,
            canonical_label_cols=canonical_label_cols,
            ade_analysis_results_for_study=ade_analysis,
            wilson_proba_threshold=WILSON_PROBA_THRESHOLD,
            alpha_wilson=ALPHA_WILSON,
            map_of_nctid_to_extracted_texts_from_pdfs=map_of_nctid_to_extracted_texts_from_pdfs
        )

        if ENUM_FIELDS_EXPANSION:
            attribs = attribs.expand_enums()
        dataset_attribs.append(attribs)

    logger.info("Finalized with the extraction of features, labels and the metadata.")

    # -------------------------------------------------
    # 6) Dataset splitting
    # -------------------------------------------------
    splitter = ListSplitter(split_proportions=(TRAINING_SIZE, VALIDATION_SIZE, TEST_SIZE))
    train_idx, valid_idx, test_idx = splitter.get_split_indices(
        data=dataset_attribs,
        key=ListSplitter.chronological_key(dataset_attribs, f"{ATTRIBS_METADATA_PREFIX}completionDate")
    )

    dataset_attribs_train: List[AttributesList] = [dataset_attribs[i] for i in train_idx]
    dataset_attribs_valid: List[AttributesList] = [dataset_attribs[i] for i in valid_idx]
    dataset_attribs_test: List[AttributesList] = [dataset_attribs[i] for i in test_idx]

    logger.info("Split the dataset into train/valid/test with sizes: {}/{}/{}.".format(
        len(dataset_attribs_train), len(dataset_attribs_valid), len(dataset_attribs_test)))

    # -------------------------------------------------
    # 7)  Dataset creation with schemas
    # -----------------------------------
    first_row = dataset_attribs[0]
    attrib_names = first_row.get_names()
    attrib_types = first_row.get_types()
    sample_values = first_row.get_values()

    schema_dict = build_struct_schema_from_attributes(attrib_names, attrib_types, sample_values)
    schema = Features(schema_dict)

    ds_info = make_dataset_info(
        dataset_version=DATASET_VERSION,
        description="""{} (v{}): A dataset to study the medication errors (dosing errors) in clinical trials. 
        
        Based on the studies from `www.clinicaltrials.gov`, completed before {} and downloaded at {}, and the 
        medical dictionary of `www.meddra.org`, with version {}.""".format(
            DATASET_NAME,
            DATASET_VERSION,
            CTGOV_KNOWLEDGE_CUTOFF_DATE.strftime("%Y-%m-%dT%HZ") if isinstance(CTGOV_KNOWLEDGE_CUTOFF_DATE, datetime)
            else datetime.now().strftime("%Y-%m-%dT%HZ"),
            ctgov_download_timestamp.strftime("%Y-%m-%dT%HZ"),
            MEDDRA_VERSION),
        license_str="CC BY 4.0",
        package_name=PACKAGE_NAME,
        features=schema
    )

    hf_dataset_train = Dataset.from_list(
        [serialize_attributes_for_hf(fl) for fl in dataset_attribs_train],
        features=schema,
        info=ds_info
    )

    hf_dataset_valid = Dataset.from_list(
        [serialize_attributes_for_hf(fl) for fl in dataset_attribs_valid],
        features=schema,
        info=ds_info
    )

    hf_dataset_test = Dataset.from_list(
        [serialize_attributes_for_hf(fl) for fl in dataset_attribs_test],
        features=schema,
        info=ds_info
    )

    hf_dataset_dict = DatasetDict({
        "train": hf_dataset_train,
        "validation": hf_dataset_valid,
        "test": hf_dataset_test})

    logger.info("Created a `datasets.DatasetDict` instance with train/valid/test splits.")

    # -------------------------------------------------
    # 8) Saving
    # -------------------------------------------------

    hf_dataset_dict.save_to_disk(END_POINT_HF_DATASET_PATH)
    logger.info(f"Saved the dataset {DATASET_NAME} with version {DATASET_VERSION} to {END_POINT_HF_DATASET_PATH} ...")


if __name__ == '__main__':
    try:
        load_from_disk(END_POINT_HF_DATASET_PATH)
        logger.info(f"The dataset already exists at {END_POINT_HF_DATASET_PATH}, so not re-creating it.")
    except FileNotFoundError:
        main()
