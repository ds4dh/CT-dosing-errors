import os
import shutil
import logging
import time
from typing import Dict, Any, List

from datasets import load_dataset, DatasetDict, Features
from huggingface_hub import DatasetCard, HfApi
from aidose.dataset import DATASET_VERSION
from aidose import END_POINT_HF_DATASET_PATH, DATASETS_ROOT

# --- CONFIGURATION ---
# OPTIONS: "phase1", "phase2", "release"
PHASE = "phase2"

HF_HUB_REPO_ID = "sssohrab/ct-dosing-errors-benchmark"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def adapt_features(original_features: Features) -> Features:
    """Creates a new Features schema based on the original one."""
    new_features_dict = {}
    for col_name, feature in original_features.items():
        if col_name.startswith("FEATURE_"):
            new_name = col_name.replace("FEATURE_", "")
            new_features_dict[new_name] = feature
        elif col_name == "LABEL_wilson_label":
            new_features_dict["target"] = feature
        elif col_name == "METADATA_nctId":
            new_features_dict["nctid"] = feature
    return Features(new_features_dict)


def clean_and_rename(example: Dict[str, Any]) -> Dict[str, Any]:
    new_example: Dict[str, Any] = {}
    for key, value in example.items():
        if key.startswith("FEATURE_"):
            new_key = key.replace("FEATURE_", "")
            new_example[new_key] = value
    if "LABEL_wilson_label" in example:
        new_example["target"] = example["LABEL_wilson_label"]
    if "METADATA_nctId" in example:
        new_example["nctid"] = example["METADATA_nctId"]
    return new_example


def process_splits(ds_dict: DatasetDict) -> DatasetDict:
    original_cols: List[str] = ds_dict["train"].column_names
    logger.info("Adapting feature schema (preserving ClassLabels)...")
    new_features = adapt_features(ds_dict["train"].features)

    logger.info("Cleaning and renaming features across all splits...")
    cleaned_ds = ds_dict.map(
        clean_and_rename,
        remove_columns=original_cols,
        features=new_features,
        desc="Cleaning features"
    )

    # Set high-level info
    for split in cleaned_ds:
        cleaned_ds[split].info.license = "CC BY 4.0"
        cleaned_ds[split].info.description = "CT-DOSING-ERRORS: Clinical trials dosing error benchmark."

    return cleaned_ds


def save_phase_data(dataset, phase_name: str, output_root: str):
    """Saves server-side reference data (Always unmasked labels for scoring)."""
    input_dir = os.path.join(output_root, f"{phase_name}_input")
    ref_dir = os.path.join(output_root, f"{phase_name}_ref")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    cols_to_keep = [c for c in dataset.column_names if c != "target"]
    features = dataset.select_columns(cols_to_keep)

    feat_path = os.path.join(input_dir, f"{phase_name}_features.parquet")
    features.to_parquet(feat_path)
    logger.info(f"Saved {phase_name} features to {feat_path}")

    if "target" in dataset.column_names and "nctid" in dataset.column_names:
        labels = dataset.select_columns(["nctid", "target"])
        label_path = os.path.join(ref_dir, f"{phase_name}_labels.csv")
        labels.to_csv(label_path, index=False)
        logger.info(f"Saved {phase_name} labels to {label_path}")

    return input_dir, ref_dir


def update_hf_metadata(repo_id: str, token: str, version_str: str):
    logger.info("Waiting 5s for HF API to settle before updating metadata...")
    time.sleep(5)

    logger.info(f"Updating Dataset Card metadata (Version: {version_str})...")
    try:
        card = DatasetCard.load(repo_id, token=token)
        card.data.license = "cc-by-4.0"
        card.data.tags = ["clinical-trials", "medication-safety", "tabular-classification"]
        card.data.version = version_str
        card.push_to_hub(repo_id, token=token)
        logger.info("Dataset Card metadata updated successfully.")
    except Exception as e:
        logger.error(f"Failed to update Dataset Card metadata: {e}")


def create_hf_tag(repo_id: str, token: str, version_str: str):
    api = HfApi(token=token)
    tag_name = f"v{version_str}"
    logger.info(f"Creating Git Tag '{tag_name}' on HF Hub...")
    try:
        api.create_tag(
            repo_id=repo_id,
            repo_type="dataset",
            tag=tag_name,
            tag_message=f"Release {version_str} for CT-DEB'26"
        )
        logger.info(f"Successfully created tag: {tag_name}")
    except Exception as e:
        if "already exists" in str(e):
            logger.warning(f"Tag '{tag_name}' already exists. Skipping.")
        else:
            logger.error(f"Failed to create Git tag: {e}")


def export_for_codabench(ds_dict: DatasetDict, output_path: str, public_version: str) -> None:
    if os.path.exists(output_path):
        logger.info(f"Cleaning existing directory: {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    zips_dir = os.path.join(output_path, "zips_to_upload")
    os.makedirs(zips_dir, exist_ok=True)

    # --- 1. Construct Public Dataset based on PHASE ---
    logger.info(f"Preparing data for PHASE: {PHASE}...")

    # Always include Train (Unmasked)
    splits_to_push = {"train": ds_dict["train"]}

    if PHASE == "phase1":
        # Phase 1: Validation (Masked), Test (Hidden)
        logger.info("[Phase 1] Masking Validation targets. Excluding Test set.")
        splits_to_push["validation"] = ds_dict["validation"].map(lambda x: {"target": -1})
        # Test set is deliberately omitted

    elif PHASE == "phase2":
        # Phase 2: Validation (Unmasked), Test (Masked)
        logger.info("[Phase 2] Unmasking Validation. Masking Test targets.")
        splits_to_push["validation"] = ds_dict["validation"]  # Fully revealed
        splits_to_push["test"] = ds_dict["test"].map(lambda x: {"target": -1})

    elif PHASE == "release":
        # Release: All Unmasked
        logger.info("[Release] All splits unmasked.")
        splits_to_push["validation"] = ds_dict["validation"]
        splits_to_push["test"] = ds_dict["test"]

    else:
        raise ValueError(f"Unknown PHASE: {PHASE}")

    public_ds = DatasetDict(splits_to_push)

    # --- 2. Push to Hugging Face Hub ---
    try:
        logger.info(f"Pushing to HF Hub: {HF_HUB_REPO_ID}...")
        public_ds.push_to_hub(HF_HUB_REPO_ID, token=HF_TOKEN, private=True)
        logger.info("Push successful.")

        update_hf_metadata(HF_HUB_REPO_ID, HF_TOKEN, public_version)
        create_hf_tag(HF_HUB_REPO_ID, HF_TOKEN, public_version)

    except Exception as e:
        logger.error(f"Failed to push to HF Hub. Error: {e}")

    # --- 3. Generate Server-Side Zips (Always Unmasked for Scoring) ---
    # These files are for YOU to upload to CodaBench private resources.
    # They must always contain the ground truth labels to perform scoring.
    logger.info("Generating server-side scoring zips (Unmasked reference)...")

    val_in, val_ref = save_phase_data(ds_dict["validation"], "val", output_path)
    shutil.make_archive(os.path.join(zips_dir, "val_input"), 'zip', val_in)
    shutil.make_archive(os.path.join(zips_dir, "val_ref"), 'zip', val_ref)

    test_in, test_ref = save_phase_data(ds_dict["test"], "test", output_path)
    shutil.make_archive(os.path.join(zips_dir, "test_input"), 'zip', test_in)
    shutil.make_archive(os.path.join(zips_dir, "test_ref"), 'zip', test_ref)

    logger.info(f"Export complete. Server zips are ready in '{zips_dir}'.")


if __name__ == "__main__":
    # Construct version string based on dataset version AND phase
    # e.g., "0.2.2_CT-DEB26-phase1"
    PUBLIC_VERSION = f"{DATASET_VERSION}_CT-DEB26-{PHASE}"

    SOURCE_PRIVATE_DATASET_PATH = END_POINT_HF_DATASET_PATH
    DESTINATION_PUBLIC_DATASET_PATH = os.path.join(
        DATASETS_ROOT,
        "CT-DOSING-ERROR-BENCHMARK",
        f"{PUBLIC_VERSION}"
    )

    logger.info(f"Loading dataset from {SOURCE_PRIVATE_DATASET_PATH}...")
    dataset = load_dataset(SOURCE_PRIVATE_DATASET_PATH)

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Loaded dataset is not a DatasetDict.")

    # Run Pipeline
    cleaned_dataset = process_splits(dataset)
    export_for_codabench(cleaned_dataset, DESTINATION_PUBLIC_DATASET_PATH, PUBLIC_VERSION)