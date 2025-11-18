from aidose.dataset import DATASET_VERSION
from aidose import END_POINT_HF_DATASET_PATH, DATASETS_ROOT

from datasets import load_dataset, DatasetDict

from typing import Dict, Any, List
import os
import shutil



def clean_and_rename(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filters and renames columns:
    1. Strips 'FEATURE_' prefix.
    2. Retains 'LABEL_wilson_label' as 'target'.
    3. Drops all other metadata/labels.
    """
    new_example: Dict[str, Any] = {}

    # 1. Keep and rename FEATURE columns
    for key, value in example.items():
        if key.startswith("FEATURE_"):
            new_key = key.replace("FEATURE_", "")
            new_example[new_key] = value

    # 2. Keep only the specific Wilson label and rename it to 'target'
    if "LABEL_wilson_label" in example:
        new_example["target"] = example["LABEL_wilson_label"]

    return new_example


def process_splits(ds_dict: DatasetDict) -> DatasetDict:
    """
    Applies the cleaning function to all splits in the DatasetDict.
    """
    original_cols: List[str] = ds_dict["train"].column_names

    cleaned_ds = ds_dict.map(
        clean_and_rename,
        remove_columns=original_cols,
        desc="Cleaning and renaming features"
    )

    return cleaned_ds


def save_phase_data(dataset, phase_name: str, output_root: str):
    """
    Helper to split a dataset split into Input (Features) and Reference (Labels)
    and save them to separate folders.
    """
    input_dir = os.path.join(output_root, f"{phase_name}_input")
    ref_dir = os.path.join(output_root, f"{phase_name}_ref")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    # 1. Save X (Features) - Drop the target
    if "target" in dataset.column_names:
        features = dataset.remove_columns(["target"])
    else:
        features = dataset

    features.to_parquet(os.path.join(input_dir, f"{phase_name}_features.parquet"))

    # 2. Save Y (Labels) - Keep only target
    if "target" in dataset.column_names:
        labels = dataset.select_columns(["target"])
        labels.to_parquet(os.path.join(ref_dir, f"{phase_name}_labels.parquet"))

    return input_dir, ref_dir


def export_for_codabench(ds_dict: DatasetDict, output_path: str) -> None:
    """
    Generates the following structure and ZIPS them:
    1. public_data.zip      -> Contains Train features+labels (For users to download)
    2. val_input.zip        -> Validation Features (Phase 1 Server Input)
    3. val_ref.zip          -> Validation Labels (Phase 1 Server Ground Truth)
    4. test_input.zip       -> Test Features (Phase 2 Server Input)
    5. test_ref.zip         -> Test Labels (Phase 2 Server Ground Truth)
    """

    # Safety: Clean output directory
    if os.path.exists(output_path):
        print(f"Cleaning existing directory: {output_path}")
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    zips_dir = os.path.join(output_path, "zips_to_upload")
    os.makedirs(zips_dir, exist_ok=True)

    # --- 1. Public Data (Train Only) ---
    # Users download this to train their models
    print("Processing Public Data (Train)...")
    public_dir = os.path.join(output_path, "public_data")
    os.makedirs(public_dir, exist_ok=True)
    ds_dict["train"].to_parquet(os.path.join(public_dir, "train_data.parquet"))

    # Zip Public Data
    shutil.make_archive(os.path.join(zips_dir, "public_data"), 'zip', public_dir)

    # --- 2. Phase 1: Validation Data ---
    # Hidden on server. Users predict on this for the Leaderboard.
    print("Processing Phase 1 (Validation)...")
    val_in, val_ref = save_phase_data(ds_dict["validation"], "val", output_path)
    shutil.make_archive(os.path.join(zips_dir, "val_input"), 'zip', val_in)
    shutil.make_archive(os.path.join(zips_dir, "val_ref"), 'zip', val_ref)

    # --- 3. Phase 2: Test Data ---
    # Hidden on server. Used for final auto-migration grading.
    print("Processing Phase 2 (Test)...")
    test_in, test_ref = save_phase_data(ds_dict["test"], "test", output_path)
    shutil.make_archive(os.path.join(zips_dir, "test_input"), 'zip', test_in)
    shutil.make_archive(os.path.join(zips_dir, "test_ref"), 'zip', test_ref)

    print(f"Export complete. Upload the files in '{zips_dir}' to CodaBench.")


if __name__ == "__main__":
    # Configuration
    SOURCE_PRIVATE_DATASET_PATH = END_POINT_HF_DATASET_PATH
    DESTINATION_PUBLIC_DATASET_PATH = os.path.join(
        DATASETS_ROOT,
        "CT-DOSING-ERROR-BENCHMARK",
        f"{DATASET_VERSION}-rc1"
    )

    print(f"Loading dataset from {SOURCE_PRIVATE_DATASET_PATH}...")
    dataset = load_dataset(SOURCE_PRIVATE_DATASET_PATH)

    if not isinstance(dataset, DatasetDict):
        raise ValueError("Loaded dataset is not a DatasetDict.")

    # Run Pipeline
    cleaned_dataset = process_splits(dataset)
    export_for_codabench(cleaned_dataset, DESTINATION_PUBLIC_DATASET_PATH)
