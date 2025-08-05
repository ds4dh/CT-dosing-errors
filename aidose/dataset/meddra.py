from aidose.meddra.graph import MedDRA, MedDRALevel
from aidose.meddra.utils import (get_descendant_terms, get_descendant_info,
                                 has_complete_path, DescendantEntry)

from aidose.meddra import MEDDRA_DATASET_PATH, MEDDRA_CREATED_ARTIFACTS_DIR

import ast
import json
import csv
import argparse
import os

MEDDRA_TERMS_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")
MEDDRA_PATHS_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_paths.json")

MEDDRA_LABELS_JSON_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")
MEDDRA_LABELS_CSV_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.csv")
MEDDRA_PATHS_JSON_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_paths.json")
MEDDRA_PATHS_CSV_PATH = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_paths.csv")

# TODO: This whole script is very bad. Core utilities should be factored out and API's moved to main.py.

def parse_arguments():
    """Parse command-line arguments. Only HLGT codes are configurable."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract descendant PT/LLT terms and full MedDRA paths from HLGT nodes. "
            "Input and output directories are fixed in the config."
        )
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="[('HLGT', '10079145'), ('HLGT', '10079159')]",
        help=(
            "List of HLGT codes as a string, e.g. "
            "\"[('HLGT', '10079145'), ('HLGT', '10079159')]\""
        ),
    )

    args = parser.parse_args()

    # Inject fixed paths from config
    args.data = MEDDRA_DATASET_PATH
    args.output = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_positive_labels.json")
    args.paths_output = os.path.join(MEDDRA_CREATED_ARTIFACTS_DIR, "meddra_paths.json")

    return args


def get_meddra_positive_labels() -> None:
    """
        Main entry point for extracting MedDRA descendant terms and their hierarchical paths.

        This function:
          - Loads the MedDRA dataset from a fixed path defined in the config.
          - Parses a list of HLGT codes (as --codes argument).
          - For each HLGT node:
              - Finds all descendant PT and LLT terms.
              - Records full hierarchical paths (SOC ➝ HLGT ➝ HLT ➝ PT ➝ LLT).
              - Filters for complete paths when available.
          - Saves:
              - A sorted list of unique descendant terms as JSON.
              - A mapping from HLGT codes to full path information as JSON.
          - Prints summary statistics including completeness of paths.

        Note:
            All paths for loading and saving data are controlled by `MEDDRA_DATASET_PATH`
            and `MEDDRA_CREATED_ARTIFACTS_DIR` from the config file.
        """

    args = parse_arguments()

    try:
        raw_codes: list[tuple[str, str]] = ast.literal_eval(args.codes)
        codes_list: list[tuple[MedDRALevel, str]] = [
            (MedDRALevel.from_str(level), code) for level, code in raw_codes
        ]
    except Exception as e:
        print("Error parsing HLGT codes:", e)
        return

    if not os.path.exists(MEDDRA_DATASET_PATH):
        raise FileNotFoundError(f"Data directory not found: {MEDDRA_DATASET_PATH}")

    # Load MedDRA dataset
    meddra = MedDRA()
    meddra.load_data(MEDDRA_DATASET_PATH)

    all_terms: set[str] = set()
    all_paths_info: dict[str, dict[str, DescendantEntry]] = {}

    total_descendants = 0
    no_complete_path_count = 0

    for level, code in codes_list:
        node_key = (level, code)
        if node_key not in meddra.nodes:
            print(f"Node {level}@{code} not found in MedDRA.")
            continue

        hlgt_node = meddra.nodes[node_key]

        # 1. Collect descendant terms (LLT, PT)
        descendant_terms = get_descendant_terms(hlgt_node, meddra)
        all_terms |= descendant_terms

        # 2. Collect full path info
        descendant_info = get_descendant_info(hlgt_node, meddra)
        all_paths_info[f"{code}@{level}"] = descendant_info

        # 3. Track completeness
        for desc_key, info in descendant_info.items():
            total_descendants += 1

            try:
                _, desc_level_str = desc_key.split("@")
                desc_level = MedDRALevel.from_str(desc_level_str)
            except Exception:
                continue

            expected_levels = {
                MedDRALevel.LLT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT, MedDRALevel.LLT],
                MedDRALevel.PT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT],
            }.get(desc_level)

            if expected_levels and not has_complete_path(info["paths"], expected_levels):
                no_complete_path_count += 1

    # Save outputs

    try:
        with open(MEDDRA_TERMS_PATH, "w") as f:
            json.dump({"terms": sorted(all_terms)}, f, indent=4)
        print(f"✔ Terms saved to: {MEDDRA_TERMS_PATH}")
    except IOError as e:
        print(f"✘ Failed to write terms file: {e}")

    try:
        with open(MEDDRA_PATHS_PATH, "w") as f:
            json.dump(all_paths_info, f, indent=4)
        print(f"✔ Paths saved to: {MEDDRA_PATHS_PATH}")
    except IOError as e:
        print(f"✘ Failed to write paths file: {e}")

    # Print summary
    print("\nSummary:")
    print(f"  ➤ Total descendant nodes processed: {total_descendants}")
    print(f"  ➤ Descendants with NO complete path: {no_complete_path_count}")


def convert_meddra_descendant_terms_and_paths_from_json_to_csv() -> None:
    """
    Converts MedDRA descendant term and path data from JSON to CSV format.

    Reads:
        - meddra_positive_labels.json: List of unique descendant terms.
        - meddra_paths.json: Full path information for descendant nodes.

    Writes:
        - meddra_positive_labels.csv: CSV file with one column "term"
        - meddra_paths.csv: CSV file with columns:
              [HLGT_code, Descendant_code, Descendant_term, Path_index, Path_string]

    Raises:
        FileNotFoundError: If either of the required JSON input files is missing.
    """

    # --- Process labels JSON -> CSV
    if not os.path.exists(MEDDRA_LABELS_JSON_PATH):
        raise FileNotFoundError(f"Missing: {MEDDRA_LABELS_JSON_PATH}")

    with open(MEDDRA_LABELS_JSON_PATH, "r") as f:
        labels_data = json.load(f)
    terms = labels_data.get("terms", [])

    with open(MEDDRA_LABELS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["term"])
        writer.writerows([[term] for term in terms])

    print(f"✓ Saved term labels to {MEDDRA_LABELS_CSV_PATH}")

    # --- Process paths JSON -> CSV
    if not os.path.exists(MEDDRA_PATHS_JSON_PATH):
        raise FileNotFoundError(f"Missing: {MEDDRA_PATHS_JSON_PATH}")

    def format_path(path):
        return " -> ".join(f"{step[0]}: {step[1]}" for step in path)

    with open(MEDDRA_PATHS_JSON_PATH, "r") as f:
        paths_data = json.load(f)

    with open(MEDDRA_PATHS_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["HLGT_code", "Descendant_code", "Descendant_term", "Path_index", "Path"])

        for hlgt_key, descendants in paths_data.items():
            for desc_key, entry in descendants.items():
                term = entry.get("term", "")
                path_list = entry.get("paths", [])
                if path_list:
                    for i, path in enumerate(path_list, start=1):
                        writer.writerow([hlgt_key, desc_key, term, i, format_path(path)])
                else:
                    writer.writerow([hlgt_key, desc_key, term, "", ""])

    print(f"✓ Saved path summaries to {MEDDRA_PATHS_CSV_PATH}")


if __name__ == "__main__":
    get_meddra_positive_labels()
    convert_meddra_descendant_terms_and_paths_from_json_to_csv()
