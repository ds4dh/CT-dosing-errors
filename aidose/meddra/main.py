from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import (parse_arguments, get_descendant_terms, get_descendant_info, get_all_ancestors,
                                 has_complete_path)

from aidose import RESOURCES_DIR

import ast
import os
import json
import csv

# Define file paths (adjust these as needed)
positive_labels_json = os.path.join(RESOURCES_DIR, "meddra_positive_labels.json")
mapped_labels_json = os.path.join(RESOURCES_DIR, "mapped_labels.json")
paths_json = os.path.join(RESOURCES_DIR, "meddra_paths.json")

positive_labels_csv = os.path.join(RESOURCES_DIR, "meddra_positive_labels.csv")
mapped_labels_csv = os.path.join(RESOURCES_DIR, "mapped_labels.csv")
paths_csv = os.path.join(RESOURCES_DIR, "meddra_paths.csv")


def main_a0():
    args = parse_arguments()

    try:
        codes_list = ast.literal_eval(args.codes)
    except Exception as e:
        print("Error parsing codes argument:", e)
        return

    meddra = MedDRA()
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data directory not found: {args.data}")

    meddra.load_data(args.data)

    # Produce the same union of descendant terms as before.
    all_terms = set()
    # Also collect full paths info for descendant nodes per HLGT code.
    all_paths_info = {}

    # Counter for descendant instances with 0 complete paths.
    instances_with_no_complete_paths = 0
    total_descendant_instances = 0

    for level, code in codes_list:
        node_key = (level, code)
        if node_key not in meddra.nodes:
            print(f"Code {node_key} not found in MedDRA data.")
            continue
        hlgt_node = meddra.nodes[node_key]

        # Get the descendant terms for the original output.
        descendant_terms = get_descendant_terms(hlgt_node, meddra)
        all_terms |= descendant_terms

        # Get the descendant info including full paths as tuples.
        descendant_info = get_descendant_info(hlgt_node, meddra)
        all_paths_info[f"{code}@{level}"] = descendant_info

        # Count instances with 0 complete paths.
        for desc_key, info in descendant_info.items():
            total_descendant_instances += 1
            # Determine expected levels from the descendant key (format "code@level").
            try:
                _, desc_level = desc_key.split("@")
            except ValueError:
                desc_level = ""
            if desc_level == "LLT":
                expected = ["SOC", "HLGT", "HLT", "PT", "LLT"]
            elif desc_level == "PT":
                expected = ["SOC", "HLGT", "HLT", "PT"]
            else:
                expected = None

            if expected is not None:
                # Check if any of the paths is a complete path.
                if not has_complete_path(info["paths"], expected):
                    instances_with_no_complete_paths += 1

    # Write the terms file (unchanged output).
    output_dict = {"terms": sorted(list(all_terms))}
    try:
        with open(args.output, "w") as f:
            json.dump(output_dict, f, indent=4)
        print(f"Terms output successfully written to {args.output}")
    except IOError as e:
        print(f"Error writing to terms output file: {e}")

    # Write the paths file (with paths as tuples and only full paths if available).
    try:
        with open(args.paths_output, "w") as f:
            json.dump(all_paths_info, f, indent=4)
        print(f"Paths output successfully written to {args.paths_output}")
    except IOError as e:
        print(f"Error writing to paths output file: {e}")

    # Print statistics
    print("Total descendant instances processed:", total_descendant_instances)
    print("Number of descendant instances with 0 complete paths:", instances_with_no_complete_paths)


def main_a1():
    # ----------------------------
    # Process meddra_positive_labels.json
    # ----------------------------
    if os.path.exists(positive_labels_json):
        with open(positive_labels_json, "r") as f:
            pos_labels = json.load(f)

        # Expecting a JSON object with a key "terms"
        terms = pos_labels.get("terms", [])

        with open(positive_labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["term"])
            for term in terms:
                writer.writerow([term])
        print(f"Saved {positive_labels_csv}")
    else:
        raise FileNotFoundError(f"File not found: {positive_labels_json}")

        # ----------------------------
        # Process mapped_labels.json
        # ----------------------------
    if os.path.exists(mapped_labels_json):
        with open(mapped_labels_json, "r") as f:
            mapped_labels = json.load(f)

        # Assuming it's a list of terms
        with open(mapped_labels_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["term"])
            for term in mapped_labels:
                writer.writerow([term])
        print(f"Saved {mapped_labels_csv}")
    else:
        raise FileNotFoundError(f"File not found: {mapped_labels_json}")

        # ----------------------------
        # Process meddra_paths.json
        # ----------------------------

    def format_path(path):
        """
        Convert a list of tuples (code@level, term) into a formatted string.
        Each tuple becomes "code@level: term" and steps are joined by " -> ".
        """
        return " -> ".join([f"{step[0]}: {step[1]}" for step in path])

    if os.path.exists(paths_json):
        with open(paths_json, "r") as f:
            paths_data = json.load(f)

        with open(paths_csv, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header row
            writer.writerow(["HLGT_code", "Descendant_code", "Descendant_term", "Path_index", "Path"])

            # Iterate over each HLGT key
            for hlgt_key, descendant_info in paths_data.items():
                # descendant_info is a dictionary mapping descendant node to its info.
                for descendant_code, info in descendant_info.items():
                    descendant_term = info.get("term", "")
                    paths_list = info.get("paths", [])

                    if paths_list:
                        for idx, path in enumerate(paths_list, start=1):
                            formatted = format_path(path)
                            writer.writerow([hlgt_key, descendant_code, descendant_term, idx, formatted])
                    else:
                        writer.writerow([hlgt_key, descendant_code, descendant_term, "", ""])
        print(f"Saved {paths_csv}")
    else:
        raise FileNotFoundError(f"File not found: {paths_json}")


if __name__ == "__main__":
    main_a0()
    main_a1()
