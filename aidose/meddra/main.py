from aidose.meddra.graph import MedDRA
from aidose.meddra.utils import (parse_arguments, get_descendant_terms, get_descendant_info, get_all_ancestors,
                                 has_complete_path)

import ast
import os
import json


def main():
    args = parse_arguments()

    try:
        codes_list = ast.literal_eval(args.codes)
    except Exception as e:
        print("Error parsing codes argument:", e)
        return

    meddra = MedDRA()
    if not os.path.exists(args.data):
        print(f"Data directory not found: {args.data}")
        return
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


if __name__ == "__main__":
    main()
