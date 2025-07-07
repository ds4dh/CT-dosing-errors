from aidose.meddra.graph import MedDRA
from aidose import RESOURCES_DIR

import argparse
import ast
import json
import os


def get_all_ancestors(node, ancestors=None):
    """Recursively collect all ancestors of a node."""
    if ancestors is None:
        ancestors = set()
    for parent in node.parents:
        if parent not in ancestors:
            ancestors.add(parent)
            get_all_ancestors(parent, ancestors)
    return ancestors


def get_descendant_terms(hlgt_node, meddra, target_levels={"PT", "LLT"}):
    """
    For a given HLGT node, get a set of unique descendant node terms
    that are in the specified target levels.
    """
    descendant_terms = set()
    for node in meddra.nodes.values():
        if node.level in target_levels:
            if hlgt_node in get_all_ancestors(node):
                descendant_terms.add(node.term)
    return descendant_terms


def clean_paths(paths):
    """
    Remove any BYPASS placeholders from a list of paths.
    Each path is a list of "code@level" strings.
    """
    cleaned = []
    for path in paths:
        new_path = [step for step in path if not step.startswith("BYPASS@")]
        cleaned.append(new_path)
    return cleaned


def convert_paths_to_tuples(paths, meddra):
    """
    Convert a list of paths (lists of "code@level" strings) to lists of tuples.
    Each tuple is of the form ("code@level", term), where term is obtained from the MedDRA nodes.
    """
    converted_paths = []
    for path in paths:
        new_path = []
        for step in path:
            try:
                code, level = step.split("@")
            except ValueError:
                code = step
                level = ""
            node_key = (level, code)
            term = meddra.nodes[node_key].term if node_key in meddra.nodes else None
            new_path.append((step, term))
        converted_paths.append(new_path)
    return converted_paths


def is_full_path(path, expected_levels):
    """
    Check if a path (list of tuples) exactly matches the expected level sequence.
    Each tuple is ("code@level", term) so we extract the level from the string.
    """
    levels = []
    for step, _ in path:
        parts = step.split("@")
        level = parts[1] if len(parts) > 1 else ""
        levels.append(level)
    return levels == expected_levels


def has_complete_path(paths, expected_levels):
    """
    Returns True if at least one path in paths is a complete path.
    """
    for path in paths:
        if is_full_path(path, expected_levels):
            return True
    return False


def get_descendant_info(hlgt_node, meddra, target_levels={"PT", "LLT"}):
    """
    For a given HLGT node, return a dictionary mapping each descendant node
    (keyed by its unique "code@level") to its term and the list of cleaned paths
    (as tuples) from SOC.
    
    For each descendant:
      - If at least one full path exists (SOC->HLGT->HLT->PT->LLT for LLT,
        SOC->HLGT->HLT->PT for PT), only those full paths are kept.
      - Otherwise, the incomplete paths are kept.
    """
    descendant_info = {}
    for node in meddra.nodes.values():
        if node.level in target_levels:
            if hlgt_node in get_all_ancestors(node):
                raw_paths = meddra.find_paths(node.code, node.level)
                cleaned_paths = clean_paths(raw_paths)
                tuple_paths = convert_paths_to_tuples(cleaned_paths, meddra)
                
                # Define the expected complete path levels based on the descendant's level.
                if node.level == "LLT":
                    expected_levels = ["SOC", "HLGT", "HLT", "PT", "LLT"]
                elif node.level == "PT":
                    expected_levels = ["SOC", "HLGT", "HLT", "PT"]
                else:
                    expected_levels = None

                # If expected_levels is defined, filter for complete paths if any exist.
                if expected_levels is not None:
                    complete_paths = [path for path in tuple_paths if is_full_path(path, expected_levels)]
                    if complete_paths:
                        tuple_paths = complete_paths

                descendant_info[f"{node.code}@{node.level}"] = {
                    "term": node.term,
                    "paths": tuple_paths
                }
    return descendant_info


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract union of PT and LLT terms for given HLGT codes from MedDRA, "
            "and record full SOC-to-term paths as tuples (only complete paths are shown if they exist)."
        )
    )
    parser.add_argument(
        "--codes",
        type=str,
        default="[('HLGT', '10079145'), ('HLGT', '10079159')]",
        help=(
            "List of HLGT codes as a string, e.g. "
            "\"[('HLGT', '10079145'), ('HLGT', '10079159')]\" "
            "(default: \"[('HLGT', '10079145'), ('HLGT', '10079159')]\")"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(RESOURCES_DIR, "MedDRA_27_1_English", "MedAscii"),
        help='Path to MedDRA data directory',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESOURCES_DIR, "meddra_positive_labels.json"),
        help='Path to output JSON file for terms',
    )
    parser.add_argument(
        "--paths-output",
        type=str,
        default=os.path.join(RESOURCES_DIR, "meddra_paths.json"),
        help='Path to output JSON file for paths',
    )
    return parser.parse_args()

