import argparse
import os
from typing import List, Set, Tuple, Dict, TypedDict

from aidose.meddra.graph import MedDRALevel, Node, MedDRA


class DescendantEntry(TypedDict):
    term: str
    paths: List[List[Tuple[str, str | None]]]


def get_all_ancestors(node: Node, ancestors: Set[Node] | None = None) -> Set[Node]:
    """Recursively collect all ancestors of a node."""
    if ancestors is None:
        ancestors = set()
    for parent in node.parents:
        if parent not in ancestors:
            ancestors.add(parent)
            get_all_ancestors(parent, ancestors)
    return ancestors


def get_descendant_terms(
        hlgt_node: Node,
        meddra: MedDRA,
        target_levels: Set[MedDRALevel] | None = None,
) -> Set[str]:
    """Get descendant terms of a HLGT node at specific levels."""
    if target_levels is None:
        target_levels = {MedDRALevel.PT, MedDRALevel.LLT}

    return {
        node.term
        for node in meddra.nodes.values()
        if node.level in target_levels and hlgt_node in get_all_ancestors(node)
    }


def clean_paths(paths: List[List[str]]) -> List[List[str]]:
    """Remove any BYPASS placeholders from a list of paths."""
    return [[step for step in path if not step.startswith("BYPASS@")] for path in paths]


def convert_paths_to_tuples(
        paths: List[List[str]],
        meddra: MedDRA,
) -> List[List[Tuple[str, str | None]]]:
    """Convert a list of paths into (code@level, term) tuples."""
    converted_paths = []
    for path in paths:
        new_path = []
        for step in path:
            try:
                code, level_str = step.split("@")
                level = MedDRALevel.from_str(level_str)
                term = meddra.nodes.get((level, code)).term if (level, code) in meddra.nodes else None
            except Exception:
                code, term = step, None
            new_path.append((step, term))
        converted_paths.append(new_path)
    return converted_paths


def is_full_path(path: List[Tuple[str, str | None]], expected_levels: List[MedDRALevel]) -> bool:
    """Check if a path matches the expected level sequence."""
    try:
        actual_levels = [MedDRALevel.from_str(step.split("@")[1]) for step, _ in path]
        return actual_levels == expected_levels
    except Exception:
        return False


def has_complete_path(paths: List[List[Tuple[str, str | None]]], expected_levels: List[MedDRALevel]) -> bool:
    """Returns True if at least one path is a complete path."""
    return any(is_full_path(path, expected_levels) for path in paths)


def get_descendant_info(
        hlgt_node: Node,
        meddra: MedDRA,
        target_levels: Set[MedDRALevel] | None = None,
) -> Dict[str, DescendantEntry]:
    """
    Return all descendant node info under a HLGT node including term and cleaned full paths.
    """
    if target_levels is None:
        target_levels = {MedDRALevel.PT, MedDRALevel.LLT}

    descendant_info: dict[str, DescendantEntry] = {}

    for node in meddra.nodes.values():
        if node.level not in target_levels:
            continue
        if hlgt_node not in get_all_ancestors(node):
            continue

        raw_paths = meddra.find_paths(node.code, node.level)
        cleaned_paths = clean_paths(raw_paths)
        tuple_paths = convert_paths_to_tuples(cleaned_paths, meddra)

        # Define expected full path structure
        expected_levels = {
            MedDRALevel.LLT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT, MedDRALevel.LLT],
            MedDRALevel.PT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT],
        }.get(node.level, [])

        # Keep only full paths if any are present
        if expected_levels:
            complete_paths = [p for p in tuple_paths if is_full_path(p, expected_levels)]
            if complete_paths:
                tuple_paths = complete_paths

        descendant_info[f"{node.code}@{node.level}"] = DescendantEntry(
            term=node.term,
            paths=tuple_paths,
        )

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
            "\"[('HLGT', '10079145'), ('HLGT', '10079159')]\""
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join("resources", "MedDRA_27_1_English", "MedAscii"),
        help="Path to MedDRA data directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("resources", "meddra_positive_labels.json"),
        help="Output JSON file for terms",
    )
    parser.add_argument(
        "--paths-output",
        type=str,
        default=os.path.join("resources", "meddra_paths.json"),
        help="Output JSON file for paths",
    )
    return parser.parse_args()
