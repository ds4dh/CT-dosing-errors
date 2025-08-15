from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

from aidose.meddra.graph import MedDRA, MedDRALevel
from aidose.meddra.utils import (
    get_descendant_terms,
    get_descendant_info,
    has_complete_path,
    DescendantEntry
)


@dataclass(frozen=True)
class MedDRADescendantResult:
    """
    In-memory result for MedDRA descendant mining.

    Attributes
    ----------
    terms : set[str]
        Unique descendant PT/LLT terms under the provided HLGT nodes.
    paths : dict[str, dict[str, DescendantEntry]]
        Mapping "HLGT_CODE@HLGT" -> {"DESC_CODE@LEVEL": DescendantEntry}.
    total_descendants : int
        Total descendant nodes processed (PT + LLT).
    no_complete_path_count : int
        Number of descendants without a complete path along the expected chain.
    """
    terms: Set[str]
    paths: Dict[str, Dict[str, DescendantEntry]]
    total_descendants: int
    no_complete_path_count: int


def build_meddra_descendants(
        meddra: MedDRA,
        hlgt_codes: Iterable[Tuple[MedDRALevel, str]],
) -> MedDRADescendantResult:
    """
    Compute all descendant PT/LLT terms and full hierarchical paths for a set of HLGT codes.

    The function:
      - Collects unique descendant terms (PT/LLT).
      - Builds full paths (SOC → HLGT → HLT → PT → [LLT]).
      - Counts total descendants and how many lack a complete expected path.

    Parameters
    ----------
    meddra : MedDRA
        An initialized MedDRA graph (already loaded).
    hlgt_codes : Iterable[tuple[MedDRALevel, str]]
        Iterable of (level, code) tuples; typically HLGT nodes.

    Returns
    -------
    MedDRADescendantResult
        Pure, in-memory artifact ready for further processing or export.
    """
    all_terms: Set[str] = set()
    all_paths_info: Dict[str, Dict[str, DescendantEntry]] = {}

    total_descendants = 0
    no_complete_path_count = 0

    for level, code in hlgt_codes:
        key = (level, code)
        node = meddra.nodes.get(key)
        if node is None:
            # Skip missing nodes; the caller can decide how to report/log this.
            continue

        # unique descendant terms
        descendant_terms = get_descendant_terms(node, meddra)
        all_terms |= descendant_terms

        # full path information
        descendant_info = get_descendant_info(node, meddra)
        all_paths_info[f"{code}@{level.name}"] = descendant_info

        # completeness accounting
        for desc_key, entry in descendant_info.items():
            total_descendants += 1

            # desc_key: "CODE@LEVEL"
            try:
                _, level_str = desc_key.split("@", 1)
                desc_level = MedDRALevel.from_str(level_str)
            except Exception:
                continue

            expected_chain = {
                MedDRALevel.LLT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT, MedDRALevel.LLT],
                MedDRALevel.PT: [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT],
            }.get(desc_level)

            if expected_chain and not has_complete_path(entry["paths"], expected_chain):
                no_complete_path_count += 1

    return MedDRADescendantResult(
        terms=all_terms,
        paths=all_paths_info,
        total_descendants=total_descendants,
        no_complete_path_count=no_complete_path_count,
    )
