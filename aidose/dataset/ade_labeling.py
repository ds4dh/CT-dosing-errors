from __future__ import annotations

from typing import Mapping, MutableMapping, Sequence, List, Dict

from aidose.dataset.ade import (
    PositiveTermMatch,
    LabelMatch,
)


def select_best_label(matches: Sequence[LabelMatch]) -> LabelMatch | None:
    """
    Pick the best label from a list of LabelMatch models:
    - Highest score wins.
    - If scores tie, lexicographically smaller label wins (determinism).

    Args:
        matches: A sequence of LabelMatch models.

    Returns:
        The best LabelMatch, or None if no matches.
    """
    best: LabelMatch | None = None
    for m in matches or []:
        if best is None:
            best = m
        else:
            if m.score > best.score:
                best = m
            elif m.score == best.score and m.label < best.label:
                best = m
    return best


def term_to_best_label_map_from_positive_terms(
        positive_terms: Mapping[str, PositiveTermMatch]
) -> Dict[str, str]:
    """
    Build a mapping: actual ADE term -> best canonical MedDRA label
    for a single study, using typed PositiveTermMatch values.

    Args:
        positive_terms: Mapping term -> PositiveTermMatch

    Returns:
        Dict of {term: best_label}
    """
    mapping: Dict[str, str] = {}
    for term, pt in (positive_terms or {}).items():
        best = select_best_label(pt.matches)
        if best is not None:
            mapping[term] = best.label
    return mapping


def canonical_labels_from_positive_terms(
        positive_terms: Mapping[str, PositiveTermMatch]
) -> List[str]:
    """
    Extract the unique set of best‑match canonical labels for a study.

    Args:
        positive_terms: Mapping term -> PositiveTermMatch

    Returns:
        Sorted list of unique canonical labels.
    """
    term_map = term_to_best_label_map_from_positive_terms(positive_terms)
    return sorted(set(term_map.values()))


def add_label_count_features_for_study(
        *,
        positive_terms: Mapping[str, PositiveTermMatch],
        canonical_label_columns: Sequence[str],
        out_features: MutableMapping[str, int],
) -> None:
    """
    Populate per‑label counts (sum of numAffected) for a single study.

    Behavior:
      - Ensures every 'label_<LABEL>' in canonical_label_columns exists in out_features (initialized to 0).
      - For each PositiveTermMatch, pick the best label.
      - Add that term’s numAffected (from trial‑level stats) to the corresponding feature bucket.

    Args:
        positive_terms: Mapping term -> PositiveTermMatch
        canonical_label_columns: Global set of canonical labels defining the feature columns.
        out_features: Mutable mapping to be filled with 'label_<LABEL>' integer counts.
    """
    # Initialize all requested columns
    for lbl in canonical_label_columns:
        out_features.setdefault(f"label_{lbl}", 0)

    # Sum counts by best label
    for term, pt in (positive_terms or {}).items():
        best = select_best_label(pt.matches)
        if best is None:
            continue
        key = f"label_{best.label}"
        if key in out_features:
            out_features[key] += int(pt.stats.numAffected)
