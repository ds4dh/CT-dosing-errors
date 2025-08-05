from aidose.ctgov.structures import Study, Event

from rapidfuzz import fuzz

from collections import defaultdict
from typing import Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class ADEEventStats:
    """
    Container for statistics related to a specific adverse drug event (ADE) within a group.

    Attributes:
        numAffected (int): Number of patients affected by the ADE.
        numAtRisk (int): Number of patients at risk in the group.
    """
    numAffected: int
    numAtRisk: int


@dataclass
class ADEGroupAggregate:
    """
    Aggregated ADE statistics for a specific event group.

    Attributes:
        population (int): Number of patients at risk in the group.
        events (Dict[str, ADEEventStats]): Mapping from ADE term to its aggregated stats.
    """
    population: int
    events: Dict[str, ADEEventStats]


def extract_group_populations(study: Study) -> Dict[str, int]:
    """
    Extracts and validates the population size for each event group in the study.

    Ensures that seriousNumAtRisk and otherNumAtRisk are present and consistent.

    Args:
        study (Study): The clinical trial study object.

    Returns:
        Dict[str, int]: Mapping from group ID to population size.

    Raises:
        ValueError: If any group has missing or inconsistent at-risk numbers.
    """
    event_groups = study.resultsSection.adverseEventsModule.eventGroups
    group_populations: Dict[str, int] = {}

    for group in event_groups:
        group_id = group.id
        serious_at_risk = group.seriousNumAtRisk
        other_at_risk = group.otherNumAtRisk

        if serious_at_risk is None or other_at_risk is None:
            raise ValueError(f"Invalid at-risk numbers for group {group_id}.")

        if serious_at_risk != other_at_risk:
            raise ValueError(
                f"Inconsistent at-risk numbers for group {group_id}: {serious_at_risk} vs {other_at_risk}."
            )

        group_populations[group_id] = serious_at_risk

    return group_populations


def process_events_by_group(events: list[Event], group_populations: Dict[str, int]) -> Dict[
    str, Dict[str, ADEEventStats]]:
    """
    Processes a list of adverse events and aggregates statistics per group.

    Args:
        events (list[Event]): List of serious or other adverse events.
        group_populations (Dict[str, int]): Mapping from group ID to population size.

    Returns:
        Dict[str, Dict[str, ADEEventStats]]: Nested mapping of group ID to event term to statistics.

    Raises:
        ValueError: For missing ADE terms, unknown group IDs, or inconsistent numAtRisk.
    """
    grouped_data: Dict[str, Dict[str, ADEEventStats]] = defaultdict(dict)

    for event in events:
        if not event.term or not event.term.strip():
            raise ValueError("Invalid ADE term: term is missing or empty.")

        term = event.term.strip()

        for stat in event.stats:
            group_id = stat.groupId
            num_affected = stat.numAffected
            num_at_risk = stat.numAtRisk

            if group_id not in group_populations:
                raise ValueError(f"Group ID {group_id} found in stats but not in eventGroups.")

            expected_population = group_populations[group_id]

            if num_at_risk != expected_population:
                raise ValueError(
                    f"Inconsistent numAtRisk for group {group_id} in event '{term}': {num_at_risk} != {expected_population}."
                )

            if num_affected is None:
                continue

            if term in grouped_data[group_id]:
                grouped_data[group_id][term].numAffected += num_affected
            else:
                grouped_data[group_id][term] = ADEEventStats(
                    numAffected=num_affected, numAtRisk=expected_population
                )

    return grouped_data


def aggregate_ade_by_group(study: Study) -> Dict[str, ADEGroupAggregate]:
    """
    Aggregates adverse drug events (ADEs) by group across serious and other events.

    Args:
        study (Study): A study object containing the results section.

    Returns:
        Dict[str, ADEGroupAggregate]: Aggregated ADE data per event group.
    """
    ae_module = study.resultsSection.adverseEventsModule
    group_populations = extract_group_populations(study)

    serious = process_events_by_group(ae_module.seriousEvents, group_populations)
    other = process_events_by_group(ae_module.otherEvents, group_populations)

    aggregated: Dict[str, ADEGroupAggregate] = {}

    for group_id, population in group_populations.items():
        all_events = {**serious.get(group_id, {}), **other.get(group_id, {})}
        aggregated[group_id] = ADEGroupAggregate(
            population=population,
            events=all_events
        )

    return aggregated


def aggregate_ade_clinical_trial_view(study: Study) -> Tuple[Dict[str, Dict[str, int]], int]:
    # TODO: Refactor this properly.
    """
    Aggregates ADE statistics into a unified clinical trial view.

    For each unique ADE term, this function sums the `numAffected` and `numAtRisk` values
    across all event groups in the study.

    Args:
        study (Study): A structured Study object containing parsed clinical trial data.

    Returns:
        Tuple[Dict[str, Dict[str, int]], int]:
            - A dictionary mapping ADE terms to summed statistics:
              {
                "Event Term A": {"numAffected": <total>, "numAtRisk": <total>},
                ...
              }
            - The total population at risk across all event groups.
    """
    grouped_ade_data = aggregate_ade_by_group(study)

    clinical_view: Dict[str, Dict[str, int]] = {}
    total_population = 0

    for group_aggregate in grouped_ade_data.values():
        total_population += group_aggregate.population

        for ade_term, stats in group_aggregate.events.items():
            if ade_term not in clinical_view:
                clinical_view[ade_term] = {"numAffected": 0, "numAtRisk": 0}

            clinical_view[ade_term]["numAffected"] += stats.numAffected
            clinical_view[ade_term]["numAtRisk"] += stats.numAtRisk

    return clinical_view, total_population


def get_positive_ade_terms(event_stats_by_term: Dict[str, ADEEventStats]) -> list[str]:
    """
    Returns a list of ADE terms that have a positive number of affected patients.

    Args:
        event_stats_by_term (Dict[str, ADEEventStats]): Mapping from ADE term to its stats.

    Returns:
        list[str]: A list of ADE terms with numAffected > 0.
    """
    return [
        term for term, stats in event_stats_by_term.items()
        if stats.numAffected is not None and stats.numAffected > 0
    ]


def normalize_ade_error_message(msg: str) -> str:
    """
    Normalizes an ADE-related error message into a predefined error category.

    Categories include:
      - "Invalid at-risk numbers"
      - "Inconsistent at-risk numbers"
      - "Group ID not in eventGroups"
      - "Inconsistent numAtRisk"
      - "Invalid ADE term"
      - "Other Error"

    Args:
        msg (str): The original error message.

    Returns:
        str: A normalized error category.
    """
    if "Invalid at-risk numbers" in msg:
        return "Invalid at-risk numbers"
    elif "Inconsistent at-risk numbers" in msg:
        return "Inconsistent at-risk numbers"
    elif "found in stats but not in eventGroups" in msg:
        return "Group ID not in eventGroups"
    elif "Inconsistent numAtRisk" in msg:
        return "Inconsistent numAtRisk"
    elif "Invalid ADE term" in msg:
        return "Invalid ADE term"
    else:
        return "Other Error"


def process_study_for_ade_risks(
        study: Study,
        meddra_terms: list[str],
        match_threshold: int = 95,
) -> Tuple[Dict[str, Any], str | None]:
    """
    Processes a single Study object to compute ADE aggregates and match positive ADE terms.

    Args:
        study (Study): Parsed Study instance.
        meddra_terms (list[str]): List of positive MedDRA ADE terms.
        match_threshold (int): Similarity threshold for fuzzy matching (0–100).

    Returns:
        Tuple:
            - Dict[str, Any]: A dictionary containing:
                - 'study': the original Study
                - 'ade_by_group': dict of group-level ADEs
                - 'ade_clinical': clinical-trial level ADE view
                - 'positive_terms': matched ADE terms
            - str | None: Normalized error message if failed, otherwise None
    """
    try:
        # Aggregate ADEs
        grouped = aggregate_ade_by_group(study)
        clinical_view, _ = aggregate_ade_clinical_trial_view(study)

        # Match terms to positive MedDRA labels
        normalized_labels = [(label, label.strip().lower()) for label in meddra_terms]
        positive_terms: Dict[str, Any] = {}

        for term, stats in clinical_view.items():
            normalized_term = term.strip().lower()
            matches = [
                {"label": orig_label, "score": fuzz.ratio(normalized_term, norm_label)}
                for orig_label, norm_label in normalized_labels
                if fuzz.ratio(normalized_term, norm_label) >= match_threshold
            ]
            if matches:
                positive_terms[term] = {"stats": stats, "matches": matches}

        return {
            "study": study,
            "ade_by_group": grouped,
            "ade_clinical": clinical_view,
            "positive_terms": positive_terms,
        }, None

    except Exception as e:
        return {}, normalize_ade_error_message(str(e))
