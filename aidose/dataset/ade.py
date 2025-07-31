from aidose.ctgov.structures import Study, Event

from collections import defaultdict
from typing import Dict
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
