from __future__ import annotations

from typing import Dict, List, Tuple, Any, Mapping

from pydantic import BaseModel, Field, ConfigDict

from aidose.ctgov.structures import Study, Event
from aidose.dataset.utils import match_terms_fuzzy


# ------------------------------------------------------------------------------
# Core stats containers (Pydantic)
# ------------------------------------------------------------------------------

class ADEEventStats(BaseModel):
    """Stats for a specific ADE term within a *single* group."""
    numAffected: int
    numAtRisk: int


class ADEGroupAggregate(BaseModel):
    """Aggregated ADE stats for a *single* event group."""
    population: int
    events: Dict[str, ADEEventStats]  # term -> stats


class ADEClinicalTermStats(BaseModel):
    """Trial-level summary for a single ADE term (summed across all groups)."""
    model_config = ConfigDict(frozen=True)
    numAffected: int
    numAtRisk: int


class LabelMatch(BaseModel):
    """A fuzzy match between a candidate term and a canonical MedDRA label."""
    model_config = ConfigDict(frozen=True)
    label: str
    score: int  # 0..100 similarity


class PositiveTermMatch(BaseModel):
    """A positive term finding: the ADE term, its clinical stats, and its matched labels."""
    model_config = ConfigDict(frozen=True)
    term: str
    stats: ADEClinicalTermStats
    matches: List[LabelMatch]


class ADEAnalysisResultForStudy(BaseModel):
    """
    The structured result for a processed study. Keeps everything typed and JSON-ready.
    """
    nctid: str
    ade_by_group: Dict[str, ADEGroupAggregate]  # group_id -> aggregate
    ade_clinical: Dict[str, ADEClinicalTermStats]  # term -> trial-level stats
    positive_terms: Dict[str, PositiveTermMatch] = Field(default_factory=dict)  # term -> match


# ------------------------------------------------------------------------------
# Computation utilities
# ------------------------------------------------------------------------------

def extract_group_populations(study: Study) -> Dict[str, int]:
    """
    Returns per-group population after validating consistency between seriousNumAtRisk and otherNumAtRisk.
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


def process_events_by_group(
        events: List[Event],
        group_populations: Mapping[str, int],
) -> Dict[str, Dict[str, ADEEventStats]]:
    """
    Aggregates stats per group for a list of adverse events.
    Returns: group_id -> (term -> ADEEventStats)
    """
    from collections import defaultdict

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
                    f"Inconsistent numAtRisk for group {group_id} in event '{term}': "
                    f"{num_at_risk} != {expected_population}."
                )

            if num_affected is None:
                continue

            if term in grouped_data[group_id]:
                # mutate allowed here (this model isn't frozen)
                grouped_data[group_id][term].numAffected += num_affected
            else:
                grouped_data[group_id][term] = ADEEventStats(
                    numAffected=num_affected, numAtRisk=expected_population
                )

    return grouped_data


def aggregate_ade_by_group(study: Study) -> Dict[str, ADEGroupAggregate]:
    """
    Aggregates ADEs by group across serious and other events.
    """
    ae_module = study.resultsSection.adverseEventsModule
    group_populations = extract_group_populations(study)

    serious = process_events_by_group(ae_module.seriousEvents, group_populations)
    other = process_events_by_group(ae_module.otherEvents, group_populations)

    aggregated: Dict[str, ADEGroupAggregate] = {}
    for group_id, population in group_populations.items():
        all_events = {**serious.get(group_id, {}), **other.get(group_id, {})}
        aggregated[group_id] = ADEGroupAggregate(population=population, events=all_events)

    return aggregated


def aggregate_ade_clinical_trial_view(study: Study) -> Dict[str, ADEClinicalTermStats]:
    """
    Sums ADE stats across groups into a trial-level view:
        term -> ADEClinicalTermStats(numAffected, numAtRisk)
    """
    grouped = aggregate_ade_by_group(study)
    clinical: Dict[str, ADEClinicalTermStats] = {}

    for group_agg in grouped.values():
        for term, stats in group_agg.events.items():
            if term not in clinical:
                clinical[term] = ADEClinicalTermStats(numAffected=0, numAtRisk=0)
            current = clinical[term]
            # create a new (frozen) instance with updated totals
            clinical[term] = ADEClinicalTermStats(
                numAffected=current.numAffected + stats.numAffected,
                numAtRisk=current.numAtRisk + stats.numAtRisk,
            )

    return clinical


def get_positive_ade_terms(event_stats_by_term: Mapping[str, ADEClinicalTermStats]) -> List[str]:
    """Returns ADE terms with a positive number of affected patients."""
    return [t for t, s in event_stats_by_term.items() if s.numAffected > 0]


def normalize_ade_error_message(msg: str) -> str:
    """Buckets common failure messages into stable categories."""
    if "Invalid at-risk numbers" in msg:
        return "Invalid at-risk numbers"
    if "Inconsistent at-risk numbers" in msg:
        return "Inconsistent at-risk numbers"
    if "found in stats but not in eventGroups" in msg:
        return "Group ID not in eventGroups"
    if "Inconsistent numAtRisk" in msg:
        return "Inconsistent numAtRisk"
    if "Invalid ADE term" in msg:
        return "Invalid ADE term"
    return "Other Error"


# ------------------------------------------------------------------------------
# Study-level processing with typed outputs
# ------------------------------------------------------------------------------

def _to_positive_term_matches(
        clinical_view: Mapping[str, ADEClinicalTermStats],
        fuzzy_output: Dict[str, Dict[str, Any]],
) -> Dict[str, PositiveTermMatch]:
    """
    Adapt match_terms_fuzzy(...) dict into typed PositiveTermMatch models.
    Expected fuzzy_output:
        {
          "<term>": {
             "stats": {"numAffected": int, "numAtRisk": int} OR ADEClinicalTermStats,
             "matches": [{"label": str, "score": int}, ...]
          },
          ...
        }
    """
    result: Dict[str, PositiveTermMatch] = {}

    for term, payload in fuzzy_output.items():
        stats_payload = payload.get("stats")
        if isinstance(stats_payload, ADEClinicalTermStats):
            stats = stats_payload
        else:
            stats = ADEClinicalTermStats(
                numAffected=int(stats_payload["numAffected"]),
                numAtRisk=int(stats_payload["numAtRisk"]),
            )

        matches_raw = payload.get("matches", [])
        matches = [LabelMatch(label=m["label"], score=int(m["score"])) for m in matches_raw]

        result[term] = PositiveTermMatch(term=term, stats=stats, matches=matches)

    return result


def process_study_for_ade_risks(
        study: Study,
        meddra_terms: List[str],
        match_threshold: int = 95,
) -> Tuple[ADEAnalysisResultForStudy | None, str | None]:
    """
    Computes group-level, trial-level ADE aggregates and fuzzy matches,
    returning a typed ADEAnalysisResultForStudy (or an error category).
    """
    try:
        ade_by_group = aggregate_ade_by_group(study)
        ade_clinical = aggregate_ade_clinical_trial_view(study)

        fuzzy = match_terms_fuzzy(
            candidate_terms=ade_clinical,  # term -> ADEClinicalTermStats
            positive_labels=meddra_terms,  # list[str]
            match_threshold=match_threshold,
        )
        positive_terms = _to_positive_term_matches(ade_clinical, fuzzy)

        return ADEAnalysisResultForStudy(
            nctid=study.protocolSection.identificationModule.nctId,
            ade_by_group=ade_by_group,
            ade_clinical=ade_clinical,
            positive_terms=positive_terms,
        ), None

    except Exception as e:
        return None, normalize_ade_error_message(str(e))
