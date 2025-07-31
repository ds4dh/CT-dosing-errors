from aidose.ctgov.structures import Study

import re


def trial_has_at_least_one_drug_intervention(study: Study) -> bool:
    if not study.protocolSection:
        return False
    if not study.protocolSection.armsInterventionsModule:
        return False
    if not study.protocolSection.armsInterventionsModule.interventions:
        return False

    return any(
        intervention.type and intervention.type.strip().upper() == "DRUG"
        for intervention in study.protocolSection.armsInterventionsModule.interventions
    )


def trial_has_results_section(study: Study) -> bool:
    results = getattr(study, "resultsSection", None)
    return results is not None and bool(vars(results))


def trial_has_adverse_events_module(study: Study) -> bool:
    results_section = getattr(study, "resultsSection", None)
    if not results_section:
        return False

    return getattr(results_section, "adverseEventsModule", None) is not None


def include_trial_after_sequential_filtering(study: Study) -> bool:
    """
    Sequentially filter trials based on:
      1. At least one intervention with type "DRUG".
      2. Presence of a 'resultsSection'.
      3. Presence of an 'adverseEventsModule' in the 'resultsSection'.

    Returns True if the trial passes all criteria.
    """
    if not trial_has_at_least_one_drug_intervention(study):
        return False
    if not trial_has_results_section(study):
        return False
    if not trial_has_adverse_events_module(study):
        return False

    return True


def sanitize_number_from_string(input_str_with_some_numerical_val: str) -> float | None:
    """
    Extracts the first well-formed numeric token from a string and returns it as a float.

    Accepts numbers with:
        - Optional single leading minus
        - Digits, commas
        - One decimal point

    Rejects malformed patterns like:
        '--1.0.0', '1.2.3', 'value: --12.3'

    Args:
        input_str_with_some_numerical_val (str): Input string to parse.

    Returns:
        float | None: Parsed float if valid, else None.
    """
    if not isinstance(input_str_with_some_numerical_val, str):
        raise TypeError("Input must be a string.")

    # Reject clearly malformed patterns in the full string
    if re.search(r"--|\.\.|(\d\.\d\.\d)", input_str_with_some_numerical_val):
        return None

    # Look for possible numeric candidates
    matches = re.findall(r"-?\d[\d,]*\.?\d*", input_str_with_some_numerical_val)
    for candidate in matches:
        # Skip empty or obviously malformed entries
        if candidate.count("-") > 1 or candidate.count(".") > 1:
            continue
        try:
            return float(candidate.replace(",", ""))
        except ValueError:
            continue

    return None
