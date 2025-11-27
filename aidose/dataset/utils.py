from aidose.dataset.attribute import AttributesList
from aidose.ctgov.structures import Study, InterventionType, StudyType, Status, StatusModule

from aidose.meddra.graph import MedDRALevel
from aidose.meddra.utils import DescendantEntry

from datasets import DatasetInfo, Features, Value, Version as HFVersion, ClassLabel
from datasets import Sequence as HFSequence
from rapidfuzz import fuzz

from typing import Dict, List, Tuple, Any, Sequence, Iterable, Type
from enum import Enum
import re
from datetime import datetime
from importlib.metadata import version as pkg_version, PackageNotFoundError
import tomllib
from pathlib import Path
import subprocess


# =========================
# Trial Inclusion Criteria: Include only these studies to our dataset:
# =========================

def trial_study_type_is_interventional(study: Study) -> bool:
    dm = study.protocolSection.designModule
    if dm is None:
        return False
    if dm.studyType == StudyType.INTERVENTIONAL:
        return True
    else:
        return False


def trial_status_is_either_completed_or_terminated(study: Study) -> bool:
    if study.protocolSection.statusModule.overallStatus == Status.COMPLETED:
        return True
    elif study.protocolSection.statusModule.overallStatus == Status.TERMINATED:
        return True
    else:
        return False


def trial_study_has_a_completion_date(study: Study) -> bool:
    # Sometimes a trial is completed but has no completion date ! Example: NCT00939705
    sm = study.protocolSection.statusModule
    completion_date = get_study_completion_date(sm)
    if isinstance(completion_date, datetime):
        return True
    return False


def trial_completion_date_before_cutoff(study: Study, knowledge_cutoff_date: datetime) -> bool:
    sm = study.protocolSection.statusModule
    completion_date = get_study_completion_date(sm)
    if isinstance(completion_date, datetime) and completion_date <= knowledge_cutoff_date:
        return True
    return False


def trial_has_at_least_one_drug_intervention(study: Study) -> bool:
    if not study.protocolSection:
        return False
    if not study.protocolSection.armsInterventionsModule:
        return False
    if not study.protocolSection.armsInterventionsModule.interventions:
        return False

    return any(
        intervention.type == InterventionType.DRUG
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


def include_trial_after_sequential_filtering(study: Study, knowledge_cutoff_date: datetime | None) -> bool:
    """
    Sequentially filter trials based on:
    1. Study type must be "Interventional".
    2. Status must be either "Completed" or "Terminated".
    3. Study must have a completion date.
    4. Study completion date must be before or on the knowledge cutoff date.
    5. At least one intervention must be of type "DRUG".
    6. Presence of a 'resultsSection'.
    7. Presence of an 'adverseEventsModule' in the 'resultsSection'.

    Returns True if the trial passes all criteria.
    """
    if not trial_study_type_is_interventional(study):
        return False
    if not trial_status_is_either_completed_or_terminated(study):
        return False
    if not trial_study_has_a_completion_date(study):
        return False
    if knowledge_cutoff_date is not None:
        if not trial_completion_date_before_cutoff(study, knowledge_cutoff_date):
            return False
    if not trial_has_at_least_one_drug_intervention(study):
        return False
    if not trial_has_results_section(study):
        return False
    if not trial_has_adverse_events_module(study):
        return False

    return True


# =========================
# MedDRA and Fuzzy Matching Utilities
# =========================


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


def match_terms_fuzzy(
        candidate_terms: Dict[str, Any],
        positive_labels: List[str],
        match_threshold: int = 95,
) -> Dict[str, Dict[str, Any]]:
    """
    Performs fuzzy matching between candidate ADE terms and a list of positive MedDRA terms.

    Args:
        candidate_terms (Dict[str, Any]): Mapping from term → stats (e.g., from clinical view).
        positive_labels (List[str]): List of MedDRA positive ADE terms.
        match_threshold (int): Minimum similarity score (0–100) to count as a match.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of matched terms to their stats and matched labels:
            {
                "headache": {
                    "stats": {...},
                    "matches": [{"label": "Headache", "score": 96}]
                },
                ...
            }
    """
    normalized_labels = [(label, label.strip().lower()) for label in positive_labels]
    matched_terms: Dict[str, Dict[str, Any]] = {}

    for term, stats in candidate_terms.items():
        normalized_term = term.strip().lower()
        matches = [
            {"label": orig_label, "score": fuzz.ratio(normalized_term, norm_label)}
            for orig_label, norm_label in normalized_labels
            if fuzz.ratio(normalized_term, norm_label) >= match_threshold
        ]
        if matches:
            matched_terms[term] = {"stats": stats, "matches": matches}

    return matched_terms


# TODO: These are unnecessary stuff it seems. I'll just delete them soon:
def meddra_labels_to_csv_rows(terms: Iterable[str]) -> Tuple[List[str], List[List[str]]]:
    """
    Convert a set/iterable of MedDRA terms into CSV-friendly rows (no I/O).

    Returns
    -------
    header : list[str]
        ["term"]
    rows : list[list[str]]
        One row per term, sorted.
    """
    header = ["term"]
    rows = [[t] for t in sorted(terms)]
    return header, rows


def format_meddra_path(path: Sequence[Tuple[MedDRALevel | str, str]]) -> str:
    """
    Human-friendly formatter for a single MedDRA path:
        [(LEVEL, CODE), ...] -> "LEVEL: CODE -> LEVEL: CODE -> ..."

    Parameters
    ----------
    path : Sequence[tuple[MedDRALevel | str, str]]
        A path where each step is (level, code). Level may be an enum or str.

    Returns
    -------
    str
        Pretty string representation of the path.
    """
    parts: list[str] = []
    for lvl, code in path:
        lvl_str = lvl.name if isinstance(lvl, MedDRALevel) else str(lvl)
        parts.append(f"{lvl_str}: {code}")
    return " -> ".join(parts)


def meddra_paths_to_csv_rows(
        paths: Dict[str, Dict[str, DescendantEntry]]
) -> Tuple[List[str], List[List[str]]]:
    """
    Convert MedDRA descendant paths to CSV-friendly rows (no I/O).

    Input structure
    ---------------
    paths:
      {
        "HLGT_CODE@HLGT": {
           "DESC_CODE@PT":  {"term": str, "paths": [[(lvl, code), ...], ...]},
           "DESC_CODE@LLT": {...}
        },
        ...
      }

    Returns
    -------
    header : list[str]
        ["HLGT_code", "Descendant_code", "Descendant_term", "Path_index", "Path"]
    rows : list[list[str]]
        One row per descendant path. If no paths, an empty index and path string are emitted.
    """
    header = ["HLGT_code", "Descendant_code", "Descendant_term", "Path_index", "Path"]
    rows: list[list[str]] = []

    for hlgt_key, descendants in paths.items():
        for desc_key, entry in descendants.items():
            term: str = entry.get("term", "")
            path_list = entry.get("paths", [])

            if path_list:
                for i, path in enumerate(path_list, start=1):
                    rows.append([hlgt_key, desc_key, term, str(i), format_meddra_path(path)])
            else:
                rows.append([hlgt_key, desc_key, term, "", ""])

    return header, rows


# =========================
# Optional parity helpers
# =========================

def get_flow_group_descriptions(study: Study) -> List[str]:
    rs = study.resultsSection
    if not rs or not rs.participantFlowModule:
        return []
    groups = rs.participantFlowModule.groups or []
    out: List[str] = []
    for g in groups:
        desc = getattr(g, "description", None)
        if isinstance(desc, str) and desc.strip():
            out.append(desc)
    return out


def get_location_details(study: Study) -> List[str]:
    ps = study.protocolSection
    if not ps or not ps.contactsLocationsModule:
        return []
    locs = ps.contactsLocationsModule.locations or []
    rows: List[str] = []
    for loc in locs:
        city = getattr(loc, "city", None) or "N/A"
        state = getattr(loc, "state", None) or "N/A"
        country = getattr(loc, "country", None) or "N/A"
        geo = getattr(loc, "geoPoint", None)
        lon = getattr(geo, "lon", None) if geo else None
        lat = getattr(geo, "lat", None) if geo else None
        rows.append(" | ".join(map(str, [
            city, state, country,
            lon if lon is not None else "N/A",
            lat if lat is not None else "N/A",
        ])))
    return rows


# =========================
# Date-related helpers
# =========================
def get_study_completion_date(status_module: StatusModule) -> datetime | None:
    completion_date_struct = getattr(status_module, "completionDateStruct", None)
    if completion_date_struct:
        return completion_date_struct.date.dt
    else:
        # TODO: THIS LOGIC MAY BE WRONG ! Primary completion date and completion date are NOT the same !
        primary_completion_date_struct = getattr(status_module, "primaryCompletionDateStruct", None)
        if primary_completion_date_struct:
            return primary_completion_date_struct.date.dt

    return None


# =========================
# Dataset versioning
# =========================


def get_code_version(package_name: str) -> str | None:
    """Return the package/repo version from importlib"""
    try:
        return pkg_version(package_name)
    except PackageNotFoundError:
        pass

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return None

    data = tomllib.load(pyproject.open("rb"))

    v = (data or {}).get("project", {}).get("version")
    return str(v) if v else None


def get_git_sha(short: bool = False) -> str:
    """Return current git SHA if available."""
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
    return sha[:7] if short else sha


def make_dataset_info(
        *,
        dataset_version: str,
        description: str,
        features: Features | None = None,
        license_str: str | None = None,
        homepage: str | None = None,
        citation: str | None = None,
        package_name: str | None = None,
) -> DatasetInfo:
    code_version = get_code_version(package_name) if package_name else None
    git_sha = get_git_sha(short=False)

    built_by = []
    if package_name and code_version:
        built_by.append(f"\nThis dataset is built by {package_name} v{code_version}.")
    if git_sha:
        built_by.append(f"Commit {git_sha}.")
    desc = description.strip()
    if built_by:
        desc += "\n" + " ".join(built_by)

    return DatasetInfo(
        description=desc,
        version=HFVersion(dataset_version),
        features=features,
        license=license_str,
        homepage=homepage,
        citation=citation,
    )


# =========================
# Dataset type helpers and schema builders
# =========================

def build_struct_schema_from_attributes(
        attrib_names: List[str],
        attrib_types: List[Type],
        sample_values: List[Any]
) -> dict:
    """
    Builds a HF Schema dict by looking at declared types AND sample values
    (to detect lists of Enums).
    """
    schema_dict = {}

    for name, declared_type, value in zip(attrib_names, attrib_types, sample_values):

        # 1. Handle Enums (Scalar or List)
        if isinstance(declared_type, type) and issubclass(declared_type, Enum):
            # Extract all member names for the ClassLabel
            # We use member.name ensures alignment with the string conversion later
            labels = [member.name for member in declared_type]
            feature = ClassLabel(names=labels)

            # Check if the actual value is a list (Multi-label/Sequence)
            # We check the value because declared_type is the scalar Enum class in your design
            if isinstance(value, list):
                feature = HFSequence(feature)

            schema_dict[name] = feature

        # 2. Handle Standard Scalars
        elif declared_type is str:
            schema_dict[name] = Value("string")
        elif declared_type is int:
            schema_dict[name] = Value("int64")
        elif declared_type is float:
            schema_dict[name] = Value("float32")
        elif declared_type is bool:
            schema_dict[name] = Value("bool")
        elif declared_type is datetime:
            schema_dict[name] = Value("date32")
        else:
            # Fallback or error
            raise NotImplementedError(f"Type {declared_type} for attribute {name} is not supported.")

    return schema_dict


def serialize_attributes_for_hf(attributes: AttributesList) -> dict:
    """
    Converts an AttributesList to a dict, transforming Enums to their string names.
    HF Dataset.from_list cannot handle raw Enum objects.
    """
    row = {}
    for attr in attributes:
        val = attr.value

        # TODO: Why aren't we using the declared type here ? (attr.declared_type)
        if val is None:
            row[attr.name] = None
            continue

        # Handle List of Enums (convert [Enum.A, Enum.B] -> ["A", "B"])
        if isinstance(val, list) and len(val) > 0 and isinstance(val[0], Enum): # TODO: What if it is an attributes list but the val = [None] ?
            row[attr.name] = [e.name for e in val]

        # Handle Single Enum (convert Enum.A -> "A")
        elif isinstance(val, Enum):
            row[attr.name] = val.name

        # Handle Standard Types
        else:
            row[attr.name] = val

    return row
# TODO: Test the transition from multi-hot or one-hot to dense and list-based representations.
