from __future__ import annotations

from .feature import Feature, FeaturesList
from .ade import ADEAnalysisResultForStudy
from .ade_labeling import term_to_best_label_map_from_positive_terms
from aidose.ctgov.structures import (
    Study,
    InterventionType,
    StudyType,
    Phase,
    PrimaryPurpose,
    Masking,
    Sex,
    AgencyClass,
    Status, ArmGroupType,
)

from typing import Any, List, Sequence, Dict
from enum import Enum

JJ_KEYWORDS = ("johnson", "janssen", "mcneil", "j&j", "j and j")


# TODO: Certain fields and Enum-types are missing in the feature extraction, e.g., ResponsibleParty, Role,
#  MoreInfoModule, CertaintyModule, etc.
# =========================
# Intervention accessors
# =========================

def get_protocol_interventions(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.interventions or []


def get_protocol_arm_groups(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.armGroups or []


# =========================
# Document helpers
# =========================

def _has_doc_flag(study: Study, flag_name: str) -> bool:
    ds = study.documentSection
    if not ds or not ds.largeDocumentModule:
        return False
    large_docs = ds.largeDocumentModule.largeDocs or []
    for doc in large_docs:
        val = getattr(doc, flag_name, None)
        if isinstance(val, bool) and val:
            return True
    return False


def has_protocol(study: Study) -> bool: return _has_doc_flag(study, "hasProtocol")


def has_sap(study: Study) -> bool:       return _has_doc_flag(study, "hasSap")


def has_icf(study: Study) -> bool:       return _has_doc_flag(study, "hasIcf")


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
# ADE helpers
# =========================

def _total_ade_population(ade_result: ADEAnalysisResultForStudy) -> int | None:
    if not ade_result.ade_by_group:
        return None
    return sum(group.population for group in ade_result.ade_by_group.values())


def _label_count_features_from_positive_terms(
        *,
        positive_terms: Dict[str, Any],
        canonical_label_cols: Sequence[str],
) -> List[Feature]:
    """
    Build features for each canonical label:
      Feature(name=f"label.{LABEL}", value=<sum numAffected>, declared_type=int)
    If a label is unused for the study, value=0.
    """
    # TODO: This is murky.
    feats: List[Feature] = []
    term_to_label = term_to_best_label_map_from_positive_terms(positive_terms)

    counts: Dict[str, int] = {lbl: 0 for lbl in canonical_label_cols}
    for term, best_label in term_to_label.items():
        payload = positive_terms.get(term)
        if not payload:
            continue
        stats = getattr(payload, "stats", None) or payload.get("stats")
        num_aff = getattr(stats, "numAffected", None)
        if num_aff is None and isinstance(stats, dict):
            num_aff = stats.get("numAffected")  # TODO: Do we need this?
        try:
            num_aff_int = int(num_aff)
        except (TypeError, ValueError):
            continue
        if best_label in counts:
            counts[best_label] += num_aff_int

    for label in canonical_label_cols:
        feats.append(Feature(name=f"label.{label}", value=counts.get(label, 0), declared_type=int))
    return feats


# =========================
# Main extractor
# =========================

def extract_features_for_study(
        study: Study,
        *,
        canonical_label_cols: Sequence[str],
        ade_analysis_results_for_study: ADEAnalysisResultForStudy,
) -> FeaturesList:
    """
    Build a typed feature list for a single Study.
    NOTE: This function does NOT perform one-hot/multi-hot expansion.
          Call `features.expand_enums()` later if you want hot encodings.
    """
    feats = FeaturesList()

    # --- Identification ---
    ps = study.protocolSection
    nctid = ps.identificationModule.nctId if ps and ps.identificationModule else None
    feats.append(Feature(name="nctId", value=nctid, declared_type=str))

    # --- Design ---
    design = ps.designModule if ps and ps.designModule else None
    feats.append(Feature(name="studyType", value=(design.studyType if design else None), declared_type=StudyType))

    # phases: list[Phase] (keep as Enum list; expand later)
    phases_list = list(design.phases) if (design and design.phases) else None
    feats.append(Feature(name="phases", value=phases_list, declared_type=Phase))

    enroll = design.enrollmentInfo if design and design.enrollmentInfo else None
    feats.append(Feature(name="enrollmentCount", value=(enroll.count if enroll else None), declared_type=int))

    d_info = design.designInfo if design and design.designInfo else None
    # TODO: Enum should be used for allocation and interventionModel:
    feats.append(Feature(name="allocation", value=(d_info.allocation if d_info else None), declared_type=str))
    feats.append(
        Feature(name="interventionModel", value=(d_info.interventionModel if d_info else None), declared_type=str))
    feats.append(
        Feature(name="primaryPurpose", value=(d_info.primaryPurpose if d_info else None), declared_type=PrimaryPurpose))

    masking_val = d_info.maskingInfo.masking if d_info and d_info.maskingInfo else None
    feats.append(Feature(name="masking", value=masking_val, declared_type=Masking))

    # --- Eligibility ---
    elig = ps.eligibilityModule if ps and ps.eligibilityModule else None
    feats.append(
        Feature(name="healthyVolunteers", value=(elig.healthyVolunteers if elig else None), declared_type=bool))
    feats.append(Feature(name="sex", value=(elig.sex if elig else None), declared_type=Sex))

    std_ages = list(elig.stdAges) if (elig and elig.stdAges) else None
    if std_ages and isinstance(std_ages[0], Enum):  # cautious
        feats.append(Feature(name="stdAges", value=std_ages, declared_type=type(std_ages[0])))

    # --- Sponsor & status ---
    sc = ps.sponsorCollaboratorsModule if ps and ps.sponsorCollaboratorsModule else None
    lead = sc.leadSponsor if sc and sc.leadSponsor else None
    lead_name = lead.name if lead else None
    feats.append(Feature(name="leadSponsorName", value=lead_name, declared_type=str))
    feats.append(Feature(name="leadSponsorClass", value=(lead.class_ if lead else None), declared_type=AgencyClass))

    status = ps.statusModule if ps and ps.statusModule else None
    feats.append(Feature(name="overallStatus", value=(status.overallStatus if status else None), declared_type=Status))

    oversight = ps.oversightModule if ps and ps.oversightModule else None
    feats.append(
        Feature(name="oversightHasDmc", value=(oversight.oversightHasDmc if oversight else None), declared_type=bool))

    feats.append(Feature(name="isJJ", value=bool(lead_name and any(k in lead_name.lower() for k in JJ_KEYWORDS)),
                         declared_type=bool))
    # --- Description Module ---
    feats.append(Feature(name="briefSummary",
                         value=ps.descriptionModule.briefSummary,
                         declared_type=str))

    feats.append(Feature(name="detailedDescription",
                         value=ps.descriptionModule.detailedDescription,
                         declared_type=str))
    # --- Conditions Module ---
    feats.append(Feature(name="conditions",
                         value=" ".join(ps.conditionsModule.conditions) if ps and ps.conditionsModule else None,
                         declared_type=str))

    feats.append(Feature(name="conditionsKeywords",
                         value=" ".join(ps.conditionsModule.keywords) if ps and ps.conditionsModule else None,
                         declared_type=str))

    # --- Documents ---
    feats.append(Feature(name="hasProtocol", value=has_protocol(study), declared_type=bool))
    feats.append(Feature(name="hasSap", value=has_sap(study), declared_type=bool))
    feats.append(Feature(name="hasIcf", value=has_icf(study), declared_type=bool))

    # --- Arms & interventions ---
    arms = get_protocol_arm_groups(study)

    num_arms = len(arms)
    feats.append(Feature(name="numArms", value=num_arms, declared_type=int))

    arm_descriptions = [getattr(arm, "description", None) for arm in arms]
    feats.append(Feature(name="armDescriptions",
                         value=" ".join(
                             f"arm {i + 1}: {s}" for i, s in enumerate(arm_descriptions)) if arm_descriptions else None,
                         declared_type=str))
    arm_group_types = [getattr(arm, "type", None) for arm in arms]
    feats.append(Feature(name="armGroupTypes", value=(arm_group_types if arm_group_types else None),
                         declared_type=ArmGroupType))

    interventions = get_protocol_interventions(study)
    feats.append(Feature(name="numInterventions", value=len(interventions), declared_type=int))

    itypes = [getattr(itv, "type", None) for itv in interventions]
    feats.append(Feature(name="interventionTypes", value=itypes, declared_type=InterventionType))

    i_descriptions = [getattr(itv, "description", None) for itv in interventions]
    feats.append(Feature(name="interventionDescriptions",
                         value=" ".join(f"intervention {i + 1}: {s}" for i, s in
                                        enumerate(i_descriptions)) if i_descriptions else None,
                         declared_type=str))
    i_names = [getattr(itv, "name", None) for itv in interventions]
    feats.append(Feature(name="interventionNames",
                         value=" ".join(
                             f"intervention {i + 1}: {s}" for i, s in enumerate(i_names)) if i_names else None,
                         declared_type=str))

    # --- Locations ---
    loc_details = get_location_details(study)
    feats.append(Feature(name="numLocations", value=len(loc_details), declared_type=int))
    feats.append(
        Feature(name="locationDetails", value="\n".join(loc_details) if loc_details else None, declared_type=str))

    # --- ADE enrichment ---
    ade = ade_analysis_results_for_study
    feats.append(Feature(name="num_ct_level_ade_terms", value=len(ade.ade_clinical), declared_type=int))
    feats.append(Feature(name="ct_level_ade_population", value=_total_ade_population(ade), declared_type=int))
    feats.append(Feature(name="num_positive_terms_matched", value=len(ade.positive_terms), declared_type=int))

    # canonical label counts
    feats.extend(_label_count_features_from_positive_terms(
        positive_terms=ade.positive_terms,
        canonical_label_cols=canonical_label_cols,
    ))

    return feats
