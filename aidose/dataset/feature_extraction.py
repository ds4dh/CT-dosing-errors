from __future__ import annotations

from aidose.ctgov.structures import (
    Study,
    InterventionType,
    Phase,
    PrimaryPurpose,
    Masking,
    Sex,
    AgencyClass,
    Status, ArmGroupType,
)
from aidose.ctgov.utils_protocol import (has_protocol,
                                         has_sap,
                                         has_icf,
                                         get_protocol_arm_groups,
                                         get_protocol_interventions)

from aidose.ctgov.utils_protocol import concatenate_pdf_texts_for_nctid, get_large_protocols_pdf_links

from aidose.ctgov import CTGOV_DATASET_EXTENSIONS_PATH

from .utils import get_location_details, get_study_completion_date

from .attribute import Attribute, AttributesList
from .ade import ADEAnalysisResultForStudy
from .ade_labeling import term_to_best_label_map_from_positive_terms

from statsmodels.stats.proportion import proportion_confint

from typing import Any, List, Sequence, Dict
from enum import Enum
import os
from datetime import datetime

JJ_KEYWORDS = ("johnson", "janssen", "mcneil", "j&j", "j and j")

CANONICAL_COUNT_PREFIX = "count."

ATTRIBS_FEATURE_PREFIX = "f."
ATTRIBS_LABEL_PREFIX = "l."
ATTRIBS_METADATA_PREFIX = "m."


# TODO: Certain fields and Enum-types are missing in the feature extraction, e.g., ResponsibleParty, Role,
#  MoreInfoModule, CertaintyModule, etc.


# =========================
# ADE helpers
# =========================

def _total_ade_population(ade_result: ADEAnalysisResultForStudy) -> int | None:
    if not ade_result.ade_by_group:
        return None
    return sum(group.population for group in ade_result.ade_by_group.values())


def _get_ade_count_attributes_from_positive_terms(
        *,
        positive_terms: Dict[str, Any],
        canonical_label_cols: Sequence[str],
) -> List[Attribute]:
    """
    Build attributes for each canonical label:
    """
    # TODO: This is murky.
    attribs: List[Attribute] = []
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

    for col in canonical_label_cols:
        attribs.append(Attribute(name=f"{CANONICAL_COUNT_PREFIX}{col}", value=counts.get(col, 0), declared_type=int))
    return attribs


def add_new_label_features_from_existing(attribs: AttributesList,
                                         alpha_wilson: float,
                                         wilson_proba_threshold: float
                                         ) -> AttributesList:
    def get_wilson_lower_bound(x: int, n: int, alpha: float) -> float:
        """
        Computes the lower bound of the Wilson confident interval.

        :param x: Number of successes (e.g., number errors in the trials).
        :param n: Total number of person involved in the trials.
        :param alpha: Significance level for the confidence interval (default is 0.05).
        :return: Lower bound of the Wilson score interval.
        """
        if n == 0:
            # in some trials, we have no people at risk -> in this case, we are sure the error rate will be 0 -> lower bound is also zer0
            return 0.0
        lower, _ = proportion_confint(count=x, nobs=n, alpha=alpha, method='wilson')
        return lower

    sum_dosing_error = Attribute(name="sum_dosing_errors",
                                 value=sum(
                                     [feat.value for feat in attribs if feat.name.startswith(CANONICAL_COUNT_PREFIX)]),
                                 declared_type=int)

    ct_level_ade_population = next((feat.value for feat in attribs if feat.name == "ct_level_ade_population"))

    dosing_error_rate = Attribute(name="dosing_error_rate",
                                  value=(
                                      sum_dosing_error.value / ct_level_ade_population
                                      if ct_level_ade_population else None),
                                  declared_type=float)

    wilson_lower_bound = Attribute(name="wilson_lower_bound",
                                   value=get_wilson_lower_bound(
                                       x=sum_dosing_error.value,
                                       n=ct_level_ade_population,
                                       alpha=alpha_wilson),
                                   declared_type=float)

    wilson_label = Attribute(name="wilson_label",
                             value=(1 if wilson_lower_bound.value >= wilson_proba_threshold else 0),
                             # TODO: Check this with Félicien
                             declared_type=int)

    attribs.append(sum_dosing_error)
    attribs.append(dosing_error_rate)
    attribs.append(wilson_lower_bound)
    attribs.append(wilson_label)

    return attribs


# =========================
# Main extractors
# =========================

def extract_attributes_from_study(
        study: Study,
        canonical_label_cols: Sequence[str],
        ade_analysis_results_for_study: ADEAnalysisResultForStudy,
        alpha_wilson: float,
        wilson_proba_threshold: float
) -> AttributesList:
    """
    Extract attributes from a Study.
    This includes:
      - features for training
      - labels
      - metadata (non-training features)
    """

    attribs_features = AttributesList()
    attribs_labels = AttributesList()
    attribs_metadata = AttributesList()

    ps = study.protocolSection

    # --- Design ---
    design = ps.designModule if ps and ps.designModule else None

    # phases: list[Phase] (keep as Enum list; expand later)
    phases_list = list(design.phases) if (design and design.phases) else None
    attribs_features.append(Attribute(name="phases", value=phases_list, declared_type=Phase))

    enroll = design.enrollmentInfo if design and design.enrollmentInfo else None
    attribs_features.append(
        Attribute(name="enrollmentCount", value=(enroll.count if enroll else None), declared_type=int))

    d_info = design.designInfo if design and design.designInfo else None
    # TODO: Enum should be used for allocation and interventionModel:
    attribs_features.append(
        Attribute(name="allocation", value=(d_info.allocation if d_info else None), declared_type=str))
    attribs_features.append(
        Attribute(name="interventionModel", value=(d_info.interventionModel if d_info else None), declared_type=str))
    attribs_features.append(
        Attribute(name="primaryPurpose", value=(d_info.primaryPurpose if d_info else None),
                  declared_type=PrimaryPurpose))

    masking_val = d_info.maskingInfo.masking if d_info and d_info.maskingInfo else None
    attribs_features.append(Attribute(name="masking", value=masking_val, declared_type=Masking))

    # --- Identification ---
    nctid = ps.identificationModule.nctId
    attribs_metadata.append(Attribute(name="nctId", value=nctid, declared_type=str))

    # --- Status ---
    sm = ps.statusModule
    attribs_metadata.append(Attribute(name="overallStatus", value=sm.overallStatus, declared_type=Status))

    completion_date = get_study_completion_date(sm)

    attribs_metadata.append(Attribute(name="completionDate",
                                      value=completion_date,
                                      declared_type=datetime))

    attribs_metadata.append(Attribute(name="startDate",
                                      value=(getattr(getattr(getattr(sm, "startDateStruct", None), "date", None), "dt",
                                                     None)),
                                      declared_type=datetime))

    sc = ps.sponsorCollaboratorsModule if ps and ps.sponsorCollaboratorsModule else None
    lead = sc.leadSponsor if sc and sc.leadSponsor else None
    lead_name = lead.name if lead else None
    attribs_metadata.append(Attribute(name="leadSponsorName", value=lead_name, declared_type=str))

    attribs_metadata.append(
        Attribute(name="isJJ", value=bool(lead_name and any(k in lead_name.lower() for k in JJ_KEYWORDS)),
                  declared_type=bool))

    # --- Eligibility ---
    elig = ps.eligibilityModule if ps and ps.eligibilityModule else None
    attribs_features.append(
        Attribute(name="healthyVolunteers", value=(elig.healthyVolunteers if elig else None), declared_type=bool))
    attribs_features.append(Attribute(name="sex", value=(elig.sex if elig else None), declared_type=Sex))

    std_ages = list(elig.stdAges) if (elig and elig.stdAges) else None
    if std_ages and isinstance(std_ages[0], Enum):  # cautious
        attribs_features.append(Attribute(name="stdAges", value=std_ages, declared_type=type(std_ages[0])))

    # --- Sponsor ---
    sc = ps.sponsorCollaboratorsModule if ps and ps.sponsorCollaboratorsModule else None
    lead = sc.leadSponsor if sc and sc.leadSponsor else None
    lead_name = lead.name if lead else None
    attribs_features.append(Attribute(name="leadSponsorName", value=lead_name, declared_type=str))
    attribs_features.append(
        Attribute(name="leadSponsorClass", value=(lead.class_ if lead else None), declared_type=AgencyClass))

    oversight = ps.oversightModule if ps and ps.oversightModule else None
    attribs_features.append(
        Attribute(name="oversightHasDmc", value=(oversight.oversightHasDmc if oversight else None), declared_type=bool))

    # --- Description Module ---
    attribs_features.append(Attribute(name="briefSummary",
                                      value=ps.descriptionModule.briefSummary,
                                      declared_type=str))

    attribs_features.append(Attribute(name="detailedDescription",
                                      value=ps.descriptionModule.detailedDescription,
                                      declared_type=str))

    # --- Conditions Module ---
    attribs_features.append(Attribute(name="conditions",
                                      value=" ".join(
                                          ps.conditionsModule.conditions) if ps and ps.conditionsModule else None,
                                      declared_type=str))

    attribs_features.append(Attribute(name="conditionsKeywords",
                                      value=" ".join(
                                          ps.conditionsModule.keywords) if ps and ps.conditionsModule else None,
                                      declared_type=str))

    # --- Documents and Protocol PDF's extraction ---
    if has_protocol(study):
        pdf_text = concatenate_pdf_texts_for_nctid(nctid, CTGOV_DATASET_EXTENSIONS_PATH)
    else:
        pdf_text = None
    attribs_features.append(Attribute(name="protocolPdfText", value=pdf_text, declared_type=str))

    attribs_metadata.append(Attribute(name="hasProtocol", value=has_protocol(study), declared_type=bool))
    attribs_metadata.append(Attribute(name="hasSap", value=has_sap(study), declared_type=bool))
    attribs_metadata.append(Attribute(name="hasIcf", value=has_icf(study), declared_type=bool))
    pdf_links = get_large_protocols_pdf_links(study, check_link_status=False)
    attribs_metadata.append(Attribute(name="protocolPdfLinks",
                                      value=" ".join(pdf_links) if pdf_links else None,
                                      declared_type=str))

    # --- Arms & interventions ---
    arms = get_protocol_arm_groups(study)

    num_arms = len(arms)
    attribs_features.append(Attribute(name="numArms", value=num_arms, declared_type=int))

    arm_descriptions = [getattr(arm, "description", None) for arm in arms]
    attribs_features.append(Attribute(name="armDescriptions",
                                      value=" ".join(
                                          f"arm {i + 1}: {s}" for i, s in
                                          enumerate(arm_descriptions)) if arm_descriptions else None,
                                      declared_type=str))
    arm_group_types = [getattr(arm, "type", None) for arm in arms]
    attribs_features.append(Attribute(name="armGroupTypes", value=(arm_group_types if arm_group_types else None),
                                      declared_type=ArmGroupType))

    interventions = get_protocol_interventions(study)
    attribs_features.append(Attribute(name="numInterventions", value=len(interventions), declared_type=int))

    itypes = [getattr(itv, "type", None) for itv in interventions]
    attribs_features.append(Attribute(name="interventionTypes", value=itypes, declared_type=InterventionType))

    i_descriptions = [getattr(itv, "description", None) for itv in interventions]
    attribs_features.append(Attribute(name="interventionDescriptions",
                                      value=" ".join(f"intervention {i + 1}: {s}" for i, s in
                                                     enumerate(i_descriptions)) if i_descriptions else None,
                                      declared_type=str))
    i_names = [getattr(itv, "name", None) for itv in interventions]
    attribs_features.append(Attribute(name="interventionNames",
                                      value=" ".join(
                                          f"intervention {i + 1}: {s}" for i, s in
                                          enumerate(i_names)) if i_names else None,
                                      declared_type=str))

    # --- Locations ---
    loc_details = get_location_details(study)
    attribs_features.append(Attribute(name="numLocations", value=len(loc_details), declared_type=int))
    attribs_features.append(
        Attribute(name="locationDetails", value="\n".join(loc_details) if loc_details else None, declared_type=str))

    # --- ADE enrichment ---
    ade = ade_analysis_results_for_study
    attribs_labels.append(Attribute(name="num_ct_level_ade_terms", value=len(ade.ade_clinical), declared_type=int))
    attribs_labels.append(
        Attribute(name="ct_level_ade_population", value=_total_ade_population(ade), declared_type=int))
    attribs_labels.append(
        Attribute(name="num_positive_terms_matched", value=len(ade.positive_terms), declared_type=int))

    # --- canonical label counts ---
    attribs_labels.extend(_get_ade_count_attributes_from_positive_terms(
        positive_terms=ade.positive_terms,
        canonical_label_cols=canonical_label_cols,
    ))
    # Creating new label-related fields based on existing ones
    attribs_labels = add_new_label_features_from_existing(attribs_labels, alpha_wilson, wilson_proba_threshold)

    # --- Renaming attributes with descriptive prefixes ---
    all_attribs = AttributesList(
        attribs_features.with_prefix(ATTRIBS_FEATURE_PREFIX)
        + attribs_labels.with_prefix(ATTRIBS_LABEL_PREFIX)
        + attribs_metadata.with_prefix(ATTRIBS_METADATA_PREFIX)
    )

    return all_attribs
