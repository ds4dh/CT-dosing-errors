"""
This peace of code was initially automatically generated using Gemini 2.5 Pro, refined using ChatGPT o3 and was modified
and extended with some human interventions.

The idea was to reflect the data structures defined in https://clinicaltrials.gov/data-api/about-api/study-data-structure
into python's stdlib Enum's and dataclass instances, so that typical type-related errors in parsing of the data from
CTGov would be avoided.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from enum import Enum
from typing import Any
from datetime import datetime

import re


# ==============================================================================
#  Enums
# ==============================================================================

class StrEnumWithNumericDeprecated(str, Enum):
    def __new__(cls, numeric: int, label: str):
        obj = str.__new__(cls, label)
        obj._value_ = label
        obj.numeric = numeric
        return obj

    def __int__(self) -> int:
        return self.numeric

    def __str__(self) -> str:
        return self.value


class StudyType(Enum):
    INTERVENTIONAL = "INTERVENTIONAL"
    OBSERVATIONAL = "OBSERVATIONAL"
    EXPANDED_ACCESS = "EXPANDED_ACCESS"


class PatientRegistry(Enum):
    YES = "YES"
    NO = "NO"

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in {"yes", "true"}:
                return cls.YES
            if value_lower in {"no", "false"}:
                return cls.NO
        if isinstance(value, bool):
            return cls.YES if value else cls.NO
        return None


class Phase(Enum):
    NA = "NA"
    EARLY_PHASE1 = "EARLY_PHASE1"
    PHASE1 = "PHASE1"
    PHASE2 = "PHASE2"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"


class PrimaryPurpose(Enum):
    TREATMENT = "TREATMENT"
    PREVENTION = "PREVENTION"
    DIAGNOSTIC = "DIAGNOSTIC"
    ECT = "ECT"
    SUPPORTIVE_CARE = "SUPPORTIVE_CARE"
    SCREENING = "SCREENING"
    HEALTH_SERVICES_RESEARCH = "HEALTH_SERVICES_RESEARCH"
    BASIC_SCIENCE = "BASIC_SCIENCE"
    DEVICE_FEASIBILITY = "DEVICE_FEASIBILITY"
    OTHER = "OTHER"


class Masking(Enum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    DOUBLE = "DOUBLE"
    TRIPLE = "TRIPLE"
    QUADRUPLE = "QUADRUPLE"


class WhoMasked(Enum):
    PARTICIPANT = "PARTICIPANT"
    CARE_PROVIDER = "CARE_PROVIDER"
    INVESTIGATOR = "INVESTIGATOR"
    OUTCOMES_ASSESSOR = "OUTCOMES_ASSESSOR"


class InterventionType(Enum):
    DRUG = "DRUG"
    DEVICE = "DEVICE"
    BIOLOGICAL = "BIOLOGICAL"
    PROCEDURE = "PROCEDURE"
    RADIATION = "RADIATION"
    BEHAVIORAL = "BEHAVIORAL"
    GENETIC = "GENETIC"
    DIETARY_SUPPLEMENT = "DIETARY_SUPPLEMENT"
    COMBINATION_PRODUCT = "COMBINATION_PRODUCT"
    DIAGNOSTIC_TEST = "DIAGNOSTIC_TEST"
    OTHER = "OTHER"


class ArmGroupType(Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    ACTIVE_COMPARATOR = "ACTIVE_COMPARATOR"
    PLACEBO_COMPARATOR = "PLACEBO_COMPARATOR"
    SHAM_COMPARATOR = "SHAM_COMPARATOR"
    NO_INTERVENTION = "NO_INTERVENTION"
    OTHER = "OTHER"


class MeasureParam(Enum):
    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"
    GEOMETRIC_LEAST_SQUARES_MEAN = "GEOMETRIC_LEAST_SQUARES_MEAN"
    LEAST_SQUARES_MEAN = "LEAST_SQUARES_MEAN"
    LOG_MEAN = "LOG_MEAN"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    NUMBER = "NUMBER"
    COUNT_OF_PARTICIPANTS = "COUNT_OF_PARTICIPANTS"
    COUNT_OF_UNITS = "COUNT_OF_UNITS"


class OutcomeMeasureType(Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    OTHER_PRE_SPECIFIED = "OTHER_PRE_SPECIFIED"
    POST_HOC = "POST_HOC"


class ReportingStatus(Enum):
    NOT_POSTED = "NOT_POSTED"
    POSTED = "POSTED"


class SamplingMethod(Enum):
    PROBABILITY_SAMPLE = "PROBABILITY_SAMPLE"
    NON_PROBABILITY_SAMPLE = "NON_PROBABILITY_SAMPLE"


class Sex(Enum):
    ALL = "ALL"
    FEMALE = "FEMALE"
    MALE = "MALE"


class GenderBased(Enum):
    YES = "YES"
    NO = "NO"

    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            value_lower = value.strip().lower()
            if value_lower in {"yes", "true"}:
                return cls.YES
            if value_lower in {"no", "false"}:
                return cls.NO
        if isinstance(value, bool):
            return cls.YES if value else cls.NO
        return None


class Role(Enum):
    SPONSOR = "SPONSOR"
    PRINCIPAL_INVESTIGATOR = "PRINCIPAL_INVESTIGATOR"
    SPONSOR_INVESTIGATOR = "SPONSOR_INVESTIGATOR"


class AgencyClass(Enum):
    NIH = "NIH"
    FED = "FED"
    OTHER_GOV = "OTHER_GOV"
    INDIV = "INDIV"
    INDUSTRY = "INDUSTRY"
    NETWORK = "NETWORK"
    AMBIG = "AMBIG"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class Status(Enum):
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    ENROLLING_BY_INVITATION = "ENROLLING_BY_INVITATION"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    RECRUITING = "RECRUITING"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    AVAILABLE = "AVAILABLE"
    NO_LONGER_AVAILABLE = "NO_LONGER_AVAILABLE"
    TEMPORARILY_NOT_AVAILABLE = "TEMPORARILY_NOT_AVAILABLE"
    APPROVED_FOR_MARKETING = "APPROVED_FOR_MARKETING"
    WITHHELD = "WITHHELD"
    UNKNOWN = "UNKNOWN"


class Certainty(Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    NO_ESTIMATE = "NO_ESTIMATE"


class Direction(Enum):
    UP = "UP"
    DOWN = "DOWN"
    UP_OR_DOWN = "UP_OR_DOWN"
    NO_CHANGE = "NO_CHANGE"


class GroupCode(Enum):
    EXP_ARM = "EXP_ARM"
    COMP_ARM = "COMP_ARM"
    TOTAL = "TOTAL"


class Type(Enum):
    SERIOUS = "SERIOUS"
    OTHER = "OTHER"


class StatisticalMethod(Enum):
    ANCOVA = "ANCOVA"
    ANOVA = "ANOVA"
    CHI_SQUARED = "Chi-Squared"
    CHI_SQUARED_CORRECTED = "Chi-Squared, Corrected"
    COCHRAN_MANTEL_HAENSZEL = "Cochran-Mantel-Haenszel"
    FISHER_EXACT = "Fisher Exact"
    KRUSKAL_WALLIS = "Kruskal-Wallis"
    LOG_RANK = "Log Rank"
    MANTEL_HAENSZEL = "Mantel Haenszel"
    MCNEMAR = "McNemar"
    MIXED_MODELS_ANALYSIS = "Mixed Models Analysis"
    REGRESSION_COX = "Regression, Cox"
    REGRESSION_LINEAR = "Regression, Linear"
    REGRESSION_LOGISTIC = "Regression, Logistic"
    SIGN_TEST = "Sign Test"
    T_TEST_1_SIDED = "t-Test, 1-Sided"
    T_TEST_2_SIDED = "t-Test, 2-Sided"
    WILCOXON_MANN_WHITNEY = "Wilcoxon (Mann-Whitney)"
    OTHER = "Other"


class ConfidenceIntervalNumSides(Enum):
    ONE_SIDED = "ONE_SIDED"
    TWO_SIDED = "TWO_SIDED"


class NonInferiorityType(Enum):
    SUPERIORITY_OR_OTHER = "SUPERIORITY_OR_OTHER"
    NON_INFERIORITY = "NON_INFERIORITY"
    EQUIVALENCE = "EQUIVALENCE"


class AnnotationType(Enum):
    COMMENT = "COMMENT"
    PRIMARY_CP = "PRIMARY_CP"
    SECONDARY_CP = "SECONDARY_CP"


class UnpostedEventType(Enum):
    RESET = "RESET"
    RELEASE = "RELEASE"
    UNRELEASE = "UNRELEASE"
    CANCELED = "CANCELED"


class UnpostedAnnotationSource(Enum):
    NLM = "NLM"
    SPONSOR = "SPONSOR"


class SubmissionStatus(Enum):
    PENDING = "PENDING"
    RELEASED = "RELEASED"
    RESET = "RESET"


# ==============================================================================
#  Pydantic Models
# ==============================================================================

class ExpandedAccessTypes(BaseModel):
    individual: bool | None = Field(default=None,
                                    description="For individual participants, including for emergency use.")
    intermediate: bool | None = Field(default=None, description="For intermediate-size participant populations.")
    treatment: bool | None = Field(default=None,
                                   description="Under a treatment IND or treatment protocol for a large, widespread population.")


class OrgStudyIdInfo(BaseModel):
    """A unique identifier for the study assigned by the sponsoring organization."""
    id: str | None = None
    type: str | None = None
    link: str | None = None


class SecondaryIdInfo(BaseModel):
    """Information about one or more secondary identifiers assigned to the study."""
    id: str | None = None
    type: str | None = None
    domain: str | None = None
    link: str | None = None


class IdentificationModule(BaseModel):
    """A module of a clinical study protocol that includes the primary and secondary identifiers of a study."""
    nctId: str | None = None
    nctIdAliases: list[str] = Field(default_factory=list)
    briefTitle: str | None = None
    officialTitle: str | None = None
    acronym: str | None = None
    orgStudyIdInfo: OrgStudyIdInfo | None = None
    secondaryIdInfos: list[SecondaryIdInfo] = Field(default_factory=list)
    organization: Organization | None = None


class StatusModule(BaseModel):
    """A module of a clinical study protocol that includes the status of the study and other administrative information."""
    status: Status | None = None
    statusVerifiedDate: Date | None = None
    overallStatus: Status | None = None
    lastKnownStatus: Status | None = None
    delayedPosting: bool | None = None
    whyStopped: str | None = None
    expandedAccessInfo: ExpandedAccessInfo | None = None
    startDateStruct: DateStruct | None = None
    primaryCompletionDateStruct: DateStruct | None = None
    completionDateStruct: DateStruct | None = None
    studyFirstSubmitDate: Date | None = None
    studyFirstSubmitQcDate: Date | None = None
    studyFirstPostDateStruct: DateStruct | None = None
    lastUpdateSubmitDate: datetime | None = None
    lastUpdatePostDateStruct: DateStruct | None = None


class SponsorCollaboratorsModule(BaseModel):
    """A module of a clinical study protocol that includes information about the organizations and individuals responsible for the study."""
    responsibleParty: ResponsibleParty | None = None
    leadSponsor: LeadSponsor | None = None
    collaborators: list[Collaborator] = Field(default_factory=list)


class OversightModule(BaseModel):
    """A module of a clinical study protocol that includes information about the oversight of the study."""
    oversightHasDmc: bool | None = None
    isFdaRegulatedDrug: bool | None = None
    isFdaRegulatedDevice: bool | None = None
    isUnapprovedDevice: bool | None = None
    isPpsd: bool | None = None
    isUsExport: bool | None = None
    fdaaa801Violation: bool | None = None


class DescriptionModule(BaseModel):
    """A module of a clinical study protocol that includes the brief and detailed descriptions of the study."""
    briefSummary: str | None = None
    detailedDescription: str | None = None


class ConditionsModule(BaseModel):
    """A module of a clinical study protocol that includes the conditions or diseases being studied."""
    conditions: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class DesignModule(BaseModel):
    """A module of a clinical study protocol that includes information about the design of the study."""
    studyType: StudyType | None = None
    nPtrsToThisExpAccNctId: int | None = None
    expandedAccessTypes: ExpandedAccessTypes | None = None
    patientRegistry: PatientRegistry | None = None
    targetDuration: str | None = None
    phases: list[Phase] = Field(default_factory=list)
    designInfo: DesignInfo | None = None
    bioSpec: BioSpec | None = None
    enrollmentInfo: EnrollmentInfo | None = None


class ArmsInterventionsModule(BaseModel):
    """A module of a clinical study protocol that includes information about the arms and interventions of the study."""
    armGroups: list[ArmGroup] = Field(default_factory=list)
    interventions: list[Intervention] = Field(default_factory=list)


class OutcomesModule(BaseModel):
    """A module of a clinical study protocol that includes information about the outcome measures of the study."""
    primaryOutcomes: list[PrimaryOutcome] = Field(default_factory=list)
    secondaryOutcomes: list[SecondaryOutcome] = Field(default_factory=list)
    otherOutcomes: list[OtherOutcome] = Field(default_factory=list)


class EligibilityModule(BaseModel):
    """A module of a clinical study protocol that includes information about the eligibility criteria for the study."""
    eligibilityCriteria: str | None = None
    healthyVolunteers: bool | None = None
    sex: Sex | None = None
    genderBased: GenderBased | None = None
    genderDescription: str | None = None
    minimumAge: str | None = None
    maximumAge: str | None = None
    stdAges: list[str] = Field(default_factory=list)
    studyPopulation: str | None = None
    samplingMethod: SamplingMethod | None = None


class ContactsLocationsModule(BaseModel):
    """A module of a clinical study protocol that includes information about the contacts and locations for the study."""
    centralContacts: list[CentralContact] = Field(default_factory=list)
    overallOfficials: list[OverallOfficial] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)


class ReferencesModule(BaseModel):
    """A module of a clinical study protocol that includes references to publications and other resources related to the study."""
    references: list[Reference] = Field(default_factory=list)
    seeAlsoLinks: list[SeeAlsoLink] = Field(default_factory=list)
    availIpds: list[AvailIpd] = Field(default_factory=list)


class IpdsSharingStatementModule(BaseModel):
    """A module of a clinical study protocol that includes information about the plan to share individual participant data."""
    ipdSharing: str | None = None
    description: str | None = None
    infoTypes: list[str] = Field(default_factory=list)
    timeFrame: str | None = None
    accessCriteria: str | None = None
    url: str | None = None


class ProtocolSection(BaseModel):
    """The section of the protocol that contains the descriptive information about a clinical study."""
    identificationModule: IdentificationModule | None = None
    statusModule: StatusModule | None = None
    sponsorCollaboratorsModule: SponsorCollaboratorsModule | None = None
    oversightModule: OversightModule | None = None
    descriptionModule: DescriptionModule | None = None
    conditionsModule: ConditionsModule | None = None
    designModule: DesignModule | None = None
    armsInterventionsModule: ArmsInterventionsModule | None = None
    outcomesModule: OutcomesModule | None = None
    eligibilityModule: EligibilityModule | None = None
    contactsLocationsModule: ContactsLocationsModule | None = None
    referencesModule: ReferencesModule | None = None
    ipdsSharingStatementModule: IpdsSharingStatementModule | None = None


class ParticipantFlowModule(BaseModel):
    """A module of a clinical study results that includes information about the flow of participants through the study."""
    preAssignmentDetails: str | None = None
    recruitmentDetails: str | None = None
    typeUnitsAnalyzed: str | None = None
    groups: list[Group] = Field(default_factory=list)
    periods: list[Period] = Field(default_factory=list)


class BaselineCharacteristicsModule(BaseModel):
    """A module of a clinical study results that includes information about the baseline characteristics of the study participants."""
    populationDescription: str | None = None
    typeUnitsAnalyzed: str | None = None
    groups: list[Group] = Field(default_factory=list)
    measures: list[Measure] = Field(default_factory=list)
    denoms: list[Denom] = Field(default_factory=list)


class OutcomeMeasuresModule(BaseModel):
    """A module of a clinical study results that includes information about the outcome measures of the study."""
    outcomeMeasures: list[Measure] = Field(default_factory=list)


class AdverseEventsModule(BaseModel):
    """A module of a clinical study results that includes information about the adverse events that occurred during the study."""
    frequencyThreshold: int | float | None = None
    timeFrame: str | None = None
    description: str | None = None
    allCauseMortalityRate: int | float | None = None
    eventGroups: list[EventGroup] = Field(default_factory=list)
    seriousEvents: list[Event] = Field(default_factory=list)
    otherEvents: list[Event] = Field(default_factory=list)

    @field_validator('frequencyThreshold', mode='before')
    def clean_frequency_threshold(cls, v: Any) -> float | None:
        """
        Custom validator to handle non-numeric strings and comma-formatted numbers
        before standard validation runs.
        """
        if v is None or isinstance(v, (int, float)):
            return v

        if isinstance(v, str):
            cleaned_str = v.strip().lower()

            if cleaned_str == 'any':
                return None

            if ',' in cleaned_str:
                cleaned_str = cleaned_str.replace(',', '.')

            try:
                return float(cleaned_str)
            except ValueError:
                raise ValueError(f"Could not parse '{v}' as a valid number.")

        raise ValueError(f"Unexpected type for frequencyThreshold: {type(v)}")


class MoreInfoModule(BaseModel):
    """A module of a clinical study results that includes additional information about the study."""
    limitationsAndCaveats: LimitationsAndCaveats | None = None
    certaintyRatings: list[CertaintyRating] = Field(default_factory=list)
    pointOfContact: PointOfContact | None = None


class ResultsSection(BaseModel):
    """The section of the results that contains the summary of the outcomes of a clinical study."""
    participantFlowModule: ParticipantFlowModule | None = None
    baselineCharacteristicsModule: BaselineCharacteristicsModule | None = None
    outcomeMeasuresModule: OutcomeMeasuresModule | None = None
    adverseEventsModule: AdverseEventsModule | None = None
    moreInfoModule: MoreInfoModule | None = None


class UnpostedAnnotation(BaseModel):
    """An annotation that is not posted on the study record."""
    source: UnpostedAnnotationSource | None = None
    unpostedResponsibleParty: str | None = None
    unpostedDate: Date | None = None


class Annotation(BaseModel):
    """An annotation to the study record."""
    type: AnnotationType | None = None
    unpostedAnnotation: UnpostedAnnotation | None = None
    pi: str | None = None
    label: str | None = None
    description: str | None = None


class AnnotationSection(BaseModel):
    """The section of the study record that contains annotations."""
    annotations: list[Annotation] = Field(default_factory=list)


class LargeDocumentModule(BaseModel):
    """A module that contains information about large documents."""
    largeDocs: list[LargeDoc] = Field(default_factory=list)


class DocumentSection(BaseModel):
    """The section of the study record that contains documents."""
    largeDocumentModule: LargeDocumentModule | None = None


class Submission(BaseModel):
    """A submission of a study record to PRS."""
    date: Date | None = None
    status: SubmissionStatus | None = None
    releaseDate: Date | None = None
    unreleaseDate: Date | None = None
    returnDate: Date | None = None
    resetDate: Date | None = None
    cancelDate: Date | None = None
    disposition: str | None = None
    postDate: Date | None = None


class History(BaseModel):
    """The history of changes to the study record."""
    version: str | None = None
    date: Date | None = None
    submissions: list[Submission] = Field(default_factory=list)


class MiscInfoModule(BaseModel):
    """A module that contains miscellaneous information about the study record."""
    versionHolder: str | None = None
    removedCountries: list[str] = Field(default_factory=list)
    history: list[History] = Field(default_factory=list)


class DerivedSection(BaseModel):
    """The section of the study record that contains derived information."""
    miscInfoModule: MiscInfoModule | None = None
    conditionBrowseModule: ConditionBrowseModule | None = None
    interventionBrowseModule: InterventionBrowseModule | None = None


class UnpostedEvent(BaseModel):
    """An event that is not posted on the study record."""
    type: UnpostedEventType | None = None
    date: Date | None = None
    dateUnknown: bool | None = None


class UnpostedModule(BaseModel):
    """A module that contains unposted information."""
    unpostedEvents: list[UnpostedEvent] = Field(default_factory=list)


class PmcLink(BaseModel):
    """A link to a PubMed Central article."""
    id: str | None = None
    url: str | None = None


class ProvidedDocument(BaseModel):
    """A document provided by the sponsor."""
    type: str | None = None
    url: str | None = None
    hasProtocol: bool | None = None
    hasSap: bool | None = None
    hasIcf: bool | None = None
    date: Date | None = None


class LargeDoc(BaseModel):
    """A large document."""
    typeAbbrev: str | None = None
    hasProtocol: bool | None = None
    hasSap: bool | None = None
    hasIcf: bool | None = None
    label: str | None = None
    date: Date | None = None
    uploadDate: Date | None = None
    filename: str | None = None
    size: int | None = None


class BrowseLeaf(BaseModel):
    """A leaf in the browse tree."""
    id: str | None = None
    name: str | None = None
    asFound: str | None = None
    relevance: str | None = None


class BrowseBranch(BaseModel):
    """A branch in the browse tree."""
    abbrev: str | None = None
    name: str | None = None
    leaves: list[BrowseLeaf] = Field(default_factory=list)


class ConditionBrowseModule(BaseModel):
    """A module that contains information about the conditions being studied."""
    meshes: list[BrowseMesh] = Field(default_factory=list)
    ancestors: list[BrowseMesh] = Field(default_factory=list)
    browseLeaves: list[BrowseLeaf] = Field(default_factory=list)
    browseBranches: list[BrowseBranch] = Field(default_factory=list)


class InterventionBrowseModule(BaseModel):
    """A module that contains information about the interventions being studied."""
    meshes: list[BrowseMesh] = Field(default_factory=list)
    ancestors: list[BrowseMesh] = Field(default_factory=list)
    browseLeaves: list[BrowseLeaf] = Field(default_factory=list)
    browseBranches: list[BrowseBranch] = Field(default_factory=list)


class BrowseMesh(BaseModel):
    """A mesh term."""
    id: str | None = None
    term: str | None = None


class Organization(BaseModel):
    """An organization."""
    fullName: str | None = None
    class_: AgencyClass | None = Field(default=None, alias='class')


class Date(BaseModel):
    """
    Parses CTGOV date strings into datetime.

    Accepts:
      - raw string: "2012-03", "March 2012", "Mar 12, 2012", "2012-03-15"
      - dicts (for robustness): {"date": "..."} or {"dt": "..."}
      - None
    Stores the result in `dt` (datetime | None).
    """
    model_config = ConfigDict(extra="ignore")

    dt: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_input(cls, v: Any) -> dict[str, Any] | Any:
        # Allow passing the raw string OR {"date": "..."} OR {"dt": "..."}
        if isinstance(v, dict):
            if "dt" in v:
                return v
            if "date" in v:
                return {"dt": v["date"]}
        # raw scalar (e.g., string) -> funnel into "dt"
        return {"dt": v}

    @field_validator("dt", mode="before")
    @classmethod
    def _parse_ctgov_date(cls, v: Any) -> datetime | None:
        if v is None:
            return None
        if isinstance(v, datetime):
            return v

        s = str(v).strip()
        if s.lower() in {"n/a", "na", "unknown", ""}:
            return None

        # ISO 8601 (YYYY-MM-DD or full datetime)
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass

        # Year–month -> first day of the month
        try:
            dt = datetime.strptime(s, "%Y-%m")
            return datetime(dt.year, dt.month, 1)
        except ValueError:
            pass

        # Month Day, Year
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass

        # Month Year -> first day of the month
        for fmt in ("%B %Y", "%b %Y"):
            try:
                dt = datetime.strptime(s, fmt)
                return datetime(dt.year, dt.month, 1)
            except ValueError:
                pass

        # Year only -> Jan 1st
        try:
            dt = datetime.strptime(s, "%Y")
            return datetime(dt.year, 1, 1)
        except ValueError:
            pass

        raise ValueError(f"Unrecognized date format: {s!r}")


class DateStruct(BaseModel):
    """
    CTGOV date wrapper: keeps the parsed date (as Date) and the CTGOV 'type' string.
    JSON I/O uses the key 'type', but the Python attribute is `type_`.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    date: Date | None = None
    type_: str | None = Field(default=None, alias="type")


class ExpandedAccessInfo(BaseModel):
    """Information about expanded access."""
    hasExpandedAccess: bool | None = None
    nctId: str | None = None
    statusForNctId: Status | None = None


class ResponsibleParty(BaseModel):
    """The party responsible for the study."""
    type: Role | None = None
    investigatorFullName: str | None = None
    investigatorTitle: str | None = None
    investigatorAffiliation: str | None = None
    oldName: str | None = None
    oldOrganization: str | None = None


class LeadSponsor(BaseModel):
    """The lead sponsor of the study."""
    name: str | None = None
    class_: AgencyClass | None = Field(default=None, alias='class')


class Collaborator(BaseModel):
    """A collaborator in the study."""
    name: str | None = None
    class_: AgencyClass | None = Field(default=None, alias='class')


class DesignInfo(BaseModel):
    """Information about the design of the study."""
    allocation: str | None = None
    interventionModel: str | None = None
    interventionModelDescription: str | None = None
    primaryPurpose: PrimaryPurpose | None = None
    observationalModel: str | None = None
    timePerspective: str | None = None
    maskingInfo: MaskingInfo | None = None


class MaskingInfo(BaseModel):
    """Information about the masking of the study."""
    masking: Masking | None = None
    whoMasked: list[WhoMasked] = Field(default_factory=list)
    description: str | None = None


class BioSpec(BaseModel):
    """Information about the biospecimens retained for the study."""
    retention: str | None = None
    description: str | None = None


class EnrollmentInfo(BaseModel):
    """Information about the enrollment of the study."""
    count: int | None = None
    type: str | None = None


class ArmGroup(BaseModel):
    """An arm group in the study."""
    label: str | None = None
    type: ArmGroupType | None = None
    description: str | None = None
    interventionNames: list[str] = Field(default_factory=list)


class Intervention(BaseModel):
    """An intervention in the study."""
    type: InterventionType | None = None
    name: str | None = None
    description: str | None = None
    otherNames: list[str] = Field(default_factory=list)
    armGroupLabels: list[str] = Field(default_factory=list)


class PrimaryOutcome(BaseModel):
    """A primary outcome measure of the study."""
    measure: str | None = None
    description: str | None = None
    timeFrame: str | None = None


class SecondaryOutcome(BaseModel):
    """A secondary outcome measure of the study."""
    measure: str | None = None
    description: str | None = None
    timeFrame: str | None = None


class OtherOutcome(BaseModel):
    """An other outcome measure of the study."""
    measure: str | None = None
    description: str | None = None
    timeFrame: str | None = None


class CentralContact(BaseModel):
    """A central contact for the study."""
    name: str | None = None
    role: str | None = None
    phone: str | None = None
    phoneExt: str | None = None
    email: str | None = None


class OverallOfficial(BaseModel):
    """An overall official for the study."""
    name: str | None = None
    affiliation: str | None = None
    role: str | None = None


class Location(BaseModel):
    """A location where the study is being conducted."""
    facility: str | None = None
    status: Status | None = None
    city: str | None = None
    state: str | None = None
    zip: str | None = None
    country: str | None = None
    contacts: list[Contact] = Field(default_factory=list)
    geoPoint: GeoPoint | None = None


class Contact(BaseModel):
    """A contact for a location."""
    name: str | None = None
    role: str | None = None
    phone: str | None = None
    phoneExt: str | None = None
    email: str | None = None


class GeoPoint(BaseModel):
    """A geographic point."""
    lat: int | float | None = None
    lon: int | float | None = None


class Reference(BaseModel):
    """A reference to a publication or other resource related to the study."""
    pmid: str | None = None
    type: str | None = None
    citation: str | None = None
    retraction: Retraction | None = None


class Retraction(BaseModel):
    """A retraction of a publication."""
    pmid: str | None = None
    source: str | None = None


class SeeAlsoLink(BaseModel):
    """A link to a related resource."""
    label: str | None = None
    url: str | None = None


class AvailIpd(BaseModel):
    """Information about available individual participant data."""
    id: str | None = None
    type: str | None = None
    url: str | None = None
    comment: str | None = None


class Group(BaseModel):
    """A group in the study."""
    id: str | None = None
    title: str | None = None
    description: str | None = None


class Period(BaseModel):
    """A period in the study."""
    title: str | None = None
    milestones: list[Milestone] = Field(default_factory=list)
    dropWithdraws: list[DropWithdraw] = Field(default_factory=list)


class Milestone(BaseModel):
    """A milestone in the study."""
    type: str | None = None
    title: str | None = None
    counts: list[Count] = Field(default_factory=list)


class Count(BaseModel):
    """A count of participants in a group."""
    groupId: str | None = None
    value: str | None = None


class DropWithdraw(BaseModel):
    """A drop or withdrawal from the study."""
    type: str | None = None
    reason: str | None = None
    counts: list[Count] = Field(default_factory=list)


class Measure(BaseModel):
    """A measure in the study."""
    title: str | None = None
    type: OutcomeMeasureType | None = None
    description: str | None = None
    populationDescription: str | None = None
    reportingStatus: ReportingStatus | None = None
    paramType: MeasureParam | None = None
    unitOfMeasure: str | None = None
    timeFrame: str | None = None
    param: str | None = None
    dispersionType: str | None = None
    groups: list[Group] = Field(default_factory=list)
    denoms: list[Denom] = Field(default_factory=list)
    classes: list[Class] = Field(default_factory=list)
    analyses: list[OutcomeAnalysis] = Field(default_factory=list)


class OutcomeAnalysis(BaseModel):
    """
    Outcome analysis for an outcome measure.
    Mirrors fields observed under outcomeMeasures[*].analyses[*].
    """
    model_config = ConfigDict(extra="ignore")

    # Group comparison / scope
    groupIds: list[str] = Field(default_factory=list)
    groupDescription: str | None = None

    # Non-inferiority / superiority metadata
    testedNonInferiority: bool | None = None
    nonInferiorityType: NonInferiorityType | None = None

    # P-value and comments (often formatted as strings: "<0.05", ".04")
    pValue: str | None = None
    pValueComment: str | None = None

    # Statistical method and comments (highly heterogeneous strings)
    statisticalMethod: StatisticalMethod | None = None
    otherMethodName: str | None = None
    statisticalComment: str | None = None

    # Confidence interval metadata
    ciPctValue: str | None = None  # often "95"
    ciNumSides: ConfidenceIntervalNumSides | None = None
    ciLowerLimit: str | None = None
    ciUpperLimit: str | None = None

    # Analysis-level effect size fields (seen in Cox regression examples)
    paramType: str | None = None  # e.g., "Hazard Ratio (HR)"
    paramValue: str | None = None  # e.g., "0.91"

    @field_validator("pValue", mode="before")
    @classmethod
    def _normalize_pvalue(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @field_validator("statisticalMethod", mode="before")
    @classmethod
    def normalize_statistical_method(cls, v: Any) -> Any:
        """
        Map common CTGov string variants to the canonical StatisticalMethod enum values.
        Keeps strict typing while handling messy capitalization/punctuation.
        """
        if v is None or isinstance(v, StatisticalMethod):
            return v

        s = str(v).strip()
        if not s:
            return None

        # Normalize whitespace and punctuation
        s_norm = re.sub(r"\s+", " ", s).strip()
        s_key = s_norm.lower()

        # Common variants observed in CTGov
        alias_map = {
            "t-test, 2 sided": StatisticalMethod.T_TEST_2_SIDED,
            "t test, 2 sided": StatisticalMethod.T_TEST_2_SIDED,
            "t-test, 2-sided": StatisticalMethod.T_TEST_2_SIDED,
            "t test, 2-sided": StatisticalMethod.T_TEST_2_SIDED,
            "t-test, two sided": StatisticalMethod.T_TEST_2_SIDED,
            "t-test, 1 sided": StatisticalMethod.T_TEST_1_SIDED,
            "t test, 1 sided": StatisticalMethod.T_TEST_1_SIDED,
            "t-test, 1-sided": StatisticalMethod.T_TEST_1_SIDED,
            "t test, 1-sided": StatisticalMethod.T_TEST_1_SIDED,

            "wilcoxon (mann-whitney)": StatisticalMethod.WILCOXON_MANN_WHITNEY,
            "wilcoxon (mann whitney)": StatisticalMethod.WILCOXON_MANN_WHITNEY,
            "wilcoxon mann-whitney": StatisticalMethod.WILCOXON_MANN_WHITNEY,
            "wilcoxon mann whitney": StatisticalMethod.WILCOXON_MANN_WHITNEY,

            "chi-squared": StatisticalMethod.CHI_SQUARED,
            "chi squared": StatisticalMethod.CHI_SQUARED,
            "chi-squared, corrected": StatisticalMethod.CHI_SQUARED_CORRECTED,
            "chi squared, corrected": StatisticalMethod.CHI_SQUARED_CORRECTED,

            "mantel haenszel": StatisticalMethod.MANTEL_HAENSZEL,
            "cochran mantel haenszel": StatisticalMethod.COCHRAN_MANTEL_HAENSZEL,

            "regression, cox": StatisticalMethod.REGRESSION_COX,
            "regression, linear": StatisticalMethod.REGRESSION_LINEAR,
            "regression, logistic": StatisticalMethod.REGRESSION_LOGISTIC,

            "mixed models analysis": StatisticalMethod.MIXED_MODELS_ANALYSIS,
            "log rank": StatisticalMethod.LOG_RANK,
            "fisher exact": StatisticalMethod.FISHER_EXACT,
            "kruskal-wallis": StatisticalMethod.KRUSKAL_WALLIS,
            "kruskal wallis": StatisticalMethod.KRUSKAL_WALLIS,
            "ancova": StatisticalMethod.ANCOVA,
            "anova": StatisticalMethod.ANOVA,
            "mcnemar": StatisticalMethod.MCNEMAR,
            "sign test": StatisticalMethod.SIGN_TEST,
            "other": StatisticalMethod.OTHER,
        }

        if s_key in alias_map:
            return alias_map[s_key]

        # If it already matches canonical Enum values (exact), allow it
        for m in StatisticalMethod:
            if s_norm == m.value:
                return m

        # If you want strict behavior, raise here.
        # But for robustness, map unknown to OTHER and preserve via otherMethodName.
        # Returning OTHER prevents validation failure:
        return StatisticalMethod.OTHER

    @field_validator("nonInferiorityType", mode="before")
    @classmethod
    def normalize_non_inferiority_type(cls, v: Any) -> Any:
        if v is None or isinstance(v, NonInferiorityType):
            return v

        s = str(v).strip()
        if not s:
            return None

        # Legacy mapping
        if s == "SUPERIORITY_OR_OTHER_LEGACY":
            return NonInferiorityType.SUPERIORITY_OR_OTHER

        # allow canonical values
        for t in NonInferiorityType:
            if s == t.value:
                return t

        # If you prefer strict behavior, raise here.
        # For robustness, treat unknowns as None.
        return None


class Class(BaseModel):
    """A class in a measure."""
    title: str | None = None
    denoms: list[Denom] = Field(default_factory=list)
    categories: list[Category] = Field(default_factory=list)


class Category(BaseModel):
    """A category in a class."""
    title: str | None = None
    measurements: list[Measurement] = Field(default_factory=list)


class Measurement(BaseModel):
    """A measurement in a category."""
    groupId: str | None = None
    value: str | None = None
    spread: str | None = None
    lowerLimit: str | None = None
    upperLimit: str | None = None


class Denom(BaseModel):
    """A denominator in a measure."""
    units: str | None = None
    counts: list[Count] = Field(default_factory=list)


class EventGroup(BaseModel):
    """A group of events."""
    id: str | None = None
    title: str | None = None
    description: str | None = None
    deathsNumAffected: int | None = None
    deathsNumAtRisk: int | None = None
    seriousNumAffected: int | None = None
    seriousNumAtRisk: int | None = None
    otherNumAffected: int | None = None
    otherNumAtRisk: int | None = None


class Event(BaseModel):
    """An event in the study."""
    term: str | None = None
    organSystem: str | None = None
    sourceVocabulary: str | None = None
    assessment: str | None = None
    notes: str | None = None
    stats: list[EventStats] = Field(default_factory=list)


class EventStats(BaseModel):
    """Statistics for an event."""
    groupId: str | None = None
    numEvents: int | None = None
    numAffected: int | None = None
    numAtRisk: int | None = None


class LimitationsAndCaveats(BaseModel):
    """Limitations and caveats of the study."""
    description: str | None = None


class CertaintyRating(BaseModel):
    """A rating of the certainty of the evidence."""
    rating: Certainty | None = None
    assessor: str | None = None
    description: str | None = None
    direction: Direction | None = None
    outcome: str | None = None
    groupCode: GroupCode | None = None
    intervention: str | None = None
    comparison: str | None = None


class PointOfContact(BaseModel):
    """A point of contact for the study."""
    title: str | None = None
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    phoneExt: str | None = None


class Study(BaseModel):
    """The Study structure is the top-level object in the study JSON response."""
    protocolSection: ProtocolSection | None = None
    resultsSection: ResultsSection | None = None
    annotationSection: AnnotationSection | None = None
    documentSection: DocumentSection | None = None
    derivedSection: DerivedSection | None = None
    hasResults: bool | None = None


class FullStudy(BaseModel):
    """The root object of a full study response from the API."""
    study: Study | None = None
