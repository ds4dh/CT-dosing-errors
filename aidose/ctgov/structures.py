"""
This peace of code was initially automatically generated using Gemini 2.5 Pro, refined using ChatGPT o3 and was modified
and extended with some human interventions.

The idea was to reflect the data structures defined in https://clinicaltrials.gov/data-api/about-api/study-data-structure
into python's stdlib Enum's and dataclass instances, so that typical type-related errors in parsing of the data from
CTGov would be avoided.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Any


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


class MeasureType(Enum):
    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    OTHER_PRE_SPECIFIED = "OTHER_PRE_SPECIFIED"
    POST_HOC = "POST_HOC"


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


class PValue(Enum):
    STUDENT_T_TEST_2_SIDED = "STUDENT_T_TEST_2_SIDED"
    ANOVA = "ANOVA"
    CHI_SQUARED = "CHI_SQUARED"
    CHI_SQUARED_CORRECTION = "CHI_SQUARED_CORRECTION"
    COCHRAN_MANTEL_HAENSZEL = "COCHRAN_MANTEL_HAENSZEL"
    FISHER_EXACT = "FISHER_EXACT"
    KRUSKAL_WALLIS = "KRUSKAL_WALLIS"
    LOG_RANK = "LOG_RANK"
    MCNEMAR = "MCNEMAR"
    WILCOXON_MANN_WHITNEY = "WILCOXON_MANN_WHITNEY"
    REGRESSION_LINEAR = "REGRESSION_LINEAR"
    REGRESSION_LOGISTIC = "REGRESSION_LOGISTIC"
    REGRESSION_COX = "REGRESSION_COX"


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
    statusVerifiedDate: str | None = None
    overallStatus: Status | None = None
    lastKnownStatus: Status | None = None
    delayedPosting: bool | None = None
    whyStopped: str | None = None
    expandedAccessInfo: ExpandedAccessInfo | None = None
    startDateStruct: DateStruct | None = None
    primaryCompletionDateStruct: DateStruct | None = None
    completionDateStruct: DateStruct | None = None
    studyFirstSubmitDate: str | None = None
    studyFirstSubmitQcDate: str | None = None
    studyFirstPostDateStruct: DateStruct | None = None
    lastUpdateSubmitDate: str | None = None
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
    measures: list[Measure] = Field(default_factory=list)


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
    unpostedDate: str | None = None


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
    date: str | None = None
    status: SubmissionStatus | None = None
    releaseDate: str | None = None
    unreleaseDate: str | None = None
    returnDate: str | None = None
    resetDate: str | None = None
    cancelDate: str | None = None
    disposition: str | None = None
    postDate: str | None = None


class History(BaseModel):
    """The history of changes to the study record."""
    version: str | None = None
    date: str | None = None
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
    date: str | None = None
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
    date: str | None = None


class LargeDoc(BaseModel):
    """A large document."""
    typeAbbrev: str | None = None
    hasProtocol: bool | None = None
    hasSap: bool | None = None
    hasIcf: bool | None = None
    label: str | None = None
    date: str | None = None
    uploadDate: str | None = None
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


class DateStruct(BaseModel):
    """A date structure."""
    date: str | None = None
    type: str | None = None


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
    description: str | None = None
    populationDescription: str | None = None
    unit: str | None = None
    param: str | None = None
    dispersion: str | None = None
    denoms: list[Denom] = Field(default_factory=list)
    classes: list[Class] = Field(default_factory=list)


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
