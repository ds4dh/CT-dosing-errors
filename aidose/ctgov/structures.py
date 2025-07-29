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

class StrEnumWithNumeric(str, Enum):
    def __new__(cls, numeric: int, label: str):
        obj = str.__new__(cls, label)
        obj._value_ = label
        obj.numeric = numeric
        return obj

    def __int__(self) -> int:
        return self.numeric

    def __str__(self) -> str:
        return self.value


class StudyType(StrEnumWithNumeric):
    INTERVENTIONAL = (0, "INTERVENTIONAL")
    OBSERVATIONAL = (1, "OBSERVATIONAL")
    EXPANDED_ACCESS = (2, "EXPANDED_ACCESS")


class PatientRegistry(StrEnumWithNumeric):
    YES = (1, "YES")
    NO = (0, "NO")

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


class Phase(StrEnumWithNumeric):
    NA = (0, "NA")
    EARLY_PHASE1 = (1, "EARLY_PHASE1")
    PHASE1 = (2, "PHASE1")
    PHASE2 = (3, "PHASE2")
    PHASE3 = (4, "PHASE3")
    PHASE4 = (5, "PHASE4")


class PrimaryPurpose(StrEnumWithNumeric):
    TREATMENT = (0, "TREATMENT")
    PREVENTION = (1, "PREVENTION")
    DIAGNOSTIC = (2, "DIAGNOSTIC")
    ECT = (3, "ECT")
    SUPPORTIVE_CARE = (4, "SUPPORTIVE_CARE")
    SCREENING = (5, "SCREENING")
    HEALTH_SERVICES_RESEARCH = (6, "HEALTH_SERVICES_RESEARCH")
    BASIC_SCIENCE = (7, "BASIC_SCIENCE")
    DEVICE_FEASIBILITY = (8, "DEVICE_FEASIBILITY")
    OTHER = (9, "OTHER")


class Masking(StrEnumWithNumeric):
    NONE = (0, "NONE")
    SINGLE = (1, "SINGLE")
    DOUBLE = (2, "DOUBLE")
    TRIPLE = (3, "TRIPLE")
    QUADRUPLE = (4, "QUADRUPLE")


class WhoMasked(StrEnumWithNumeric):
    PARTICIPANT = (0, "PARTICIPANT")
    CARE_PROVIDER = (1, "CARE_PROVIDER")
    INVESTIGATOR = (2, "INVESTIGATOR")
    OUTCOMES_ASSESSOR = (3, "OUTCOMES_ASSESSOR")


class InterventionType(StrEnumWithNumeric):
    DRUG = (0, "DRUG")
    DEVICE = (1, "DEVICE")
    BIOLOGICAL = (2, "BIOLOGICAL")
    PROCEDURE = (3, "PROCEDURE")
    RADIATION = (4, "RADIATION")
    BEHAVIORAL = (5, "BEHAVIORAL")
    GENETIC = (6, "GENETIC")
    DIETARY_SUPPLEMENT = (7, "DIETARY_SUPPLEMENT")
    COMBINATION_PRODUCT = (8, "COMBINATION_PRODUCT")
    DIAGNOSTIC_TEST = (9, "DIAGNOSTIC_TEST")
    OTHER = (10, "OTHER")


class ArmGroupType(StrEnumWithNumeric):
    EXPERIMENTAL = (0, "EXPERIMENTAL")
    ACTIVE_COMPARATOR = (1, "ACTIVE_COMPARATOR")
    PLACEBO_COMPARATOR = (2, "PLACEBO_COMPARATOR")
    SHAM_COMPARATOR = (3, "SHAM_COMPARATOR")
    NO_INTERVENTION = (4, "NO_INTERVENTION")
    OTHER = (5, "OTHER")


class MeasureType(StrEnumWithNumeric):
    PRIMARY = (0, "PRIMARY")
    SECONDARY = (1, "SECONDARY")
    OTHER_PRE_SPECIFIED = (2, "OTHER_PRE_SPECIFIED")
    POST_HOC = (3, "POST_HOC")


class SamplingMethod(StrEnumWithNumeric):
    PROBABILITY_SAMPLE = (0, "PROBABILITY_SAMPLE")
    NON_PROBABILITY_SAMPLE = (1, "NON_PROBABILITY_SAMPLE")


class Sex(StrEnumWithNumeric):
    ALL = (0, "ALL")
    FEMALE = (1, "FEMALE")
    MALE = (2, "MALE")


class GenderBased(StrEnumWithNumeric):
    YES = (1, "YES")
    NO = (0, "NO")

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


class Role(StrEnumWithNumeric):
    SPONSOR = (0, "SPONSOR")
    PRINCIPAL_INVESTIGATOR = (1, "PRINCIPAL_INVESTIGATOR")
    SPONSOR_INVESTIGATOR = (2, "SPONSOR_INVESTIGATOR")


class AgencyClass(StrEnumWithNumeric):
    NIH = (0, "NIH")
    FED = (1, "FED")
    OTHER_GOV = (2, "OTHER_GOV")
    INDIV = (3, "INDIV")
    INDUSTRY = (4, "INDUSTRY")
    NETWORK = (5, "NETWORK")
    AMBIG = (6, "AMBIG")
    OTHER = (7, "OTHER")
    UNKNOWN = (8, "UNKNOWN")


class Status(StrEnumWithNumeric):
    ACTIVE_NOT_RECRUITING = (0, "ACTIVE_NOT_RECRUITING")
    COMPLETED = (1, "COMPLETED")
    ENROLLING_BY_INVITATION = (2, "ENROLLING_BY_INVITATION")
    NOT_YET_RECRUITING = (3, "NOT_YET_RECRUITING")
    RECRUITING = (4, "RECRUITING")
    SUSPENDED = (5, "SUSPENDED")
    TERMINATED = (6, "TERMINATED")
    WITHDRAWN = (7, "WITHDRAWN")
    AVAILABLE = (8, "AVAILABLE")
    NO_LONGER_AVAILABLE = (9, "NO_LONGER_AVAILABLE")
    TEMPORARILY_NOT_AVAILABLE = (10, "TEMPORARILY_NOT_AVAILABLE")
    APPROVED_FOR_MARKETING = (11, "APPROVED_FOR_MARKETING")
    WITHHELD = (12, "WITHHELD")
    UNKNOWN = (13, "UNKNOWN")


class Certainty(StrEnumWithNumeric):
    VERY_LOW = (0, "VERY_LOW")
    LOW = (1, "LOW")
    MODERATE = (2, "MODERATE")
    HIGH = (3, "HIGH")
    NO_ESTIMATE = (4, "NO_ESTIMATE")


class Direction(StrEnumWithNumeric):
    UP = (0, "UP")
    DOWN = (1, "DOWN")
    UP_OR_DOWN = (2, "UP_OR_DOWN")
    NO_CHANGE = (3, "NO_CHANGE")


class GroupCode(StrEnumWithNumeric):
    EXP_ARM = (0, "EXP_ARM")
    COMP_ARM = (1, "COMP_ARM")
    TOTAL = (2, "TOTAL")


class Type(StrEnumWithNumeric):
    SERIOUS = (0, "SERIOUS")
    OTHER = (1, "OTHER")


class PValue(StrEnumWithNumeric):
    STUDENT_T_TEST_2_SIDED = (0, "STUDENT_T_TEST_2_SIDED")
    ANOVA = (1, "ANOVA")
    CHI_SQUARED = (2, "CHI_SQUARED")
    CHI_SQUARED_CORRECTION = (3, "CHI_SQUARED_CORRECTION")
    COCHRAN_MANTEL_HAENSZEL = (4, "COCHRAN_MANTEL_HAENSZEL")
    FISHER_EXACT = (5, "FISHER_EXACT")
    KRUSKAL_WALLIS = (6, "KRUSKAL_WALLIS")
    LOG_RANK = (7, "LOG_RANK")
    MCNEMAR = (8, "MCNEMAR")
    WILCOXON_MANN_WHITNEY = (9, "WILCOXON_MANN_WHITNEY")
    REGRESSION_LINEAR = (10, "REGRESSION_LINEAR")
    REGRESSION_LOGISTIC = (11, "REGRESSION_LOGISTIC")
    REGRESSION_COX = (12, "REGRESSION_COX")


class AnnotationType(StrEnumWithNumeric):
    COMMENT = (0, "COMMENT")
    PRIMARY_CP = (1, "PRIMARY_CP")
    SECONDARY_CP = (2, "SECONDARY_CP")


class UnpostedEventType(StrEnumWithNumeric):
    RESET = (0, "RESET")
    RELEASE = (1, "RELEASE")
    UNRELEASE = (2, "UNRELEASE")
    CANCELED = (3, "CANCELED")


class UnpostedAnnotationSource(StrEnumWithNumeric):
    NLM = (0, "NLM")
    SPONSOR = (1, "SPONSOR")


class SubmissionStatus(StrEnumWithNumeric):
    PENDING = (0, "PENDING")
    RELEASED = (1, "RELEASED")
    RESET = (2, "RESET")


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
