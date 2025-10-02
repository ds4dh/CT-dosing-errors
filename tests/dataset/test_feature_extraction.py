import unittest
from typing import Any, Dict, List

from aidose.dataset.feature_extraction import (
    get_protocol_interventions,
    get_protocol_arm_groups,
    has_protocol,
    has_sap,
    has_icf,
    get_location_details,
    _total_ade_population,  # intentionally testing internal helper
    get_ade_count_attributes_from_positive_terms,
    extract_attributes_from_study,
    ATTRIBS_LABEL_PREFIX,
    ATTRIBS_METADATA_PREFIX,
    CANONICAL_COUNT_PREFIX,
    ATTRIBS_FEATURE_PREFIX
)
from aidose.dataset.utils import get_flow_group_descriptions
from aidose.dataset.attribute import Attribute, AttributesList
from aidose.dataset.ade import (
    ADEAnalysisResultForStudy,
    ADEGroupAggregate,
    PositiveTermMatch,
    LabelMatch,
    ADEClinicalTermStats,
)

from aidose.ctgov.structures import (
    Study,
    ProtocolSection,
    ArmsInterventionsModule,
    ResultsSection,
    ParticipantFlowModule,
    EventGroup,  # not used here, but typically present
    LargeDocumentModule,
    LargeDoc,
    DocumentSection,
    ContactsLocationsModule,
    Location,
    GeoPoint,
    Intervention,
    InterventionType,
    DesignModule,
    EnrollmentInfo,
    DesignInfo,
    MaskingInfo,
    EligibilityModule,
    SponsorCollaboratorsModule,
    LeadSponsor,
    StatusModule,
    DescriptionModule,
    IdentificationModule,
    StudyType,
    Phase,
    PrimaryPurpose,
    Masking,
    Sex,
    AgencyClass,
    Status,
)


class FeaturesExtractorTestCase(unittest.TestCase):
    def _study_with_minimal_protocol(self) -> Study:
        # Interventions
        interventions = [
            Intervention(type=InterventionType.DRUG, description="Drug A"),
            Intervention(type=InterventionType.DEVICE, description="Device B"),
            Intervention(type=InterventionType.DRUG, description="Drug C"),
        ]
        arms_mod = ArmsInterventionsModule(
            interventions=interventions,
            armGroups=[],  # not needed for most assertions here
        )

        # Identification
        ident = IdentificationModule(nctId="NCT00000001")

        # Design
        design = DesignModule(
            studyType=StudyType.INTERVENTIONAL,
            phases=[Phase.PHASE2, Phase.PHASE3],
            enrollmentInfo=EnrollmentInfo(count=123),
            designInfo=DesignInfo(
                allocation="RANDOMIZED",
                interventionModel="PARALLEL",
                primaryPurpose=PrimaryPurpose.TREATMENT,
                maskingInfo=MaskingInfo(masking=Masking.DOUBLE),
            ),
        )

        # Eligibility
        elig = EligibilityModule(
            healthyVolunteers=True,
            sex=Sex.ALL,
            stdAges=[Sex.ALL]  # intentionally non-sense just to exercise enum-list path
        )

        # Sponsor/Status
        sc = SponsorCollaboratorsModule(
            leadSponsor=LeadSponsor(name="Janssen Biotech", class_=AgencyClass.INDUSTRY)
        )
        status = StatusModule(overallStatus=Status.RECRUITING)

        # Descriptions
        desc = DescriptionModule(briefSummary="Brief.", detailedDescription="Detailed.")

        # Contacts/Locations
        locs = ContactsLocationsModule(
            locations=[
                Location(city="Boston", state="MA", country="USA", geoPoint=GeoPoint(lon=-71.0589, lat=42.3601)),
                Location(city="Paris", state=None, country="France", geoPoint=None),
            ]
        )

        protocol = ProtocolSection(
            identificationModule=ident,
            designModule=design,
            eligibilityModule=elig,
            sponsorCollaboratorsModule=sc,
            statusModule=status,
            descriptionModule=desc,
            contactsLocationsModule=locs,
            armsInterventionsModule=arms_mod,
        )

        # Results (flow descriptions)
        pf = ParticipantFlowModule(groups=[{"description": "Exp arm desc"}, {"description": "Ctrl arm desc"}])
        results = ResultsSection(participantFlowModule=pf)

        # Documents (hasProtocol/hasSap/hasIcf)
        ldm = LargeDocumentModule(largeDocs=[
            LargeDoc(hasProtocol=True, hasSap=False, hasIcf=True)
        ])
        docs = DocumentSection(largeDocumentModule=ldm)

        return Study(protocolSection=protocol, resultsSection=results, documentSection=docs)

    def test_intervention_accessors(self):
        study = self._study_with_minimal_protocol()
        ivs = get_protocol_interventions(study)
        self.assertEqual(len(ivs), 3)
        arms = get_protocol_arm_groups(study)
        self.assertEqual(len(arms), 0)

    def test_document_flags(self):
        study = self._study_with_minimal_protocol()
        self.assertTrue(has_protocol(study))
        self.assertFalse(has_sap(study))
        self.assertTrue(has_icf(study))

    def test_flow_group_descriptions(self):
        study = self._study_with_minimal_protocol()
        descs = get_flow_group_descriptions(study)
        self.assertEqual(descs, ["Exp arm desc", "Ctrl arm desc"])

    def test_location_details(self):
        study = self._study_with_minimal_protocol()
        rows = get_location_details(study)
        self.assertEqual(len(rows), 2)
        # Simple contains checks
        self.assertIn("Boston", rows[0])
        self.assertIn("USA", rows[0])
        self.assertIn("Paris", rows[1])
        self.assertIn("France", rows[1])

    def test_total_ade_population(self):
        # Build a tiny ADE result with group populations
        ade_by_group = {
            "EG1": ADEGroupAggregate(population=10, events={}),
            "EG2": ADEGroupAggregate(population=20, events={}),
        }
        ade_res = ADEAnalysisResultForStudy(
            nctid="NCT00000001",
            ade_by_group=ade_by_group,
            ade_clinical={},
            positive_terms={}
        )
        self.assertEqual(_total_ade_population(ade_res), 30)

        # None when no groups
        ade_none = ADEAnalysisResultForStudy(nctid="NCTx", ade_by_group={}, ade_clinical={}, positive_terms={})
        self.assertIsNone(_total_ade_population(ade_none))

    def test_label_count_features_from_positive_terms(self):
        # Positive terms payload mirrors ade_labeling contract
        positive_terms = {
            "Overdose": PositiveTermMatch(
                term="Overdose",
                stats=ADEClinicalTermStats(numAffected=3, numAtRisk=50),
                matches=[LabelMatch(label="Overdose", score=99)]
            ),
            "Seizure": PositiveTermMatch(
                term="Seizure",
                stats=ADEClinicalTermStats(numAffected=2, numAtRisk=50),
                matches=[LabelMatch(label="Seizure", score=98)]
            ),
            # Map another term to the same canonical label "Seizure" to test summation
            "Seizures": PositiveTermMatch(
                term="Seizures",
                stats=ADEClinicalTermStats(numAffected=4, numAtRisk=50),
                matches=[LabelMatch(label="Seizure", score=96)]
            ),
        }

        canonical = ["Overdose", "Seizure", "Fever"]
        feats = get_ade_count_attributes_from_positive_terms(
            positive_terms=positive_terms, canonical_label_cols=canonical
        )

        as_dict = {f.name: f.to_dict()["value"] for f in feats}
        self.assertEqual(as_dict[f"{CANONICAL_COUNT_PREFIX}Overdose"], 3)
        self.assertEqual(as_dict[f"{CANONICAL_COUNT_PREFIX}Seizure"], 6)  # 2 + 4 summed
        self.assertEqual(as_dict[f"{CANONICAL_COUNT_PREFIX}Fever"], 0)

    def test_extract_features_for_study_basic_shape(self):
        # TODO: Updte this broken test to follow the feature-metadata-label division.
        study = self._study_with_minimal_protocol()

        # Minimal ADE result (no events) just to satisfy extractor
        ade_res = ADEAnalysisResultForStudy(
            nctid="NCT00000001",
            ade_by_group={},  # no groups -> population None
            ade_clinical={},  # no terms
            positive_terms={},  # no positives
        )

        feats: AttributesList = extract_attributes_from_study(
            study,
            canonical_label_cols=["Overdose", "Seizure"],
            ade_analysis_results_for_study=ade_res,
            alpha_wilson=0.05,
            wilson_proba_threshold=0.001,
        )

        self.assertIsInstance(feats, list)
        # Ensure some key features exist and are typed correctly (no enum expansion here)
        as_map = {f.name: f for f in feats}

        self.assertIn(f"{ATTRIBS_METADATA_PREFIX}nctId", as_map)
        self.assertEqual(as_map[f"{ATTRIBS_METADATA_PREFIX}nctId"].declared_type, str)
        self.assertEqual(as_map[f"{ATTRIBS_METADATA_PREFIX}nctId"].value, "NCT00000001")

        self.assertIn(f"{ATTRIBS_FEATURE_PREFIX}studyType", as_map)
        self.assertEqual(as_map[f"{ATTRIBS_FEATURE_PREFIX}studyType"].declared_type, StudyType)

        self.assertIn("phases", as_map)
        self.assertEqual(as_map["phases"].declared_type, Phase)
        self.assertIsInstance(as_map["phases"].value, list)  # raw enum list pre-expansion

        self.assertIn("masking", as_map)
        self.assertEqual(as_map["masking"].declared_type, Masking)

        self.assertIn("healthyVolunteers", as_map)
        self.assertEqual(as_map["healthyVolunteers"].declared_type, bool)

        self.assertIn("leadSponsorClass", as_map)
        self.assertEqual(as_map["leadSponsorClass"].declared_type, AgencyClass)

        self.assertIn("overallStatus", as_map)
        self.assertEqual(as_map["overallStatus"].declared_type, Status)

        self.assertIn("isJJ", as_map)
        self.assertEqual(as_map["isJJ"].declared_type, bool)
        self.assertTrue(as_map["isJJ"].value)  # "Janssen Biotech" -> True

        self.assertIn("interventionTypes", as_map)
        self.assertEqual(as_map["interventionTypes"].declared_type, InterventionType)
        self.assertIsInstance(as_map["interventionTypes"].value, list)
        self.assertEqual(len(as_map["interventionTypes"].value), 3)

        # Label features should exist (numeric)
        self.assertIn("label.Overdose", as_map)
        self.assertEqual(as_map["label.Overdose"].declared_type, int)
        self.assertIn("label.Seizure", as_map)
        self.assertEqual(as_map["label.Seizure"].declared_type, int)

        # ADE summary counts
        self.assertIn("num_ct_level_ade_terms", as_map)
        self.assertIn("ct_level_ade_population", as_map)  # None when no groups
        self.assertIsNone(as_map["ct_level_ade_population"].value)


if __name__ == "__main__":
    unittest.main()
