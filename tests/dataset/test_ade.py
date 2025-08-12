import unittest

from aidose.dataset.ade import (
    ADEAnalysisResultForStudy,
    ADEGroupAggregate,
    ADEEventStats,
    ADEClinicalTermStats,
    aggregate_ade_by_group,
    extract_group_populations,
    process_events_by_group,
    aggregate_ade_clinical_trial_view,
    get_positive_ade_terms,
    normalize_ade_error_message,
)
from aidose.ctgov.structures import (
    Study, ResultsSection, AdverseEventsModule,
    EventGroup, Event, EventStats
)

from pydantic import ValidationError


class AdverseEventsAggregationTestCase(unittest.TestCase):

    def setUp(self):
        self.event_groups = [
            EventGroup(id="EG000", seriousNumAtRisk=10, otherNumAtRisk=10),
            EventGroup(id="EG001", seriousNumAtRisk=20, otherNumAtRisk=20),
        ]

        self.serious_events = [
            Event(
                term="Death",
                stats=[
                    EventStats(groupId="EG000", numAffected=2, numAtRisk=10),
                    EventStats(groupId="EG001", numAffected=3, numAtRisk=20),
                ]
            ),
            Event(
                term="Seizure",
                stats=[
                    EventStats(groupId="EG000", numAffected=1, numAtRisk=10),
                ]
            ),
        ]

        self.other_events = [
            Event(
                term="Fever",
                stats=[
                    EventStats(groupId="EG000", numAffected=4, numAtRisk=10),
                    EventStats(groupId="EG001", numAffected=5, numAtRisk=20),
                ]
            )
        ]

        self.study = Study(
            resultsSection=ResultsSection(
                adverseEventsModule=AdverseEventsModule(
                    eventGroups=self.event_groups,
                    seriousEvents=self.serious_events,
                    otherEvents=self.other_events,
                )
            )
        )

    def test_extract_group_populations(self):
        populations = extract_group_populations(self.study)
        expected = {"EG000": 10, "EG001": 20}
        self.assertEqual(populations, expected)

    def test_process_events_by_group(self):
        group_populations = extract_group_populations(self.study)
        grouped = process_events_by_group(self.serious_events, group_populations)

        self.assertIn("EG000", grouped)
        self.assertIn("Death", grouped["EG000"])
        self.assertEqual(grouped["EG000"]["Death"].numAffected, 2)
        self.assertEqual(grouped["EG000"]["Death"].numAtRisk, 10)

        self.assertEqual(grouped["EG000"]["Seizure"].numAffected, 1)
        self.assertNotIn("Fever", grouped["EG000"])  # from other events, not serious

    def test_aggregate_ade_by_group(self):
        result = aggregate_ade_by_group(self.study)

        self.assertIsInstance(result, dict)
        self.assertIn("EG000", result)
        self.assertEqual(result["EG000"].population, 10)
        self.assertIsInstance(result["EG000"], ADEGroupAggregate)

        events = result["EG000"].events
        self.assertEqual(events["Death"].numAffected, 2)
        self.assertEqual(events["Seizure"].numAffected, 1)
        self.assertEqual(events["Fever"].numAffected, 4)  # from otherEvents

        self.assertEqual(result["EG001"].events["Death"].numAffected, 3)
        self.assertEqual(result["EG001"].events["Fever"].numAffected, 5)

    def test_inconsistent_num_at_risk_raises(self):
        # Introduce inconsistency
        self.study.resultsSection.adverseEventsModule.seriousEvents[0].stats[0].numAtRisk = 999
        with self.assertRaises(ValueError):
            aggregate_ade_by_group(self.study)

    def test_missing_event_term_raises(self):
        # Remove term from an event
        self.study.resultsSection.adverseEventsModule.seriousEvents[0].term = None
        with self.assertRaises(ValueError):
            aggregate_ade_by_group(self.study)

    def test_unknown_group_id_raises(self):
        # Use unknown group ID in stats
        self.study.resultsSection.adverseEventsModule.seriousEvents[0].stats[0].groupId = "UNKNOWN"
        with self.assertRaises(ValueError):
            aggregate_ade_by_group(self.study)


class AggregateADEClinicalTrialViewTest(unittest.TestCase):
    def test_basic_aggregation(self):
        study = Study(
            resultsSection=ResultsSection(
                adverseEventsModule=AdverseEventsModule(
                    eventGroups=[
                        EventGroup(id="EG001", seriousNumAtRisk=10, otherNumAtRisk=10),
                        EventGroup(id="EG002", seriousNumAtRisk=20, otherNumAtRisk=20),
                    ],
                    seriousEvents=[
                        Event(term="Nausea", stats=[
                            EventStats(groupId="EG001", numAffected=3, numAtRisk=10),
                            EventStats(groupId="EG002", numAffected=5, numAtRisk=20),
                        ])
                    ]
                )
            )
        )

        clinical = aggregate_ade_clinical_trial_view(study)

        # Expect typed stats, not raw dicts
        self.assertIn("Nausea", clinical)
        self.assertIsInstance(clinical["Nausea"], ADEClinicalTermStats)
        self.assertEqual(clinical["Nausea"].numAffected, 8)
        self.assertEqual(clinical["Nausea"].numAtRisk, 30)

    def test_empty_events(self):
        study = Study(
            resultsSection=ResultsSection(
                adverseEventsModule=AdverseEventsModule(
                    eventGroups=[EventGroup(id="EG001", seriousNumAtRisk=5, otherNumAtRisk=5)],
                    seriousEvents=[],
                    otherEvents=[]
                )
            )
        )

        clinical = aggregate_ade_clinical_trial_view(study)

        self.assertEqual(clinical, {})  # still know population is 5 if needed elsewhere


class GetPositiveADETermsTestCase(unittest.TestCase):
    def test_terms_with_positive_num_affected(self):
        data = {
            "Headache": ADEClinicalTermStats(numAffected=5, numAtRisk=100),
            "Nausea": ADEClinicalTermStats(numAffected=0, numAtRisk=100),
            "Vomiting": ADEClinicalTermStats(numAffected=3, numAtRisk=100),
        }
        result = get_positive_ade_terms(data)
        self.assertEqual(set(result), {"Headache", "Vomiting"})

    def test_all_zero(self):
        data = {
            "Dizziness": ADEClinicalTermStats(numAffected=0, numAtRisk=100),
            "Fatigue": ADEClinicalTermStats(numAffected=0, numAtRisk=100),
        }
        result = get_positive_ade_terms(data)
        self.assertEqual(result, [])

    def test_all_positive(self):
        data = {
            "Pain": ADEClinicalTermStats(numAffected=1, numAtRisk=50),
            "Fever": ADEClinicalTermStats(numAffected=10, numAtRisk=50),
        }
        result = get_positive_ade_terms(data)
        self.assertEqual(set(result), {"Pain", "Fever"})


class NormalizeADEErrorMessageTestCase(unittest.TestCase):
    def test_known_errors(self):
        self.assertEqual(
            normalize_ade_error_message("Invalid at-risk numbers for group EG001"),
            "Invalid at-risk numbers"
        )
        self.assertEqual(
            normalize_ade_error_message("Inconsistent at-risk numbers for group EG002"),
            "Inconsistent at-risk numbers"
        )
        self.assertEqual(
            normalize_ade_error_message("Group ID EG003 found in stats but not in eventGroups"),
            "Group ID not in eventGroups"
        )
        self.assertEqual(
            normalize_ade_error_message("Inconsistent numAtRisk for group EG004"),
            "Inconsistent numAtRisk"
        )
        self.assertEqual(
            normalize_ade_error_message("Invalid ADE term: term is missing or empty."),
            "Invalid ADE term"
        )

    def test_other_error(self):
        self.assertEqual(
            normalize_ade_error_message("Some unexpected issue occurred."),
            "Other Error"
        )


class ADESerializationTestCase(unittest.TestCase):
    def setUp(self):
        # Minimal but non-trivial study
        self.study = Study(
            resultsSection=ResultsSection(
                adverseEventsModule=AdverseEventsModule(
                    eventGroups=[
                        EventGroup(id="EG000", seriousNumAtRisk=10, otherNumAtRisk=10),
                        EventGroup(id="EG001", seriousNumAtRisk=20, otherNumAtRisk=20),
                    ],
                    seriousEvents=[
                        Event(
                            term="Nausea",
                            stats=[
                                EventStats(groupId="EG000", numAffected=3, numAtRisk=10),
                                EventStats(groupId="EG001", numAffected=5, numAtRisk=20),
                            ],
                        )
                    ],
                    otherEvents=[
                        Event(
                            term="Headache",
                            stats=[
                                EventStats(groupId="EG000", numAffected=2, numAtRisk=10),
                                EventStats(groupId="EG001", numAffected=4, numAtRisk=20),
                            ],
                        )
                    ],
                )
            )
        )

    def test_round_trip_json(self):
        # Build components and compose a result model
        ade_by_group = aggregate_ade_by_group(self.study)
        ade_clinical = aggregate_ade_clinical_trial_view(self.study)

        result = ADEAnalysisResultForStudy(
            nctid="NCT-TEST-000",
            ade_by_group=ade_by_group,
            ade_clinical=ade_clinical,
            # leave positive_terms empty here to avoid depending on fuzzy matching in this test
        )

        # Serialize -> JSON
        json_str = result.model_dump_json()

        # Deserialize <- JSON
        restored = ADEAnalysisResultForStudy.model_validate_json(json_str)

        # Spot-check equality
        self.assertEqual(restored.nctid, result.nctid)
        self.assertEqual(set(restored.ade_by_group.keys()), set(result.ade_by_group.keys()))
        self.assertIn("EG000", restored.ade_by_group)
        self.assertIn("Nausea", restored.ade_clinical)
        self.assertEqual(restored.ade_clinical["Nausea"].numAffected, 8)
        self.assertEqual(restored.ade_clinical["Nausea"].numAtRisk, 30)

        # Ensure nested types survived the round trip
        self.assertIsInstance(restored.ade_by_group["EG000"], ADEGroupAggregate)
        self.assertIsInstance(restored.ade_clinical["Nausea"], ADEClinicalTermStats)

    def test_frozen_models_are_immutable(self):
        stats = ADEClinicalTermStats(numAffected=1, numAtRisk=10)
        with self.assertRaises((TypeError, ValidationError)):
            stats.numAffected = 999  # type: ignore[attr-defined]


class ADEGroupAggregateSerializationTestCase(unittest.TestCase):
    def test_group_aggregate_json(self):
        agg = ADEGroupAggregate(
            population=15,
            events={"Dizziness": ADEEventStats(numAffected=2, numAtRisk=15)},
        )

        payload = agg.model_dump()
        self.assertEqual(payload["population"], 15)
        self.assertIn("Dizziness", payload["events"])

        json_str = agg.model_dump_json()
        restored = ADEGroupAggregate.model_validate_json(json_str)
        self.assertEqual(restored.population, 15)
        self.assertEqual(restored.events["Dizziness"].numAffected, 2)
        self.assertEqual(restored.events["Dizziness"].numAtRisk, 15)


if __name__ == "__main__":
    unittest.main()
