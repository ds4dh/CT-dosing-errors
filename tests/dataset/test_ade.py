from aidose.dataset.ade import aggregate_ade_by_group
from aidose.dataset.ade import extract_group_populations
from aidose.dataset.ade import process_events_by_group
from aidose.dataset.ade import aggregate_ade_clinical_trial_view
from aidose.dataset.ade import get_positive_ade_terms
from aidose.dataset.ade import normalize_ade_error_message
from aidose.dataset.ade import ADEEventStats
from aidose.ctgov.structures import Study, ResultsSection, AdverseEventsModule, EventGroup, Event, EventStats

import unittest


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

        result, total_population = aggregate_ade_clinical_trial_view(study)

        expected_result = {
            "Nausea": {"numAffected": 8, "numAtRisk": 30}
        }

        self.assertEqual(result, expected_result)
        self.assertEqual(total_population, 30)

    def test_empty_events(self):
        study = Study(
            resultsSection=ResultsSection(
                adverseEventsModule=AdverseEventsModule(
                    eventGroups=[
                        EventGroup(id="EG001", seriousNumAtRisk=5, otherNumAtRisk=5),
                    ],
                    seriousEvents=[],
                    otherEvents=[]
                )
            )
        )

        result, total_population = aggregate_ade_clinical_trial_view(study)

        self.assertEqual(result, {})
        self.assertEqual(total_population, 5)


class GetPositiveADETermsTestCase(unittest.TestCase):
    def test_terms_with_positive_num_affected(self):
        data = {
            "Headache": ADEEventStats(numAffected=5, numAtRisk=100),
            "Nausea": ADEEventStats(numAffected=0, numAtRisk=100),
            "Vomiting": ADEEventStats(numAffected=3, numAtRisk=100),
        }
        result = get_positive_ade_terms(data)
        self.assertEqual(set(result), {"Headache", "Vomiting"})

    def test_all_zero_or_none(self):
        data = {
            "Dizziness": ADEEventStats(numAffected=0, numAtRisk=100),
            "Fatigue": ADEEventStats(numAffected=None, numAtRisk=100),
        }
        result = get_positive_ade_terms(data)
        self.assertEqual(result, [])

    def test_all_positive(self):
        data = {
            "Pain": ADEEventStats(numAffected=1, numAtRisk=50),
            "Fever": ADEEventStats(numAffected=10, numAtRisk=50),
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
            normalize_ade_error_message("Invalid ADE term: term is missing or empty"),
            "Invalid ADE term"
        )

    def test_other_error(self):
        self.assertEqual(
            normalize_ade_error_message("Some unexpected issue occurred."),
            "Other Error"
        )


if __name__ == "__main__":
    unittest.main()
