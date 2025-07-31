from aidose.dataset.ade import aggregate_ade_by_group
from aidose.dataset.ade import extract_group_populations
from aidose.dataset.ade import process_events_by_group

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


if __name__ == "__main__":
    unittest.main()
