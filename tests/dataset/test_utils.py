from aidose.dataset.utils import include_trial_after_sequential_filtering
from aidose.dataset.utils import sanitize_number_from_string
from aidose.dataset.utils import match_terms_fuzzy

from aidose.ctgov.structures import Study

from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
import aidose.ctgov.api_download as api_download

from typing import Dict, Any

import unittest
import os
import json


class CTGovSequentialFilteringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            api_download.main()

        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            cls.nctids_list = [line.strip() for line in f if line.strip()]

    def test_include_trial_after_sequential_filtering(self):

        num_included_trials = 0
        num_excluded_trials = 0

        for idx, nct_id in enumerate(self.nctids_list):
            json_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nct_id}.json")

            with open(json_path, 'r') as f:
                study_data = json.load(f)

            study = Study(**study_data)

            if include_trial_after_sequential_filtering(study):
                num_included_trials += 1
            else:
                num_excluded_trials += 1

        print("Number of included trials: {}".format(num_included_trials))
        print("Number of excluded trials: {}".format(num_excluded_trials))

        self.assertEqual(num_included_trials + num_excluded_trials, len(self.nctids_list))


class SanitizeNumberFromStringTestCase(unittest.TestCase):

    def test_valid_integer_string(self):
        self.assertAlmostEqual(sanitize_number_from_string("1234"), 1234.0)

    def test_valid_integer_with_commas(self):
        self.assertAlmostEqual(sanitize_number_from_string("1,234"), 1234.0)

    def test_valid_float_string(self):
        self.assertAlmostEqual(sanitize_number_from_string("56.78"), 56.78)

    def test_valid_float_with_text(self):
        self.assertAlmostEqual(sanitize_number_from_string("approx. 99.9 patients"), 99.9)

    def test_negative_number(self):
        self.assertAlmostEqual(sanitize_number_from_string("-42"), -42.0)

    def test_string_with_multiple_numbers(self):
        self.assertAlmostEqual(sanitize_number_from_string("First: 100, then 200"), 100.0)

    def test_invalid_string(self):
        self.assertIsNone(sanitize_number_from_string("no numbers here"))

    def test_empty_string(self):
        self.assertIsNone(sanitize_number_from_string(""))

    def test_invalid_numeric_string(self):
        self.assertIsNone(sanitize_number_from_string("--1.0.0"))

    def test_invalid_format_with_multiple_dots(self):
        self.assertIsNone(sanitize_number_from_string("1.2.3"))

    def test_invalid_format_with_multiple_dashes(self):
        self.assertIsNone(sanitize_number_from_string("value: --12.3"))


class MatchTermsFuzzyTest(unittest.TestCase):
    def setUp(self):
        self.candidate_terms: Dict[str, Dict[str, Any]] = {
            "headache": {"numAffected": 10, "numAtRisk": 100},
            "nausea": {"numAffected": 5, "numAtRisk": 100},
            "fatigue": {"numAffected": 8, "numAtRisk": 100},
            "blurred vision": {"numAffected": 2, "numAtRisk": 100},
        }
        self.positive_labels = ["Headache", "Nausea", "Fever"]

    def test_basic_matching(self):
        result = match_terms_fuzzy(self.candidate_terms, self.positive_labels, match_threshold=90)

        self.assertIn("headache", result)
        self.assertIn("nausea", result)
        self.assertNotIn("fatigue", result)
        self.assertNotIn("blurred vision", result)

        self.assertGreaterEqual(result["headache"]["matches"][0]["score"], 90)
        self.assertGreaterEqual(result["nausea"]["matches"][0]["score"], 90)

    def test_no_matches(self):
        result = match_terms_fuzzy(self.candidate_terms, ["unrelated term"], match_threshold=90)
        self.assertEqual(result, {})

    def test_low_threshold_captures_more(self):
        result = match_terms_fuzzy(self.candidate_terms, ["blurry vision"], match_threshold=60)
        self.assertIn("blurred vision", result)

    def test_exact_match_score(self):
        result = match_terms_fuzzy({"fever": {"numAffected": 3, "numAtRisk": 80}}, ["Fever"], match_threshold=100)
        self.assertIn("fever", result)
        self.assertEqual(result["fever"]["matches"][0]["score"], 100)


if __name__ == "__main__":
    unittest.main()
