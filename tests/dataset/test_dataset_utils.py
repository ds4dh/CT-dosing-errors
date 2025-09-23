from aidose.dataset.utils import include_trial_after_sequential_filtering
from aidose.dataset.utils import sanitize_number_from_string
from aidose.dataset.utils import match_terms_fuzzy
from aidose.ctgov.utils_download import get_study_path_by_nctid_and_raw_dir

from aidose.ctgov.structures import Study, Status

from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
from aidose.ctgov import download_registry_from_api

from typing import Dict, Any, List

import unittest
import tqdm
import os
from datetime import datetime


class CTGovSequentialFilteringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            download_registry_from_api()

        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            cls.nctids_list = [line.strip() for line in f if line.strip()]

        cls.studies_list_included: List[Study] = []
        cls.nctids_list_excluded: List[str] = []

        for nctid in tqdm.tqdm(cls.nctids_list, desc="Parsing all studies and filtering them for inclusion .."):
            json_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)

            with open(json_path, 'r') as f:
                study = Study.model_validate_json(f.read())

            if include_trial_after_sequential_filtering(study, datetime.today()):
                cls.studies_list_included.append(study)
                cls.studies_list_included.append(study)
            else:
                cls.nctids_list_excluded.append(nctid)

    def test_include_trial_after_sequential_filtering(self):

        num_included_trials = len(self.studies_list_included)
        num_excluded_trials = len(self.nctids_list_excluded)

        print("Number of included trials: {}".format(num_included_trials))
        print("Number of excluded trials: {}".format(num_excluded_trials))

        self.assertEqual(num_included_trials + num_excluded_trials, len(self.nctids_list))

    def test_some_stats_from_included_trials(self):
        self.assertTrue(len(self.studies_list_included) > 40000, "Expected more than 10,000 included trials.")

        completed_studies_list: List[Study] = []
        terminated_studies_list: List[Study] = []

        completed_primary_completion_dates: List[datetime | None] = []
        completed_completion_dates: List[datetime | None] = []
        terminated_primary_completion_dates: List[datetime | None] = []
        terminated_completion_dates: List[datetime | None] = []

        for study in self.studies_list_included:
            sm = study.protocolSection.statusModule
            ds = getattr(sm, "completionDateStruct", None)
            pds = getattr(sm, "primaryCompletionDateStruct", None)
            if ds:
                cd = ds.date.dt
            else:
                cd = None
            if pds:
                pcd = pds.date.dt
            else:
                pcd = None

            if sm.overallStatus == Status.COMPLETED:
                completed_studies_list.append(study)
                completed_completion_dates.append(cd)
                completed_primary_completion_dates.append(pcd)
            if sm.overallStatus == Status.TERMINATED:
                terminated_studies_list.append(study)
                terminated_completion_dates.append(cd)
                terminated_primary_completion_dates.append(pcd)

        print("Number of completed trials: {}".format(len(completed_studies_list)))
        print("Number of terminated trials: {}".format(len(terminated_studies_list)))


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
