from aidose.dataset.utils import include_trial_after_sequential_filtering

from aidose.ctgov.structures import Study

from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
import aidose.ctgov.api_download as api_download

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
