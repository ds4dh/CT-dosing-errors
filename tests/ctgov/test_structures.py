from aidose.ctgov.structures import StrEnumWithNumericDeprecated
from aidose.ctgov.structures import Study
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
from aidose.ctgov.main import download_registry_from_api
from aidose.ctgov.utils import get_study_path_by_nctid_and_raw_dir

import unittest
from pydantic import ValidationError

import os
import json
from typing import List
import tqdm


class SampleEnum(StrEnumWithNumericDeprecated):
    FOO = (0, "FOO")
    BAR = (1, "BAR")


class StrEnumWithNumericTest(unittest.TestCase):
    def test_string_value(self):
        self.assertEqual(str(SampleEnum.FOO), "FOO")
        self.assertEqual(str(SampleEnum.BAR), "BAR")

    def test_numeric_value(self):
        self.assertEqual(int(SampleEnum.FOO), 0)
        self.assertEqual(int(SampleEnum.BAR), 1)

    def test_value_property(self):
        self.assertEqual(SampleEnum.FOO.value, "FOO")
        self.assertEqual(SampleEnum.BAR.value, "BAR")

    def test_numeric_property(self):
        self.assertEqual(SampleEnum.FOO.numeric, 0)
        self.assertEqual(SampleEnum.BAR.numeric, 1)


class CTGOVEntireRegistryParsingAsStudyObjectsIntegrationTest(unittest.TestCase):
    """Test suite to verify that the entire CTGov studies can be parsed as `aidose.ctgov.structures.Study` objects."""

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            download_registry_from_api()

        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            cls.nctids_list = [line.strip() for line in f if line.strip()]

    def test_study_parsing_with_pydantic_for_entire_registry(self):
        """Test that all available study JSON files can be parsed as Study objects."""
        for nctid in tqdm.tqdm(self.nctids_list, desc="Testing all studies for successful parsing .."):
            json_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
            with open(json_path, 'r') as f:
                Study.model_validate_json(f.read())

    def test_available_studies_not_unusually_incomplete(self):
        self.assertTrue(
            len(self.nctids_list) > 500000,
            "No JSON files found to test. Check if the paths are correct and files exist."
        )


class ManuallyInspectStudyObject(unittest.TestCase):
    def setUp(self):
        example_nctid = "NCT07123909"
        with open(os.path.join(CTGOV_DATASET_RAW_PATH, f"{example_nctid}.json"), 'r') as f:
            self.study_data = json.load(f)

        self.study = Study(**self.study_data)

    def test(self):
        print(self.study_data)
        print(self.study)


if __name__ == '__main__':
    unittest.main()
