from aidose.ctgov.structures import Study
from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH
import aidose.ctgov.api_download as api_download

import unittest
from pydantic import ValidationError

import os
import json
from typing import List


class CTGOVEntireRegistryParsingAsStudyObjectsIntegrationTest(unittest.TestCase):
    """Test suite to verify that the entire CTGov studies can be parsed as `aidose.ctgov.structures.Study` objects."""

    @classmethod
    def setUpClass(cls):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            api_download.main()

        with open(CTGOV_NCTIDS_LIST_ALL_PATH, 'r') as f:
            cls.nctids_list = [line.strip() for line in f if line.strip()]

    def test_study_parsing_with_pydantic_for_entire_registry(self):
        """Test that all available study JSON files can be parsed as Study objects."""
        parsing_errors: List[tuple[str, str]] = []

        for idx, nct_id in enumerate(self.nctids_list):
            json_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nct_id}.json")

            try:
                with open(json_path, 'r') as f:
                    study_data = json.load(f)

                Study.model_validate(study_data)

            except json.JSONDecodeError as e:
                parsing_errors.append((nct_id, f"JSON decode error: {str(e)}"))
            except ValidationError as e:

                print(idx, ": Exception for NCTID:", idx, nct_id)
                parsing_errors.append((nct_id, f"Pydantic validation error: {str(e)}"))
            except Exception as e:
                parsing_errors.append((nct_id, f"Unexpected error: {str(e)}"))

        if parsing_errors:
            error_msg = "\nParsing errors occurred for the following studies:\n"
            for nct_id, error in parsing_errors:
                error_msg += f"\nNCT ID: {nct_id}\nError: {error}\n"
            self.fail(error_msg)

    def test_available_studies_not_unusually_incomplete(self):
        self.assertTrue(
            len(self.nctids_list) > 500000,
            "No JSON files found to test. Check if the paths are correct and files exist."
        )


if __name__ == '__main__':
    unittest.main()
