from aidose.ctgov import CTGOV_API_DOWNLOAD_BASE_URL

from aidose.ctgov.utils import (
    fetch_all_study_nctids_from_api_before_cutoff_date,
)

import unittest
from datetime import datetime


class TestLiveApiFetch(unittest.TestCase):
    """
    An integration test that hits the live ClinicalTrials.gov API.
    """

    def test_fetch_count_for_a_fixed_historical_date(self):
        # A date far in the past ensures the result set is stable and won't change.
        knowledge_cutoff_date = datetime(2002, 12, 31)

        expected_count = 5270

        print(f"\n[INFO] Running live API test for date <= {knowledge_cutoff_date}...")
        print("[INFO] This will take a few moments as it fetches several thousand records...")

        # --- Act ---
        nct_ids = fetch_all_study_nctids_from_api_before_cutoff_date(
            ctgov_api_download_base_url=CTGOV_API_DOWNLOAD_BASE_URL,
            knowledge_cutoff_date=knowledge_cutoff_date
        )

        # --- Assert ---
        actual_count = len(nct_ids)
        print(f"[INFO] Test finished. Expected count: {expected_count}, Got: {actual_count}")

        self.assertEqual(actual_count, expected_count)
        self.assertTrue(all(isinstance(nid, str) and nid.startswith("NCT") for nid in nct_ids))


if __name__ == '__main__':
    unittest.main()
