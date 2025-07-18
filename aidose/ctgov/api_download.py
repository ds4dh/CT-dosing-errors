from aidose.ctgov.utils import (
    fetch_all_study_nctids_from_request,
    download_all_studies_as_zip,
    unzip_and_delete_zip_file,
)
from aidose.ctgov import CTGOV_DATASET_RAW_PATH, CTGOV_API_DOWNLOAD_BASE_URL, CTGOV_NCTIDS_LIST_ALL_PATH

import os
import time
import requests



def fetch_study_json_with_retries(nct_id: str) -> dict:
    retries = 5
    backoff = 1.0

    for attempt in range(retries):
        response = requests.get(
            f"{CTGOV_API_DOWNLOAD_BASE_URL}/studies/{nct_id}",
            params={"format": "json"}
        )

        if response.status_code == 429:
            wait = backoff * (2 ** attempt)
            print(f"Rate limited: {nct_id}, retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        if response.status_code == 404:
            raise RuntimeError(f"NOT FOUND: {nct_id}")

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Too many retries (429): {nct_id}")


def main():
    if os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r") as f:
            nctids_list_all_expected = [nctid.strip() for nctid in f.readlines()]
    else:
        os.makedirs(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH), exist_ok=True)
        nctids_list_all_expected = fetch_all_study_nctids_from_request(CTGOV_API_DOWNLOAD_BASE_URL)
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "w") as f:
            for nct_id in nctids_list_all_expected:
                f.write(f"{nct_id}\n")

    if not os.path.exists(CTGOV_DATASET_RAW_PATH):
        download_all_studies_as_zip(target_zip_file_path=f"{CTGOV_DATASET_RAW_PATH}.zip",
                                    ctgov_api_download_base_url=CTGOV_API_DOWNLOAD_BASE_URL)

        unzip_and_delete_zip_file(f"{CTGOV_DATASET_RAW_PATH}.zip", CTGOV_DATASET_RAW_PATH)

    nctids_list_all_existing = [study.split(".json")[0] for study in os.listdir(CTGOV_DATASET_RAW_PATH) if
                                study.endswith(".json")]
    if set(nctids_list_all_existing) != set(nctids_list_all_expected):
        raise RuntimeError("Mismatch between expected and existing NCT IDs in the dataset.")


if __name__ == "__main__":
    main()
