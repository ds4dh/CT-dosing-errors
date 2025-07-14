from aidose.ctgov.utils import (
    fetch_all_study_nctids_from_request,
    save_study_dict_as_json
)
from aidose import RESOURCES_DIR
from aidose.ctgov import CTGOV_DATASET_RAW_PATH, CTGOV_API_DOWNLOAD_BASE_URL

import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

CTGOV_NCTIDS_LIST_ALL_PATH = os.path.join(RESOURCES_DIR, "ctgov", "ctgov_nctids_list_all.txt")
MAX_WORKERS = 12
PER_REQUEST_SLEEP = 0.1


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


def download_and_save_study_if_missing(nctid: str) -> tuple[str, str | None]:
    output_path = os.path.join(CTGOV_DATASET_RAW_PATH, f"{nctid}.json")

    if os.path.exists(output_path):
        return (nctid, None)

    try:
        time.sleep(PER_REQUEST_SLEEP)  # Gentle throttle
        study = fetch_study_json_with_retries(nctid)
        save_study_dict_as_json(nctid, study, CTGOV_DATASET_RAW_PATH)
        return (nctid, None)
    except Exception as e:
        return (nctid, str(e))


def main():
    if os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r") as f:
            nctids_list = [nctid.strip() for nctid in f.readlines()]
    else:
        os.makedirs(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH), exist_ok=True)
        nctids_list = fetch_all_study_nctids_from_request()
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "w") as f:
            for nct_id in nctids_list:
                f.write(f"{nct_id}\n")

    print(f"Processing {len(nctids_list)} studies with {MAX_WORKERS} threads...")

    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_and_save_study_if_missing, nctid): nctid
            for nctid in nctids_list
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading (skips existing)"):
            nctid, error = future.result()
            if error:
                print(f"ERROR: {nctid} -> {error}")
                errors.append(nctid)

    if errors:
        print(f"\n{len(errors)} downloads failed.")
    else:
        print("\nAll missing studies downloaded successfully.")


if __name__ == "__main__":
    main()