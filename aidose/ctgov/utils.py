import os.path

from aidose.ctgov import CTGOV_API_DOWNLOAD_BASE_URL
from typing import List, Dict

import requests
import time
import json

PAGE_SIZE = 1000
SLEEP_BETWEEN_PAGES = 0.1


def fetch_all_study_nctids_from_request() -> List[str]:
    # Conforming to the guidelines in https://clinicaltrials.gov/data-api/api
    nct_ids: List[str] = []
    page_token = None

    while True:
        params = {
            "format": "json",
            "fields": "NCTId",
            "pageSize": PAGE_SIZE
        }
        if page_token:
            params["pageToken"] = page_token

        response = requests.get(f"{CTGOV_API_DOWNLOAD_BASE_URL}/studies", params=params)
        response.raise_for_status()
        data = response.json()

        ids = [study["protocolSection"]["identificationModule"]["nctId"]
               for study in data.get("studies", [])]
        nct_ids.extend(ids)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    return sorted(nct_ids)


def fetch_study_json_by_nctid_from_request(nct_id: str) -> Dict:
    response = requests.get(f"{CTGOV_API_DOWNLOAD_BASE_URL}/studies/{nct_id}", params={"format": "json"})
    if response.status_code == 404:
        raise RuntimeError(f"NOT FOUND: {nct_id}")
    response.raise_for_status()
    study_json = response.json()

    return study_json


def save_study_dict_as_json(nctid: str, study_json: Dict, dir_path: str) -> None:
    with open(os.path.join(dir_path, "{}.json".format(nctid)), "w") as f:
        json.dump(study_json, f, indent=2)
