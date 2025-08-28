import requests
import zipfile
import time
import json
import os
from datetime import datetime

from typing import List, Dict

PAGE_SIZE = 1000
SLEEP_BETWEEN_PAGES = 0.1


def fetch_all_study_nctids_from_api_before_cutoff_date(
        ctgov_api_download_base_url: str,
        knowledge_cutoff_date: datetime | None
) -> List[str]:
    """
    Fetches all NCT IDs for studies posted on or before a specific date.


    Conforming to the guidelines in https://clinicaltrials.gov/data-api/api
    This function queries the ClinicalTrials.gov API to retrieve a list of all
    study NCT IDs that were publicly available as of the given cutoff date,
    ensuring reproducibility of the dataset.

    Args:
        ctgov_api_download_base_url: The base URL for the ClinicalTrials.gov API.
        knowledge_cutoff_date: A datetime object for the cutoff date.
                               Only trials posted on or before this date will be
                               returned. 


    Returns:
        A sorted list of NCT IDs matching the criteria.
    """
    if knowledge_cutoff_date:
        date_str_for_api = knowledge_cutoff_date.strftime('%Y-%m-%d')
        print(f"Fetching all NCTIDs for studies posted on or before {date_str_for_api}...")
    else:
        print("Fetching all NCTIDs for all studies (no date filter)...")

    nct_ids: List[str] = []
    page_token = None

    while True:
        params = {
            "format": "json",
            "fields": "NCTId",
            "pageSize": PAGE_SIZE
        }

        if knowledge_cutoff_date:
            date_str_for_api = knowledge_cutoff_date.strftime('%Y-%m-%d')
            query_expr = f"AREA[StudyFirstPostDate]RANGE[MIN,{date_str_for_api}]"
            params['query.term'] = query_expr

        if page_token:
            params["pageToken"] = page_token

        try:
            response = requests.get(f"{ctgov_api_download_base_url}/studies", params=params)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"An API error occurred: {e}")
            break

        ids = [study["protocolSection"]["identificationModule"]["nctId"]
               for study in data.get("studies", [])]
        nct_ids.extend(ids)

        page_token = data.get("nextPageToken")
        if not page_token:
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    print(f"Found {len(nct_ids)} NCTIDs.")
    return sorted(nct_ids)


def fetch_study_json_by_nctid_from_request(nct_id: str, ctgov_api_download_base_url: str) -> Dict:
    response = requests.get(f"{ctgov_api_download_base_url}/studies/{nct_id}", params={"format": "json"})
    if response.status_code == 404:
        raise RuntimeError(f"NOT FOUND: {nct_id}")
    response.raise_for_status()
    study_json = response.json()

    return study_json


def save_study_dict_as_json(nctid: str, study_json: Dict, dir_path: str) -> None:
    with open(os.path.join(dir_path, "{}.json".format(nctid)), "w") as f:
        json.dump(study_json, f, indent=2)


def download_all_studies_as_zip(target_zip_file_path: str, ctgov_api_download_base_url: str) -> None:
    url = f"{ctgov_api_download_base_url}/studies/download?format=json.zip"

    print(f"Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(target_zip_file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

        print(f"Downloaded {os.path.split(target_zip_file_path)[-1]} to {os.path.dirname(target_zip_file_path)}")


def unzip_and_delete_zip_file(zip_file_path: str, target_dir: str) -> None:
    os.makedirs(target_dir, exist_ok=False)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(zip_file_path)
    print(f"The source zip file {zip_file_path} was extracted to {target_dir} and then deleted.")
