from aidose.ctgov.utils_download import (
    fetch_all_study_nctids_from_api_before_cutoff_date,
    download_all_studies_as_zip,
    unzip_as_separate_jsons_and_delete_zip_file,
    find_files_with_extension_recursively,
    get_study_path_by_nctid_and_raw_dir
)

from aidose.ctgov.utils_pdf import extract_text_from_pdf_using_pymupdf
from aidose.ctgov.utils_pdf import DeepSeekOCRExtractor
from aidose.ctgov.utils_protocol import (
    get_large_protocols_pdf_links,
    get_protocol_pdfs_saved_dir_for_nctid,
    extract_and_concatenate_pdf_texts_for_nctid,
    IncrementalLargeTextExtractor
)

from aidose.ctgov import (CTGOV_DATASET_RAW_PATH,
                          CTGOV_DATASET_PATH,
                          CTGOV_API_DOWNLOAD_BASE_URL,
                          CTGOV_NCTIDS_LIST_ALL_PATH,
                          CTGOV_DATASET_EXTENSIONS_PATH,
                          CTGOV_PROTOCOL_PDF_LINKS_PATH,
                          CTGOV_EXTRACTED_PDFS_DATASET_PATH,
                          EXTRACT_PDFS_USING_DEEPSEEK_OCR)

from aidose.ctgov.structures import Study

from typing import List, Dict
import tqdm
import os
import time
import requests
from datetime import datetime, timezone
import logging
import urllib.request
import shutil
import json

logger = logging.getLogger(__name__)


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
            logger.warning(f"Rate limited: {nct_id}, retrying in {wait:.1f}s...")
            time.sleep(wait)
            continue

        if response.status_code == 404:
            raise RuntimeError(f"NOT FOUND: {nct_id}")

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Too many retries (429): {nct_id}")


def delete_studies_downloaded_after_cutoff(nctids_list_valid: List[str]) -> None:
    all_downloaded_studies_paths = find_files_with_extension_recursively(CTGOV_DATASET_RAW_PATH, "json")

    for path in all_downloaded_studies_paths:
        nctid = os.path.split(path)[-1].split(".json")[0]
        if nctid not in nctids_list_valid:
            os.remove(path)


def download_registry_from_api(knowledge_cutoff_date: datetime | None = None) -> None:
    if os.path.exists(CTGOV_NCTIDS_LIST_ALL_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r") as f:
            nctids_list_all_expected = [nctid.strip() for nctid in f.readlines()]
    else:
        os.makedirs(os.path.dirname(CTGOV_NCTIDS_LIST_ALL_PATH), exist_ok=True)
        nctids_list_all_expected = fetch_all_study_nctids_from_api_before_cutoff_date(CTGOV_API_DOWNLOAD_BASE_URL,
                                                                                      knowledge_cutoff_date)
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "w") as f:
            for nct_id in nctids_list_all_expected:
                f.write(f"{nct_id}\n")

    if not os.path.exists(CTGOV_DATASET_RAW_PATH):
        download_all_studies_as_zip(target_zip_file_path=f"{CTGOV_DATASET_RAW_PATH}.zip",
                                    ctgov_api_download_base_url=CTGOV_API_DOWNLOAD_BASE_URL)
        with open(os.path.join(CTGOV_DATASET_PATH, "download-time-tag.txt"), "w") as f:
            f.write("Download time (UTC): {}\n".format(datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%MZ")))

        unzip_as_separate_jsons_and_delete_zip_file(f"{CTGOV_DATASET_RAW_PATH}.zip", CTGOV_DATASET_RAW_PATH)
        delete_studies_downloaded_after_cutoff(nctids_list_all_expected)

    nctids_list_all_existing = [os.path.split(_path)[-1].split(".")[0] for _path in
                                find_files_with_extension_recursively(CTGOV_DATASET_RAW_PATH, "json")]

    if set(nctids_list_all_existing) != set(nctids_list_all_expected):
        raise RuntimeError("Mismatch between expected and existing NCT IDs in the dataset.")


def download_pdfs_for_all_trials_with_available_documents() -> None:
    if not os.path.exists(CTGOV_PROTOCOL_PDF_LINKS_PATH):
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r") as f:
            nctids_list_all = [nctid.strip() for nctid in f.readlines()]

        nctid_protocol_pdf_map: Dict[str, List[str]] = {}
        for nct_id in tqdm.tqdm(nctids_list_all, desc="Parsing trials to extract document PDF URL's."):
            json_path = get_study_path_by_nctid_and_raw_dir(nct_id, CTGOV_DATASET_RAW_PATH)
            with open(json_path, "r", encoding="utf-8") as f:
                study = Study.model_validate_json(f.read())
                pdf_links = get_large_protocols_pdf_links(study, check_link_status=False)
                if pdf_links:
                    nctid_protocol_pdf_map[nct_id] = pdf_links
        with open(CTGOV_PROTOCOL_PDF_LINKS_PATH, 'w', encoding='utf-8') as f:
            json.dump(nctid_protocol_pdf_map, f, indent=2)

        logger.info("Found {} studies with protocol/SAP PDF links.".format(len(nctid_protocol_pdf_map.keys())))

    with open(CTGOV_PROTOCOL_PDF_LINKS_PATH, "r", encoding="utf-8") as f:
        nctid_protocol_pdf_map = json.load(f)

    os.makedirs(CTGOV_DATASET_EXTENSIONS_PATH, exist_ok=True)
    for nctid, pdf_links in tqdm.tqdm(nctid_protocol_pdf_map.items(),
                                      desc="Downloading existing document PDFs of all trials (if not already done) ..."):
        for link in pdf_links:
            pdf_name = link.split("/")[-1]
            pdf_save_path = os.path.join(
                get_protocol_pdfs_saved_dir_for_nctid(nctid, CTGOV_DATASET_EXTENSIONS_PATH), pdf_name)
            if not os.path.exists(pdf_save_path):
                os.makedirs(os.path.dirname(pdf_save_path), exist_ok=True)
                with urllib.request.urlopen(link) as resp:
                    with open(pdf_save_path, "wb") as out:
                        shutil.copyfileobj(resp, out)
    logger.info("All protocol PDF's now available (either existed before or downloaded now) ...")


def extract_text_incrementally_from_downloaded_document_pdfs():
    with open(CTGOV_PROTOCOL_PDF_LINKS_PATH, "r", encoding="utf-8") as f:
        nctid_protocol_pdf_map = json.load(f)

    if EXTRACT_PDFS_USING_DEEPSEEK_OCR:
        ocr = DeepSeekOCRExtractor()
        extractor_functional = lambda pdf_path: ocr.extract_text_from_pdf(pdf_path)
    else:
        extractor_functional = extract_text_from_pdf_using_pymupdf

    text_extractor_from_pdf_documents = IncrementalLargeTextExtractor(
        save_dir=CTGOV_EXTRACTED_PDFS_DATASET_PATH,
        nctids_list=nctid_protocol_pdf_map.keys(),
        text_extract_func=lambda nctid: extract_and_concatenate_pdf_texts_for_nctid(nctid,
                                                                                    CTGOV_DATASET_EXTENSIONS_PATH,
                                                                                    extractor_functional),
        save_batch_size=1000,
        logger_=logger
    )

    text_extractor_from_pdf_documents.run()


if __name__ == "__main__":
    download_registry_from_api(knowledge_cutoff_date=None)
    download_pdfs_for_all_trials_with_available_documents()
    extract_text_incrementally_from_downloaded_document_pdfs()
