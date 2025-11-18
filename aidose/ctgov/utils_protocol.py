from aidose.ctgov.structures import Study

from datasets import Dataset, load_from_disk, concatenate_datasets

from typing import List, Any, Callable
import requests
import os
from tqdm import tqdm
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)


# =========================
# Intervention accessors
# =========================

def get_protocol_interventions(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.interventions or []


def get_protocol_arm_groups(study: Study) -> List[Any]:
    ps = study.protocolSection
    if not ps or not ps.armsInterventionsModule:
        return []
    return ps.armsInterventionsModule.armGroups or []


# =========================
# Document helpers
# =========================

def _has_doc_flag(study: Study, flag_name: str) -> bool:
    ds = study.documentSection
    if not ds or not ds.largeDocumentModule:
        return False
    large_docs = ds.largeDocumentModule.largeDocs or []
    for doc in large_docs:
        val = getattr(doc, flag_name, None)
        if isinstance(val, bool) and val:
            return True
    return False


def has_protocol(study: Study) -> bool: return _has_doc_flag(study, "hasProtocol")


def has_sap(study: Study) -> bool:       return _has_doc_flag(study, "hasSap")


def has_icf(study: Study) -> bool:       return _has_doc_flag(study, "hasIcf")


def get_large_protocols_pdf_links(study: Study, check_link_status: bool = False) -> List[str] | None:
    if not has_protocol(study):
        return None
    large_docs = study.documentSection.largeDocumentModule.largeDocs
    if not large_docs:
        return None
    links: List[str] = []
    nctid = study.protocolSection.identificationModule.nctId
    subfolder = nctid[-2:]  # Last two characters of NctId
    for doc in large_docs:
        filename = doc.filename
        if isinstance(filename, str) and filename.endswith(".pdf"):
            link = "https://cdn.clinicaltrials.gov/large-docs/{}/{}/{}".format(subfolder, nctid, filename)
            if check_link_status:
                try:
                    response = requests.head(link, timeout=5)
                    if response.status_code != 200:
                        raise RuntimeError(f"URL not found or not accessible: {link} (status {response.status_code})")

                except requests.RequestException as e:
                    raise RuntimeError(f"Error checking URL {link}: {e}")

            links.append(link)

    return links if links else None


def get_protocol_pdfs_saved_dir_for_nctid(nctid: str, extensions_dir: str) -> str | None:
    parent_identifier = nctid[-2:]
    return os.path.join(extensions_dir, "protocol-pdfs", parent_identifier, nctid)


def extract_and_concatenate_pdf_texts_for_nctid(nctid: str,
                                                extensions_dir: str,
                                                text_extract_func_from_pdf: Callable[[str], str]) -> str | None:
    pdf_dir = get_protocol_pdfs_saved_dir_for_nctid(nctid, extensions_dir)
    if not os.path.exists(pdf_dir):
        return None
    all_texts = []
    for pdf_name in [_file for _file in os.listdir(pdf_dir) if _file.endswith(".pdf")]:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        text = text_extract_func_from_pdf(pdf_path)
        all_texts.append("{}: \n{}".format(pdf_name, text))
    return "\n".join(all_texts) if all_texts else None


# =========================
# Incremental large text extractor
# =========================


class IncrementalLargeTextExtractor:
    def __init__(
            self,
            save_dir: str,
            nctids_list: List[str],
            text_extract_func: Callable[[str], str | None],
            save_batch_size: int = 1000,
            logger_: logging.Logger | None = None,
    ):
        """
        Parameters
        ----------
        save_dir : str
            Directory where the dataset will be stored (Arrow format).
        nctids_list : List[str]
            List of NCT IDs to process.
        text_extract_func : Callable[[str], str]
            Function to extract the large text for a given NCT ID.
        save_batch_size : int, optional
            Number of new records to accumulate before saving progress.
        logger_ : logging.Logger, optional
            Logger instance (defaults to a module-level logger).
        """
        self.save_dir = save_dir
        self.nctids_list = nctids_list
        self.text_extract_func = text_extract_func
        self.save_batch_size = save_batch_size
        self.logger = logger_ or logging.getLogger(__name__)

        self._load_or_initialize_dataset()

    def _load_or_initialize_dataset(self):
        if os.path.exists(self.save_dir):
            self.dataset = load_from_disk(self.save_dir)
            self.done_nctids = set(self.dataset["nctid"])
            self.logger.info(
                f"Loaded existing dataset with {len(self.done_nctids)} items already processed."
            )
        else:
            self.dataset = Dataset.from_dict({"nctid": [], "extracted_text": []})
            self.done_nctids = set()
            self.logger.info("Starting a new extraction dataset.")

    def run(self):
        """Run the extraction loop incrementally, saving progress periodically."""
        new_records = []
        total_items = len(self.nctids_list)
        processed = len(self.done_nctids)

        with tqdm(
                total=total_items,
                desc="Looping over downloaded PDF's, serializing them as text, then saving as a standalone dataset...",
                unit="item",
                initial=processed,
                dynamic_ncols=True,
                leave=True
        ) as pbar:

            for nctid in self.nctids_list:
                if nctid in self.done_nctids:
                    pbar.update(1)
                    continue

                try:
                    text = self.text_extract_func(nctid)
                except Exception as e:
                    self.logger.warning(f"Failed to extract {nctid}: {e}")
                    pbar.update(1)
                    continue

                new_records.append({"nctid": nctid, "extracted_text": text})
                processed += 1
                pbar.update(1)

                if len(new_records) >= self.save_batch_size:
                    self._save_progress(new_records)
                    new_records = []
                    self.logger.info(
                        f"Progress saved. {processed}/{total_items} items processed."
                    )

            if new_records:
                self._save_progress(new_records)
                self.logger.info(
                    f"Final batch saved. {processed}/{total_items} items processed."
                )

        self.logger.info("Extraction complete.")

    def _save_progress(self, new_records: List[dict]):
        """Append new records and safely save dataset to disk."""
        new_ds = Dataset.from_list(new_records)
        combined = concatenate_datasets([self.dataset, new_ds])

        with tempfile.TemporaryDirectory() as tmp_dir:
            combined.save_to_disk(tmp_dir)

            if os.path.exists(self.save_dir):
                shutil.rmtree(self.save_dir)
            shutil.move(tmp_dir, self.save_dir)

        self.dataset = load_from_disk(self.save_dir)
        self.done_nctids.update([r["nctid"] for r in new_records])
