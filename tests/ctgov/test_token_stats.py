from aidose.ctgov.constants import CTGOV_NCTIDS_LIST_ALL_PATH, CTGOV_DATASET_RAW_PATH, CTGOV_DATASET_EXTENSIONS_PATH
from aidose.ctgov.utils_download import get_study_path_by_nctid_and_raw_dir
from aidose.ctgov.utils_pdf import extract_text_from_pdf_using_pymupdf
from aidose import RESOURCES_DIR

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

import unittest
import random
import csv
import tqdm
import os
import json

TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "bert-base-uncased")
HF_LOCAL_FILES_ONLY = os.environ.get("HF_LOCAL_FILES_ONLY", "0").strip() == "1"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _compact_json_text(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class CTGovTokenizationStatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Quiet HF warnings like "sequence length > model max"
        hf_logging.set_verbosity_error()

        # Read NCTIDs
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r", encoding="utf-8") as f:
            cls.nctids = [line.strip() for line in f if line.strip()]
        if not cls.nctids:
            raise unittest.SkipTest("NCTID list is empty.")

        # Load tokenizer (cache-only if requested)
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME, local_files_only=HF_LOCAL_FILES_ONLY
            )
            # Some tokenizers (e.g., GPT-2) lack a pad token; set to EOS to avoid warnings.
            if cls.tokenizer.pad_token is None and cls.tokenizer.eos_token is not None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            # We only count tokens; prevent "length > 512" warnings.
            cls.tokenizer.model_max_length = 10 ** 12
        except Exception as e:
            raise unittest.SkipTest(f"Tokenizer '{TOKENIZER_NAME}' not available: {e}")

        # Output CSV path
        cls.output_csv = os.path.join(RESOURCES_DIR, "ctgov_tokenization_stats.csv")
        os.makedirs(os.path.dirname(cls.output_csv), exist_ok=True)

    def test_tokenize_and_write_csv(self):
        rows_written = 0

        with open(self.output_csv, "w", encoding="utf-8", newline="") as outf:
            writer = csv.writer(outf)
            writer.writerow(["model_name", "nctid", "file", "abs_path", "char_count", "token_count"])

            sum_all_tokens = 0
            sum_all_chars = 0

            for nctid in tqdm.tqdm(self.nctids, desc="Reading JSON and counting tokens .."):
                json_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
                if not os.path.exists(json_path):
                    continue

                try:
                    try:
                        with open(json_path, "r", encoding="utf-8") as jf:
                            obj = json.load(jf)
                    except UnicodeDecodeError:
                        with open(json_path, "r", encoding="latin-1") as jf:
                            obj = json.load(jf)

                    text = _compact_json_text(obj)
                    char_count = len(text)
                    # Count full tokens; no truncation
                    token_count = len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))

                    sum_all_tokens += token_count
                    sum_all_chars += char_count

                    rel = os.path.relpath(json_path, CTGOV_DATASET_RAW_PATH)
                    writer.writerow([
                        TOKENIZER_NAME,
                        nctid,
                        rel,
                        os.path.abspath(json_path),
                        char_count,
                        token_count,
                    ])
                    rows_written += 1
                except Exception:
                    # Skip bad/malformed files quietly
                    continue

        # Basic integrity checks
        self.assertTrue(os.path.exists(self.output_csv), "CSV file was not created.")
        with open(self.output_csv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        self.assertEqual(
            header,
            ["model_name", "nctid", "file", "abs_path", "char_count", "token_count"],
            "CSV header mismatch.",
        )
        self.assertEqual(rows_written, len(self.nctids))

        print(f"A total of {sum_all_tokens} tokens across {sum_all_chars} characters in {rows_written} files.")


class CTGovProtocolPDFTokensEstimationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Read NCTIDs
        with open(CTGOV_NCTIDS_LIST_ALL_PATH, "r", encoding="utf-8") as f:
            cls.nctids_all = [line.strip() for line in f if line.strip()]
        if not cls.nctids_all:
            raise unittest.SkipTest("NCTID list is empty.")

        cls.total_number_of_cts_with_available_protocol_pdfs = 43600
        cls.nctid_to_downloaded_pdf_paths_list_dict = {}
        for dirpath, _, filenames in os.walk(CTGOV_DATASET_EXTENSIONS_PATH):
            for filename in filenames:
                if filename.lower().endswith(".pdf"):
                    nctid = dirpath.split("/")[-1]
                    if nctid not in cls.nctid_to_downloaded_pdf_paths_list_dict:
                        cls.nctid_to_downloaded_pdf_paths_list_dict[nctid] = []
                    cls.nctid_to_downloaded_pdf_paths_list_dict[nctid].append(os.path.join(dirpath, filename))

        nctids_with_pdfs = list(cls.nctid_to_downloaded_pdf_paths_list_dict.keys())
        random.shuffle(nctids_with_pdfs)
        cls.num_samples = 100
        cls.pdf_samples = []
        for nctid in nctids_with_pdfs[:cls.num_samples]:
            cls.pdf_samples.extend(cls.nctid_to_downloaded_pdf_paths_list_dict[nctid])

        cls.ratio = cls.num_samples / cls.total_number_of_cts_with_available_protocol_pdfs

        # Quiet HF warnings like "sequence length > model max"
        hf_logging.set_verbosity_error()
        # Load tokenizer (cache-only if requested)
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_NAME, local_files_only=HF_LOCAL_FILES_ONLY
            )
            # Some tokenizers (e.g., GPT-2) lack a pad token; set to EOS to avoid warnings.
            if cls.tokenizer.pad_token is None and cls.tokenizer.eos_token is not None:
                cls.tokenizer.pad_token = cls.tokenizer.eos_token
            # We only count tokens; prevent "length > 512" warnings.
            cls.tokenizer.model_max_length = 10 ** 12
        except Exception as e:
            raise unittest.SkipTest(f"Tokenizer '{TOKENIZER_NAME}' not available: {e}")

    def test_estimate_protocol_pdf_tokens(self):
        total_tokens_seen = 0
        for pdf_path in tqdm.tqdm(self.pdf_samples, desc="Tokenizing PDFs .."):
            text = extract_text_from_pdf_using_pymupdf(pdf_path)
            token_count = len(self.tokenizer.encode(text, add_special_tokens=False, truncation=False))
            total_tokens_seen += token_count

        print("Total tokens seen in {} studies: {}".format(self.num_samples, total_tokens_seen))
        print("Estimated total tokens in all {} studies with protocol PDFs: {:.2f}".format(
            self.total_number_of_cts_with_available_protocol_pdfs,
            total_tokens_seen / self.ratio)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
