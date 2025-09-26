from aidose.ctgov.utils_pdf import extract_images_from_pdf
from aidose.ctgov.structures import Study
from aidose.ctgov.utils_protocol import get_large_protocols_pdf_links
from aidose.ctgov.utils_download import get_study_path_by_nctid_and_raw_dir
from aidose.ctgov import CTGOV_DATASET_RAW_PATH
from aidose.dataset import CTGOV_NCTIDS_LIST_FILTERED_PATH
from aidose.ctgov import CTGOV_DATASET_EXTENSIONS_PATH

import unittest
from typing import List
import urllib.request
import tqdm
import os
import shutil


class PDFAvailabilityStatsTest(unittest.TestCase):
    def setUp(self):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            self.skipTest("Required CTGov dataset files are missing. Please run the dataset API script first.")

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'r') as f:
            self.nctids_list_included = [line.strip() for line in f if line.strip()]

        self.pdf_links_list: List[List[str]] = []
        for nctid in tqdm.tqdm(self.nctids_list_included, desc="Collecting PDF links from  studies .."):
            study_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
            with open(study_path, "r", encoding="utf-8") as f:
                study = Study.model_validate_json(f.read())
            pdf_links = get_large_protocols_pdf_links(study, check_link_status=False)
            if pdf_links:
                self.pdf_links_list.append(pdf_links)

    def test_pdf_links_availability_stats(self):
        num_total_studies_included = len(self.nctids_list_included)
        num_studies_with_pdfs = len(self.pdf_links_list)
        num_studies_without_pdfs = num_total_studies_included - num_studies_with_pdfs
        pdf_availability_percentage = (num_studies_with_pdfs / num_total_studies_included) * 100
        average_pdf_per_study = (sum(len(links) for links in self.pdf_links_list) / num_studies_with_pdfs)

        print(f"Total studies analyzed: {num_total_studies_included}")
        print(f"Studies with protocol PDFs: {num_studies_with_pdfs}")
        print(f"Studies without protocol PDFs: {num_studies_without_pdfs}")
        print(f"PDF availability percentage: {pdf_availability_percentage:.2f}%")
        print("Average PDFs per study (for studies with PDFs): {:.2f}".format(average_pdf_per_study))

        self.assertGreater(num_total_studies_included, 0, "No studies found in the filtered list.")
        self.assertGreater(num_studies_with_pdfs, 0, "No studies with protocol/SAP PDFs found.")


class PDFDownloadTest(unittest.TestCase):
    def setUp(self):
        if not (os.path.exists(CTGOV_NCTIDS_LIST_FILTERED_PATH) and os.path.exists(CTGOV_DATASET_RAW_PATH)):
            self.skipTest("Required CTGov dataset files are missing. Please run the dataset API script first.")

        with open(CTGOV_NCTIDS_LIST_FILTERED_PATH, 'r') as f:
            self.nctids_list_included = [line.strip() for line in f if line.strip()]

        self.pdf_links_list: List[List[str]] = []
        for nctid in tqdm.tqdm(self.nctids_list_included, desc="Collecting PDF links from  studies .."):
            study_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
            with open(study_path, "r", encoding="utf-8") as f:
                study = Study.model_validate_json(f.read())
            pdf_links = get_large_protocols_pdf_links(study, check_link_status=False)
            if not pdf_links:
                continue

            parent_identifier = nctid[-2:]
            for link in pdf_links:
                pdf_name = link.split("/")[-1]
                pdf_save_path = os.path.join(CTGOV_DATASET_EXTENSIONS_PATH, "protocol-pdfs",
                                             parent_identifier, nctid, pdf_name)
                os.makedirs(os.path.dirname(pdf_save_path), exist_ok=True)
                if not os.path.exists(pdf_save_path):
                    with urllib.request.urlopen(link) as resp:
                        with open(pdf_save_path, "wb") as out:
                            shutil.copyfileobj(resp, out)
                extracted_images = extract_images_from_pdf(pdf_save_path)
                if not extracted_images:
                    continue

                for i, image in enumerate(extracted_images):
                    image_save_path = os.path.join(CTGOV_DATASET_EXTENSIONS_PATH, "protocol-pdfs",
                                                   parent_identifier, nctid, "images",
                                                   "{}.{}".format(str(i).zfill(2), image.ext))
                    os.makedirs(os.path.dirname(image_save_path), exist_ok=True)
                    if not os.path.exists(image_save_path):
                        image.image.save(image_save_path)

    def test(self):
        self.assertTrue(True, "PDF download and image extraction completed without exceptions.")




if __name__ == '__main__':
    unittest.main()
