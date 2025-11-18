from aidose import END_POINT_HF_DATASET_PATH, RESOURCES_DIR

from datasets import load_from_disk, DatasetDict, concatenate_datasets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import unittest
import json
from pathlib import Path
from typing import Dict, List
import os


class TestExtractedDocsAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out_dir = os.path.join(RESOURCES_DIR, "EDA-TEST", "CT-DOSING-ERRORS")
        cls.dataset_dict: DatasetDict = load_from_disk(END_POINT_HF_DATASET_PATH)

    def test(self):

        dataset = concatenate_datasets([
            self.dataset_dict["train"],
            self.dataset_dict["validation"],
            self.dataset_dict["test"]]

        )
        pdf_lengths: List[int] = []
        num_has_documents = 0
        len_sample_doc_with_176_pages = None
        for item in tqdm.tqdm(dataset):


            extracted_doc = item['FEATURE_protocolPdfText']
            has_document = item["METADATA_hasProtocol"] or item["METADATA_hasSap"] or item["METADATA_hasIcf"]
            if has_document:
                num_has_documents += 1

            if extracted_doc is not None:
                pdf_lengths.append(len(extracted_doc))

            nctid = item["METADATA_nctId"]
            if nctid == 'NCT00075803':
                len_sample_doc_with_176_pages = len(extracted_doc)


        print("Number of all CT's in the dataset: ", len(dataset))
        print("Number of CT's with documents: ", num_has_documents)
        print("The number of extracted docs: ", len(pdf_lengths))
        print("Average char length of extracted docs: ", np.mean(np.array(pdf_lengths)))
        print("Average page number: ", 176 * (np.mean(np.array(pdf_lengths)) / len_sample_doc_with_176_pages))