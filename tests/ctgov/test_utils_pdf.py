from aidose.ctgov.utils_pdf import extract_text_from_pdf, extract_images_from_pdf, ExtractedImage
from aidose.ctgov.utils_pdf import DeepSeekOCRExtractor
from aidose.ctgov.structures import Study
from aidose.ctgov.utils_protocol import get_large_protocols_pdf_links
from aidose.ctgov.utils_download import get_study_path_by_nctid_and_raw_dir
from aidose.ctgov import CTGOV_DATASET_RAW_PATH

import unittest
from typing import List
from PIL import Image
import tempfile
import urllib.request
import os
import shutil


class PDFUtilsTest(unittest.TestCase):
    """
    Test cases for PDF text and image extraction utilities.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Example NCT ID with known protocol and SAP PDFs
        nctid = "NCT07138599"
        study_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
        with open(study_path, "r", encoding="utf-8") as f:
            study = Study.model_validate_json(f.read())

        cls.protocol_pdf_links = get_large_protocols_pdf_links(study)
        cls.tmpdir = tempfile.TemporaryDirectory(prefix="pdf_dl_")

        cls.pdf_paths = []
        for link in cls.protocol_pdf_links:
            pdf_name = link.split("/")[-1].split(".")[0]
            pdf_path = os.path.join(cls.tmpdir.name, pdf_name)
            with urllib.request.urlopen(link) as resp:
                with open(pdf_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
                    cls.pdf_paths.append(pdf_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmpdir is not None:
            cls.tmpdir.cleanup()

    def test(self):
        print(self.pdf_paths)

    def test_extract_text_from_protocol_pdf(self):
        for pdf_path in self.pdf_paths:
            text = extract_text_from_pdf(pdf_path)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 1000)

    def test_extract_images_from_protocol_pdf(self):

        extracted_images = extract_images_from_pdf(self.pdf_paths[0])
        self.assertIsInstance(extracted_images, list)
        self.assertIsInstance(extracted_images[0], ExtractedImage)
        self.assertIsInstance(extracted_images[0].image, Image.Image)
        self.assertGreater(extracted_images[0].image.width, 63)
        self.assertGreater(extracted_images[0].image.height, 63)


class DeepSeekOCRExtractorIntegrationTest(unittest.TestCase):
    """
    GPU-enabled integration test for DeepSeekOCRExtractor.
    Runs actual inference on one small PDF to validate OCR pipeline.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Example NCT ID with known protocol and SAP PDFs
        nctid = "NCT07138599"
        study_path = get_study_path_by_nctid_and_raw_dir(nctid, CTGOV_DATASET_RAW_PATH)
        with open(study_path, "r", encoding="utf-8") as f:
            study = Study.model_validate_json(f.read())

        cls.protocol_pdf_links = get_large_protocols_pdf_links(study)
        cls.tmpdir = tempfile.TemporaryDirectory(prefix="pdf_dl_")

        cls.pdf_paths: List[str] = []
        for link in cls.protocol_pdf_links:
            pdf_name = os.path.basename(link).split(".")[0] + ".pdf"
            pdf_path = os.path.join(cls.tmpdir.name, pdf_name)
            with urllib.request.urlopen(link) as resp:
                with open(pdf_path, "wb") as out:
                    shutil.copyfileobj(resp, out)
                    cls.pdf_paths.append(pdf_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmpdir is not None:
            cls.tmpdir.cleanup()

    def test_extract_text_from_pdf_gpu(self) -> None:
        """Run full DeepSeek-OCR inference on GPU and validate markdown output."""
        pdf_path = self.pdf_paths[0]
        ocr = DeepSeekOCRExtractor()

        text = ocr.extract_text_from_pdf(pdf_path)
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 100, "Extracted text is unexpectedly short.")
        self.assertIn("#", text, "Markdown headings missing — output may be corrupted.")
        print("\n--- OCR Extracted Sample ---\n", text[:1000])

    def test_prompt_constant(self):
        """Ensure class prompt structure matches expected form."""
        self.assertTrue(DeepSeekOCRExtractor.PROMPT.startswith("<image>"))
        self.assertIn("Convert the document to markdown", DeepSeekOCRExtractor.PROMPT)


if __name__ == '__main__':
    unittest.main()
