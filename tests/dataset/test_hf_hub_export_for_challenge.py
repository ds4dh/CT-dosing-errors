import unittest
import os
from datasets import load_dataset, ClassLabel, Value
from huggingface_hub import HfApi
from aidose import END_POINT_HF_DATASET_PATH

# Configuration
HF_HUB_REPO_ID = "sssohrab/ct-dosing-errors-benchmark"
HF_TOKEN = os.environ.get("HF_TOKEN")


class TestHFDatasetSchema(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        1. Loads the local PRIVATE dataset (Source of Truth).
        2. Fetches metadata directly from the Hugging Face Hub API (Target).
        """
        print(f"\n[Test Setup] Loading private dataset from: {END_POINT_HF_DATASET_PATH}")
        cls.private_ds = load_dataset(END_POINT_HF_DATASET_PATH, split="train")
        cls.private_features = cls.private_ds.features

        print(f"[Test Setup] Fetching LIVE metadata from Hub API: {HF_HUB_REPO_ID}")
        api = HfApi(token=HF_TOKEN)

        # This gets the raw metadata from the server, bypassing local dataset cache
        cls.hub_info = api.dataset_info(HF_HUB_REPO_ID)

        # We also need the features to verify schema.
        # We can still use the builder for features, or rely on the dataset info if available.
        # Ideally, we check the 'cardData' (YAML in README) for license/tags
        # and the 'files' to ensure structure.

        # For feature comparison, we still need the dataset structure.
        # We use force_redownload to ensure we aren't reading stale cache.
        from datasets import load_dataset_builder
        cls.public_builder = load_dataset_builder(
            HF_HUB_REPO_ID,
            token=HF_TOKEN,
            download_mode="force_redownload"  # CRITICAL FIX
        )
        cls.public_features = cls.public_builder.info.features

    def test_01_feature_presence_and_renaming(self):
        """Verifies fields are renamed/dropped correctly."""
        expected_public_keys = set()

        for key in self.private_features.keys():
            if key.startswith("FEATURE_"):
                expected_key = key.replace("FEATURE_", "")
                expected_public_keys.add(expected_key)
                self.assertIn(expected_key, self.public_features, f"Missing: {expected_key}")
            elif key == "LABEL_wilson_label":
                expected_public_keys.add("target")
                self.assertIn("target", self.public_features)
            elif key == "METADATA_nctId":
                expected_public_keys.add("nctid")
                self.assertIn("nctid", self.public_features)
            else:
                self.assertNotIn(key, self.public_features, f"Leaked private column: {key}")

        actual_public_keys = set(self.public_features.keys())
        extra_keys = actual_public_keys - expected_public_keys
        self.assertEqual(len(extra_keys), 0, f"Unexpected extra columns: {extra_keys}")

    def test_02_feature_types_and_classlabels(self):
        """Verifies types and ClassLabels match."""
        key_mapping = {}
        for key in self.private_features.keys():
            if key.startswith("FEATURE_"):
                key_mapping[key] = key.replace("FEATURE_", "")
            elif key == "LABEL_wilson_label":
                key_mapping[key] = "target"
            elif key == "METADATA_nctId":
                key_mapping[key] = "nctid"

        for private_key, public_key in key_mapping.items():
            priv_feat = self.private_features[private_key]
            pub_feat = self.public_features[public_key]

            if isinstance(priv_feat, ClassLabel):
                self.assertIsInstance(pub_feat, ClassLabel)
                self.assertEqual(priv_feat.names, pub_feat.names, f"ClassLabel mismatch: {public_key}")
            elif isinstance(priv_feat, Value):
                self.assertIsInstance(pub_feat, Value)
                # Allow int64->int32 (parquet optimization), but ensure types match generally
                self.assertIn(pub_feat.dtype, [priv_feat.dtype, 'int32', 'int64'])

    def test_03_metadata(self):
        """
        Check high-level metadata using HfApi info (The Source of Truth).
        """
        # 1. Check License in Card Data (YAML header in README.md)
        card_data = self.hub_info.cardData
        self.assertIsNotNone(card_data, "Dataset Card (README.md) has no YAML metadata")

        # Verify License
        self.assertEqual(card_data.get('license'), "cc-by-4.0",
                         f"License mismatch in Card. Found: {card_data.get('license')}")

        # Verify Tags
        tags = card_data.get('tags', [])
        self.assertIn("clinical-trials", tags, "Missing 'clinical-trials' tag")
        self.assertIn("medication-safety", tags, "Missing 'medication-safety' tag")

        print("\nMetadata Verification Successful:")
        print(f"- License: {card_data.get('license')}")
        print(f"- Tags: {tags}")


if __name__ == "__main__":
    unittest.main()