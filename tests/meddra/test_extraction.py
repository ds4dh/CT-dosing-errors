import unittest

from aidose.meddra.graph import MedDRA, MedDRALevel
from aidose.meddra.extraction import MedDRADescendantResult, build_meddra_descendants

from dataclasses import FrozenInstanceError


class MedDRADescendantResultTest(unittest.TestCase):
    def test_dataclass_is_frozen_and_holds_values(self):
        # Construct a minimal instance
        result = MedDRADescendantResult(
            terms={"Preferred Term", "Lowest Level Term"},
            paths={
                "200@HLGT": {
                    "400@PT": {"term": "Preferred Term", "paths": [[(MedDRALevel.SOC, "100")]]},
                    "500@LLT": {"term": "Lowest Level Term", "paths": [[(MedDRALevel.SOC, "100")]]},
                }
            },
            total_descendants=2,
            no_complete_path_count=0,
        )

        # Values are held as given
        self.assertIn("Preferred Term", result.terms)
        self.assertIn("200@HLGT", result.paths)
        self.assertEqual(result.total_descendants, 2)
        self.assertEqual(result.no_complete_path_count, 0)

        # Frozen: trying to assign should raise
        with self.assertRaises(FrozenInstanceError):
            setattr(result, "total_descendants", 3)  # triggers dataclasses' __setattr__


class BuildMedDRADescendantsTest(unittest.TestCase):
    def setUp(self):
        # Build a small, complete chain SOC -> HLGT -> HLT -> PT -> LLT
        self.meddra = MedDRA()
        # Nodes
        self.meddra.add_node("100", "System Organ Class", MedDRALevel.SOC)
        self.meddra.add_node("200", "High-Level Group Term", MedDRALevel.HLGT)
        self.meddra.add_node("300", "High-Level Term", MedDRALevel.HLT)
        self.meddra.add_node("400", "Preferred Term", MedDRALevel.PT)
        self.meddra.add_node("500", "Lowest Level Term", MedDRALevel.LLT)
        # Edges (downward)
        self.meddra.add_edge("100", MedDRALevel.SOC, "200", MedDRALevel.HLGT)
        self.meddra.add_edge("200", MedDRALevel.HLGT, "300", MedDRALevel.HLT)
        self.meddra.add_edge("300", MedDRALevel.HLT, "400", MedDRALevel.PT)
        self.meddra.add_edge("400", MedDRALevel.PT, "500", MedDRALevel.LLT)

    def test_happy_path_single_hlgt(self):
        result = build_meddra_descendants(
            meddra=self.meddra,
            hlgt_codes=[(MedDRALevel.HLGT, "200")],
        )

        # The two descendants under HLGT are PT and LLT (terms)
        self.assertIn("Preferred Term", result.terms)
        self.assertIn("Lowest Level Term", result.terms)
        self.assertEqual(result.total_descendants, 2)
        self.assertEqual(result.no_complete_path_count, 0)

        # Paths map presence
        self.assertIn("200@HLGT", result.paths)
        hlgt_block = result.paths["200@HLGT"]
        self.assertIn("400@PT", hlgt_block)
        self.assertIn("500@LLT", hlgt_block)

        # Ensure each entry has a term and at least one path
        self.assertEqual(hlgt_block["400@PT"]["term"], "Preferred Term")
        self.assertTrue(hlgt_block["400@PT"]["paths"])
        self.assertEqual(hlgt_block["500@LLT"]["term"], "Lowest Level Term")
        self.assertTrue(hlgt_block["500@LLT"]["paths"])

    def test_incomplete_chain_counts_as_no_complete_path(self):
        # Add a second HLGT without a SOC ancestor to break expected chain completeness
        self.meddra.add_node("210", "HLGT-Incomplete", MedDRALevel.HLGT)
        self.meddra.add_node("310", "HLT-Incomplete", MedDRALevel.HLT)
        self.meddra.add_node("410", "PT-Incomplete", MedDRALevel.PT)
        # Missing SOC->HLGT link on purpose
        self.meddra.add_edge("210", MedDRALevel.HLGT, "310", MedDRALevel.HLT)
        self.meddra.add_edge("310", MedDRALevel.HLT, "410", MedDRALevel.PT)

        result = build_meddra_descendants(
            meddra=self.meddra,
            hlgt_codes=[(MedDRALevel.HLGT, "210")],
        )

        # Only PT descendant here
        self.assertIn("PT-Incomplete", result.terms)
        self.assertEqual(result.total_descendants, 1)

        # Because SOC is missing in the ancestor chain, path should be flagged incomplete
        self.assertEqual(result.no_complete_path_count, 1)

        # Paths structure still present
        self.assertIn("210@HLGT", result.paths)
        self.assertIn("410@PT", result.paths["210@HLGT"])
        self.assertEqual(result.paths["210@HLGT"]["410@PT"]["term"], "PT-Incomplete")
        self.assertTrue(isinstance(result.paths["210@HLGT"]["410@PT"]["paths"], list))

    def test_missing_hlgt_is_safely_ignored(self):
        # HLGT code that doesn't exist—should not raise, just yield empty result
        result = build_meddra_descendants(
            meddra=self.meddra,
            hlgt_codes=[(MedDRALevel.HLGT, "DOES_NOT_EXIST")],
        )
        self.assertEqual(result.terms, set())
        self.assertEqual(result.paths, {})
        self.assertEqual(result.total_descendants, 0)
        self.assertEqual(result.no_complete_path_count, 0)


if __name__ == "__main__":
    unittest.main()
