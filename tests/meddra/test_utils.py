import unittest
from aidose.meddra.utils import (
    get_all_ancestors,
    get_descendant_terms,
    clean_paths,
    convert_paths_to_tuples,
    is_full_path,
    has_complete_path,
    get_descendant_info,
)
from aidose.meddra.graph import MedDRA, Node, MedDRALevel


class MedDRAUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.meddra = MedDRA()

        # Create a small test graph
        self.meddra.add_node("100", "System Organ Class", MedDRALevel.SOC)
        self.meddra.add_node("200", "High-Level Group Term", MedDRALevel.HLGT)
        self.meddra.add_node("300", "High-Level Term", MedDRALevel.HLT)
        self.meddra.add_node("400", "Preferred Term", MedDRALevel.PT)
        self.meddra.add_node("500", "Lowest Level Term", MedDRALevel.LLT)

        self.meddra.add_edge("100", MedDRALevel.SOC, "200", MedDRALevel.HLGT)
        self.meddra.add_edge("200", MedDRALevel.HLGT, "300", MedDRALevel.HLT)
        self.meddra.add_edge("300", MedDRALevel.HLT, "400", MedDRALevel.PT)
        self.meddra.add_edge("400", MedDRALevel.PT, "500", MedDRALevel.LLT)

    def test_get_all_ancestors(self):
        node = self.meddra.nodes[(MedDRALevel.LLT, "500")]
        ancestors = get_all_ancestors(node)
        ancestor_codes = {n.code for n in ancestors}
        self.assertEqual(ancestor_codes, {"100", "200", "300", "400"})

    def test_get_descendant_terms(self):
        hlgt_node = self.meddra.nodes[(MedDRALevel.HLGT, "200")]
        descendants = get_descendant_terms(hlgt_node, self.meddra)
        self.assertEqual(descendants, {"Preferred Term", "Lowest Level Term"})

    def test_clean_paths(self):
        paths = [["100@SOC", "BYPASS@HLGT", "300@HLT"], ["200@HLGT", "300@HLT"]]
        cleaned = clean_paths(paths)
        self.assertEqual(cleaned, [["100@SOC", "300@HLT"], ["200@HLGT", "300@HLT"]])

    def test_convert_paths_to_tuples(self):
        paths = [["100@SOC", "200@HLGT", "300@HLT"]]
        result = convert_paths_to_tuples(paths, self.meddra)
        self.assertEqual(result[0][0][1], "System Organ Class")
        self.assertEqual(result[0][1][1], "High-Level Group Term")
        self.assertEqual(result[0][2][1], "High-Level Term")

    def test_is_full_path(self):
        path = [
            ("100@SOC", "System Organ Class"),
            ("200@HLGT", "High-Level Group Term"),
            ("300@HLT", "High-Level Term"),
            ("400@PT", "Preferred Term"),
            ("500@LLT", "Lowest Level Term"),
        ]
        expected = [
            MedDRALevel.SOC,
            MedDRALevel.HLGT,
            MedDRALevel.HLT,
            MedDRALevel.PT,
            MedDRALevel.LLT,
        ]
        self.assertTrue(is_full_path(path, expected))

    def test_has_complete_path(self):
        paths = [
            [
                ("100@SOC", "System Organ Class"),
                ("200@HLGT", "High-Level Group Term"),
                ("300@HLT", "High-Level Term"),
                ("400@PT", "Preferred Term"),
                ("500@LLT", "Lowest Level Term"),
            ]
        ]
        expected = [
            MedDRALevel.SOC,
            MedDRALevel.HLGT,
            MedDRALevel.HLT,
            MedDRALevel.PT,
            MedDRALevel.LLT,
        ]
        self.assertTrue(has_complete_path(paths, expected))

    def test_get_descendant_info(self):
        hlgt_node = self.meddra.nodes[(MedDRALevel.HLGT, "200")]
        info = get_descendant_info(hlgt_node, self.meddra)
        self.assertIn("400@PT", info)
        self.assertIn("500@LLT", info)
        self.assertEqual(info["400@PT"]["term"], "Preferred Term")
        self.assertTrue(any("100@SOC" in step[0] for step in info["500@LLT"]["paths"][0]))


if __name__ == "__main__":
    unittest.main()
