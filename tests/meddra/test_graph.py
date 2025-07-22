from aidose.meddra.graph import MedDRALevel
from aidose.meddra.graph import Node
from aidose.meddra.graph import MedDRA

import unittest

from typing import Set


class MedDRALevelUnitTest(unittest.TestCase):

    def test_ordered_levels(self):
        expected = [MedDRALevel.SOC, MedDRALevel.HLGT, MedDRALevel.HLT, MedDRALevel.PT, MedDRALevel.LLT]
        self.assertEqual(MedDRALevel.ordered_levels(), expected)

    def test_parent(self):
        self.assertEqual(MedDRALevel.HLGT.parent(), MedDRALevel.SOC)
        self.assertEqual(MedDRALevel.SOC.parent(), MedDRALevel.SOC)  # top returns self

    def test_child(self):
        self.assertEqual(MedDRALevel.HLT.child(), MedDRALevel.PT)
        self.assertEqual(MedDRALevel.LLT.child(), MedDRALevel.LLT)  # bottom returns self

    def test_is_above(self):
        self.assertTrue(MedDRALevel.HLGT.is_above(MedDRALevel.PT))
        self.assertFalse(MedDRALevel.LLT.is_above(MedDRALevel.HLT))

    def test_is_below(self):
        self.assertTrue(MedDRALevel.PT.is_below(MedDRALevel.HLGT))
        self.assertFalse(MedDRALevel.SOC.is_below(MedDRALevel.LLT))

    def test_from_str(self):
        self.assertEqual(MedDRALevel.from_str("llt"), MedDRALevel.LLT)
        self.assertEqual(MedDRALevel.from_str("HLGT"), MedDRALevel.HLGT)
        with self.assertRaises(ValueError):
            MedDRALevel.from_str("NotALevel")

    def test_str_output(self):
        self.assertEqual(str(MedDRALevel.PT), "PT")
        self.assertEqual(str(MedDRALevel.LLT), "LLT")


class NodeUnitTest(unittest.TestCase):

    def test_node_initialization(self):
        node = Node(code="123", term="Chest pain", level=MedDRALevel.PT)
        self.assertEqual(node.code, "123")
        self.assertEqual(node.term, "Chest pain")
        self.assertEqual(node.level, MedDRALevel.PT)
        self.assertEqual(len(node.parents), 0)

    def test_node_equality(self):
        node1 = Node(code="123", term="Chest pain", level=MedDRALevel.PT)
        node2 = Node(code="123", term="Chest pain", level=MedDRALevel.PT)
        self.assertEqual(node1, node2)

    def test_add_parents(self):
        parent = Node(code="456", term="Angina", level=MedDRALevel.HLT)
        child = Node(code="123", term="Chest pain", level=MedDRALevel.PT)
        child.parents.add(parent)
        self.assertIn(parent, child.parents)
        self.assertEqual(len(child.parents), 1)


class MedDRATestCase(unittest.TestCase):

    def setUp(self):
        self.graph = MedDRA()

        # Manually create a full SOC ➝ HLGT ➝ HLT ➝ PT ➝ LLT chain
        self.graph.add_node("001", "System", MedDRALevel.SOC)
        self.graph.add_node("002", "Group", MedDRALevel.HLGT)
        self.graph.add_node("003", "Term", MedDRALevel.HLT)
        self.graph.add_node("004", "Preferred", MedDRALevel.PT)
        self.graph.add_node("005", "Lowest", MedDRALevel.LLT)

        self.graph.add_edge("001", MedDRALevel.SOC, "002", MedDRALevel.HLGT)
        self.graph.add_edge("002", MedDRALevel.HLGT, "003", MedDRALevel.HLT)
        self.graph.add_edge("003", MedDRALevel.HLT, "004", MedDRALevel.PT)
        self.graph.add_edge("004", MedDRALevel.PT, "005", MedDRALevel.LLT)

    def test_nodes_exist(self):
        self.assertEqual(len(self.graph.nodes), 5)
        self.assertIn((MedDRALevel.SOC, "001"), self.graph.nodes)
        self.assertIn((MedDRALevel.LLT, "005"), self.graph.nodes)

    def test_parent_links(self):
        llt = self.graph.nodes[(MedDRALevel.LLT, "005")]
        pt = self.graph.nodes[(MedDRALevel.PT, "004")]
        self.assertIn(pt, llt.parents)

    def test_find_paths_no_padding(self):
        paths = self.graph.find_paths("005", MedDRALevel.LLT, pad_levels=False)
        self.assertIn(
            ["001@SOC", "002@HLGT", "003@HLT", "004@PT", "005@LLT"], paths
        )

    def test_find_paths_with_padding(self):
        # Remove HLT ➝ PT to create a gap that needs BYPASS@PT
        self.graph.nodes[(MedDRALevel.PT, "004")].parents.clear()
        self.graph.add_edge("002", MedDRALevel.HLGT, "004", MedDRALevel.PT)

        paths = self.graph.find_paths("005", MedDRALevel.LLT, pad_levels=True)
        found_bypass = any("BYPASS@HLT" in step for path in paths for step in path)
        self.assertTrue(found_bypass)

    def test_find_node_by_term_basic(self):
        results = self.graph.find_node_by_term("Preferred")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].code, "004")
        self.assertEqual(results[0].level, MedDRALevel.PT)

    def test_find_node_by_term_with_level_filter(self):
        levels: Set[MedDRALevel] = {MedDRALevel.LLT}
        results = self.graph.find_node_by_term("Lowest", levels=levels)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].level, MedDRALevel.LLT)

    def test_find_node_by_term_with_preprocess(self):
        preprocess = lambda s: s.lower()
        results = self.graph.find_node_by_term("preferred", preprocess=preprocess)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].code, "004")


if __name__ == "__main__":
    unittest.main()
