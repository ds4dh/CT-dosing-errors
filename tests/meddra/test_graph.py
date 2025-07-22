from aidose.meddra.graph import MedDRALevel
from aidose.meddra.graph import Node

import unittest


class MedDraLevelUnitTest(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
