import unittest
from enum import Enum
from typing import Any, Dict

from aidose.dataset.feature import Feature


class Phase(Enum):
    P1 = "Phase 1"
    P2 = "Phase 2"
    P3 = "Phase 3"


class FeatureUnitTest(unittest.TestCase):
    def test_basic_ok(self):
        f = Feature("abc", str)
        self.assertEqual(f.value, "abc")
        self.assertIs(f.declared_type, str)
        d = f.to_dict()
        self.assertEqual(d["value"], "abc")
        self.assertIs(d["type"], str)

    def test_none_allowed(self):
        f = Feature(None, int)
        self.assertIsNone(f.value)
        self.assertIs(f.declared_type, int)
        # add_to_row should carry None value with type int
        row: Dict[str, Any] = {}
        f.add_to_row(row, "enrollmentCount")
        self.assertIn("enrollmentCount", row)
        self.assertIsNone(row["enrollmentCount"]["value"])
        self.assertIs(row["enrollmentCount"]["type"], int)

    def test_type_mismatch(self):
        with self.assertRaises(TypeError):
            Feature("not-int", int)

    def test_type_argument_must_be_type(self):
        with self.assertRaises(TypeError):
            Feature("value", "not_a_type")  # type: ignore[arg-type]

    def test_bool_strictness(self):
        Feature(True, bool)  # ok
        with self.assertRaises(TypeError):
            Feature(1, bool)  # not allowed (int is not bool)

    def test_enum_basic(self):
        f = Feature(Phase.P2, Phase)
        self.assertEqual(f.value, Phase.P2)
        self.assertIs(f.declared_type, Phase)

    def test_enum_with_base_declared_type(self):
        # Previously constructed successfully; now it must fail because declared_type cannot be Enum
        with self.assertRaises(TypeError):
            Feature(Phase.P3, Enum)

    def test_enum_one_hot_to_dict(self):
        f = Feature(Phase.P2, Phase)
        hot = f.to_one_hot_entries("phase")
        names = list(hot.keys())
        self.assertEqual(names, ["phase_P1", "phase_P2", "phase_P3"])
        values = [hot[k]["value"] for k in names]
        self.assertEqual(values, [False, True, False])
        self.assertTrue(all(h["type"] is bool for h in hot.values()))

    def test_enum_one_hot_none_value(self):
        f = Feature(None, Phase)
        hot = f.to_one_hot_entries("phase")
        names = list(hot.keys())
        self.assertEqual(names, ["phase_P1", "phase_P2", "phase_P3"])
        values = [hot[k]["value"] for k in names]
        self.assertEqual(values, [None, None, None])
        self.assertTrue(all(h["type"] is bool for h in hot.values()))

    def test_add_to_row_as_one_hot(self):
        row: Dict[str, Dict] = {}
        Feature(Phase.P3, Phase).add_to_row_as_one_hot(row, "phase")
        self.assertIn("phase_P1", row)
        self.assertIn("phase_P2", row)
        self.assertIn("phase_P3", row)
        self.assertEqual(row["phase_P1"]["value"], False)
        self.assertEqual(row["phase_P2"]["value"], False)
        self.assertEqual(row["phase_P3"]["value"], True)
        self.assertIs(row["phase_P3"]["type"], bool)

    def test_one_hot_requires_enum_type(self):
        f = Feature("abc", str)
        with self.assertRaises(TypeError):
            f.to_one_hot_entries("name")
        with self.assertRaises(TypeError):
            f.add_to_row_as_one_hot({}, "name")

    def test_one_hot_none_requires_enum_subclass_declared_type(self):
        # Construction fails when declared_type is Enum
        with self.assertRaises(TypeError):
            Feature(None, Enum)


if __name__ == "__main__":
    unittest.main()
