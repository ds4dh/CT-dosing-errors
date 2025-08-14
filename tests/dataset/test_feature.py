import unittest
from enum import Enum
from typing import List

from aidose.dataset.feature import Feature


class Phase(Enum):
    P1 = "Phase 1"
    P2 = "Phase 2"
    P3 = "Phase 3"


class FeatureUnitTest(unittest.TestCase):
    def test_basic_ok(self):
        f = Feature(name="title", value="abc", declared_type=str)
        self.assertEqual(f.name, "title")
        self.assertEqual(f.value, "abc")
        self.assertIs(f.declared_type, str)
        cell = f.to_cell()
        self.assertEqual(cell["value"], "abc")
        self.assertIs(cell["type"], str)

    def test_none_allowed(self):
        f = Feature(name="enrollmentCount", value=None, declared_type=int)
        self.assertIsNone(f.value)
        self.assertIs(f.declared_type, int)
        cell = f.to_cell()
        self.assertIsNone(cell["value"])
        self.assertIs(cell["type"], int)

    def test_type_mismatch(self):
        with self.assertRaises(TypeError):
            Feature(name="count", value="not-int", declared_type=int)

    def test_type_argument_must_be_type(self):
        with self.assertRaises(TypeError):
            Feature(name="value", value="abc", declared_type="not_a_type")  # type: ignore[arg-type]

    def test_bool_strictness(self):
        Feature(name="flag", value=True, declared_type=bool)  # ok
        with self.assertRaises(TypeError):
            Feature(name="flag", value=1, declared_type=bool)

    def test_enum_basic(self):
        f = Feature(name="phase_raw", value=Phase.P2, declared_type=Phase)
        self.assertEqual(f.value, Phase.P2)
        self.assertIs(f.declared_type, Phase)

    def test_enum_with_base_declared_type(self):
        with self.assertRaises(TypeError):
            Feature(name="phase_raw", value=Phase.P3, declared_type=Enum)

    # ---------- one-hot (single Enum) ----------

    def test_enum_one_hot_features(self):
        f = Feature(name="phase_raw", value=Phase.P2, declared_type=Phase)
        feats: List[Feature] = f.as_one_hot()

        self.assertEqual([feat.name for feat in feats], ["phase_raw.P1", "phase_raw.P2", "phase_raw.P3"])
        self.assertTrue(all(feat.declared_type is bool for feat in feats))
        self.assertEqual([feat.value for feat in feats], [False, True, False])

        cells = [feat.to_cell() for feat in feats]
        self.assertTrue(all(cell["type"] is bool for cell in cells))
        self.assertEqual([cell["value"] for cell in cells], [False, True, False])

    def test_enum_one_hot_none_value(self):
        f = Feature(name="phase_raw", value=None, declared_type=Phase)
        feats = f.as_one_hot()

        self.assertEqual([feat.name for feat in feats], ["phase_raw.P1", "phase_raw.P2", "phase_raw.P3"])
        self.assertTrue(all(feat.declared_type is bool for feat in feats))
        self.assertEqual([feat.value for feat in feats], [None, None, None])

    def test_as_one_hot_rejects_list_value(self):
        f = Feature(name="phase_list", value=[Phase.P1, Phase.P2], declared_type=Phase)
        with self.assertRaises(TypeError):
            f.as_one_hot()

    # ---------- multi-hot (single enum or list of enums) ----------

    def test_multi_hot_single_enum(self):
        f = Feature(name="phase_list", value=Phase.P1, declared_type=Phase)
        feats = f.as_multi_hot()

        self.assertEqual([feat.name for feat in feats], ["phase_list.P1", "phase_list.P2", "phase_list.P3"])
        self.assertTrue(all(feat.declared_type is int for feat in feats))
        self.assertEqual([feat.value for feat in feats], [1, 0, 0])

    def test_multi_hot_none_value(self):
        f = Feature(name="phase_list", value=None, declared_type=Phase)
        feats = f.as_multi_hot()

        self.assertEqual([feat.name for feat in feats], ["phase_list.P1", "phase_list.P2", "phase_list.P3"])
        self.assertTrue(all(feat.declared_type is int for feat in feats))
        self.assertEqual([feat.value for feat in feats], [None, None, None])

    def test_multi_hot_list_values(self):
        f = Feature(name="phase_list", value=[Phase.P1, Phase.P3], declared_type=Phase)
        feats = f.as_multi_hot()

        self.assertEqual([feat.name for feat in feats], ["phase_list.P1", "phase_list.P2", "phase_list.P3"])
        self.assertEqual([feat.value for feat in feats], [1, 0, 1])

    def test_multi_hot_list_with_duplicates(self):
        f = Feature(name="phase_list", value=[Phase.P2, Phase.P2, Phase.P3], declared_type=Phase)
        feats = f.as_multi_hot()

        self.assertEqual([feat.value for feat in feats], [0, 2, 1])

    def test_enum_list_type_validation_wrong_member(self):
        class Other(Enum):
            X = "x"

        with self.assertRaises(TypeError):
            Feature(name="bad_list", value=[Phase.P1, Other.X], declared_type=Phase)  # mixed enum types not allowed


if __name__ == "__main__":
    unittest.main()
