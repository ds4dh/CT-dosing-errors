import unittest
from enum import Enum
from typing import List, Dict, Any

from aidose.dataset.feature import Feature, FeaturesList


class Phase(Enum):
    P1 = "Phase 1"
    P2 = "Phase 2"
    P3 = "Phase 3"


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


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


class FeaturesListUnitTest(unittest.TestCase):
    @staticmethod
    def _cells(feats: list[Feature]) -> Dict[str, Dict[str, Any]]:
        """Helper to view expanded features as name -> {'value', 'type'}"""
        return {f.name: f.to_cell() for f in feats}

    def test_non_enum_feature_passthrough(self):
        f = Feature(name="enrollmentCount", value=123, declared_type=int)
        feats = FeaturesList([f])
        expanded = feats.expand_enums()

        self.assertEqual(len(expanded), 1)
        self.assertIs(expanded[0], f)  # same object passthrough
        self.assertEqual(expanded[0].to_cell(), {"value": 123, "type": int})

    def test_mixed_features(self):
        base = Feature(name="enrollmentCount", value=100, declared_type=int)
        single_enum = Feature(name="phase", value=Phase.P1, declared_type=Phase)
        list_enum = Feature(name="phases", value=[Phase.P2, Phase.P3], declared_type=Phase)

        feats = FeaturesList([base, single_enum, list_enum])
        expanded = feats.expand_enums()

        # int stays 1 feature, single enum -> 3, list enum -> 3  ==> total 7
        self.assertEqual(len(expanded), 7)

        cells = self._cells(expanded)
        # int feature remains
        self.assertIn("enrollmentCount", {f.name for f in expanded})
        self.assertEqual(cells["enrollmentCount"]["value"], 100)
        self.assertIs(cells["enrollmentCount"]["type"], int)

        # one-hot for Phase
        self.assertEqual(cells["phase.P1"]["value"], True)
        self.assertEqual(cells["phases.P2"]["value"], 1)  # from multi-hot below it will be int
        self.assertTrue(cells["phase.P1"]["type"] in (bool, int))  # one-hot bool / multi-hot int coexist by name

        # multi-hot increments for the list
        p2_int_entries = [f for f in expanded if f.name == "phases.P2" and f.declared_type is int]
        p3_int_entries = [f for f in expanded if f.name == "phases.P3" and f.declared_type is int]
        self.assertTrue(any(e.value == 1 for e in p2_int_entries))
        self.assertTrue(any(e.value == 1 for e in p3_int_entries))

    def test_getters_on_plain_features(self):
        feats = FeaturesList([
            Feature("age", 42, int),
            Feature("title", "Study A", str),
            Feature("flag", True, bool),
        ])

        self.assertEqual(feats.get_names(), ["age", "title", "flag"])
        self.assertEqual(feats.get_values(), [42, "Study A", True])
        self.assertEqual(feats.get_types(), [int, str, bool])

    def test_expand_enums_one_hot_then_getters(self):
        feats = FeaturesList([
            Feature("color", Color.GREEN, Color),
            Feature("count", 7, int),
        ])
        expanded = feats.expand_enums()

        # Expect the enum feature to expand to three one-hot features; non-enum remains.
        expected_names = ["color.RED", "color.GREEN", "color.BLUE", "count"]
        self.assertEqual(expanded.get_names(), expected_names)

        # Values become booleans (True only for GREEN), plus the int feature.
        self.assertEqual(expanded.get_values(), [False, True, False, 7])

        # Types: bool for one-hot entries, then int for the remaining feature
        self.assertEqual(expanded.get_types()[:3], [bool, bool, bool])
        self.assertEqual(expanded.get_types()[3], int)

    def test_expand_enums_none_one_hot_then_getters(self):
        feats = FeaturesList([
            Feature("color", None, Color),  # missing enum -> all None one-hot
            Feature("flag", False, bool),
        ])
        expanded = feats.expand_enums()

        self.assertEqual(expanded.get_names(), ["color.RED", "color.GREEN", "color.BLUE", "flag"])
        self.assertEqual(expanded.get_values(), [None, None, None, False])
        self.assertEqual(expanded.get_types()[:3], [bool, bool, bool])
        self.assertEqual(expanded.get_types()[3], bool)

    def test_expand_enums_multi_hot_then_getters(self):
        feats = FeaturesList([
            Feature("colors", [Color.RED, Color.BLUE, Color.BLUE], Color),
            Feature("label", "x", str),
        ])
        expanded = feats.expand_enums()

        # Multi-hot counts by member, then the original string feature.
        self.assertEqual(expanded.get_names(), ["colors.RED", "colors.GREEN", "colors.BLUE", "label"])
        self.assertEqual(expanded.get_values(), [1, 0, 2, "x"])
        self.assertEqual(expanded.get_types()[:3], [int, int, int])
        self.assertEqual(expanded.get_types()[3], str)

    def test_expand_enums_mixed_list_in_order(self):
        # Ensure ordering is preserved: enum expansion first (in place of original),
        # then subsequent non-enum features in original order.
        feats = FeaturesList([
            Feature("colors", [Color.GREEN], Color),
            Feature("age", 30, int),
            Feature("primary", True, bool),
        ])
        expanded = feats.expand_enums()

        self.assertEqual(
            expanded.get_names(),
            ["colors.RED", "colors.GREEN", "colors.BLUE", "age", "primary"]
        )
        self.assertEqual(
            expanded.get_values(),
            [0, 1, 0, 30, True]
        )


if __name__ == "__main__":
    unittest.main()
