import unittest

from aidose.dataset.ade_labeling import (
    select_best_label,
    term_to_best_label_map_from_positive_terms,
    canonical_labels_from_positive_terms,
    add_label_count_features_for_study,
)
from aidose.dataset.ade import (
    ADEClinicalTermStats,
    PositiveTermMatch,
    LabelMatch,
)


class LabelsHelpersTestCase(unittest.TestCase):
    # ---------- select_best_label ----------

    def test_select_best_label_picks_highest_score(self):
        matches = [
            LabelMatch(label="Nausea", score=90),
            LabelMatch(label="Headache", score=80),
            LabelMatch(label="Vomiting", score=95),
        ]
        best = select_best_label(matches)
        self.assertIsNotNone(best)
        self.assertEqual(best.label, "Vomiting")
        self.assertEqual(best.score, 95)

    def test_select_best_label_tie_break_lexicographic(self):
        matches = [
            LabelMatch(label="Zebra", score=90),
            LabelMatch(label="Alpha", score=90),
        ]
        best = select_best_label(matches)
        self.assertIsNotNone(best)
        # Tie on score 90 -> lexicographically smallest label wins
        self.assertEqual(best.label, "Alpha")
        self.assertEqual(best.score, 90)

    def test_select_best_label_empty(self):
        self.assertIsNone(select_best_label([]))

    # ---------- term_to_best_label_map_from_positive_terms ----------

    def test_term_to_best_label_map(self):
        positive_terms = {
            "Nausea": PositiveTermMatch(
                term="Nausea",
                stats=ADEClinicalTermStats(numAffected=8, numAtRisk=30),
                matches=[
                    LabelMatch(label="Nausea", score=96),
                    LabelMatch(label="Queasiness", score=88),
                ],
            ),
            "Head pain": PositiveTermMatch(
                term="Head pain",
                stats=ADEClinicalTermStats(numAffected=3, numAtRisk=30),
                matches=[
                    LabelMatch(label="Headache", score=92),
                    LabelMatch(label="Cephalalgia", score=92),  # tie => "Cephalalgia" < "Headache"
                ],
            ),
            "Fatigue": PositiveTermMatch(
                term="Fatigue",
                stats=ADEClinicalTermStats(numAffected=0, numAtRisk=30),
                matches=[],  # no matches -> omitted
            ),
        }

        mapping = term_to_best_label_map_from_positive_terms(positive_terms)
        self.assertEqual(mapping["Nausea"], "Nausea")
        self.assertEqual(mapping["Head pain"], "Cephalalgia")
        self.assertNotIn("Fatigue", mapping)

    # ---------- canonical_labels_from_positive_terms ----------

    def test_canonical_labels_unique_and_sorted(self):
        positive_terms = {
            "Nausea": PositiveTermMatch(
                term="Nausea",
                stats=ADEClinicalTermStats(numAffected=4, numAtRisk=20),
                matches=[LabelMatch(label="Nausea", score=95)],
            ),
            "Head pain": PositiveTermMatch(
                term="Head pain",
                stats=ADEClinicalTermStats(numAffected=2, numAtRisk=20),
                matches=[LabelMatch(label="Headache", score=93)],
            ),
            "Queasy": PositiveTermMatch(
                term="Queasy",
                stats=ADEClinicalTermStats(numAffected=1, numAtRisk=20),
                matches=[LabelMatch(label="Nausea", score=91)],  # duplicate label
            ),
        }

        labels = canonical_labels_from_positive_terms(positive_terms)
        self.assertEqual(labels, ["Headache", "Nausea"])  # sorted & unique

    # ---------- add_label_count_features_for_study ----------

    def test_add_label_count_features_for_study_initializes_and_sums(self):
        positive_terms = {
            "Nausea": PositiveTermMatch(
                term="Nausea",
                stats=ADEClinicalTermStats(numAffected=5, numAtRisk=40),
                matches=[LabelMatch(label="Nausea", score=97)],
            ),
            "Head pain": PositiveTermMatch(
                term="Head pain",
                stats=ADEClinicalTermStats(numAffected=3, numAtRisk=40),
                matches=[LabelMatch(label="Headache", score=94)],
            ),
            "Queasy": PositiveTermMatch(
                term="Queasy",
                stats=ADEClinicalTermStats(numAffected=2, numAtRisk=40),
                matches=[LabelMatch(label="Nausea", score=93)],
            ),
        }

        canonical_cols = ["Headache", "Nausea", "Vomiting"]  # global columns
        out_features: dict[str, int] = {}

        add_label_count_features_for_study(
            positive_terms=positive_terms,
            canonical_label_columns=canonical_cols,
            out_features=out_features,
        )

        # Ensure all requested columns initialized
        self.assertEqual(set(out_features.keys()), {"label_Headache", "label_Nausea", "label_Vomiting"})

        # Sum per chosen best label
        self.assertEqual(out_features["label_Headache"], 3)  # from "Head pain"
        self.assertEqual(out_features["label_Nausea"], 5 + 2)  # Nausea + Queasy -> both map to "Nausea"
        self.assertEqual(out_features["label_Vomiting"], 0)  # no best matches to "Vomiting"

    def test_add_label_count_features_respects_existing_values(self):
        # Existing totals should be incremented, not overwritten
        positive_terms = {
            "Head pain": PositiveTermMatch(
                term="Head pain",
                stats=ADEClinicalTermStats(numAffected=7, numAtRisk=70),
                matches=[LabelMatch(label="Headache", score=91)],
            ),
        }
        canonical_cols = ["Headache"]
        out_features = {"label_Headache": 10}

        add_label_count_features_for_study(
            positive_terms=positive_terms,
            canonical_label_columns=canonical_cols,
            out_features=out_features,
        )

        self.assertEqual(out_features["label_Headache"], 10 + 7)

    def test_add_label_count_features_ignores_terms_with_no_matches(self):
        positive_terms = {
            "WeirdTerm": PositiveTermMatch(
                term="WeirdTerm",
                stats=ADEClinicalTermStats(numAffected=9, numAtRisk=90),
                matches=[],  # no match => ignored
            )
        }
        canonical_cols = ["Headache", "Nausea"]
        out_features: dict[str, int] = {}

        add_label_count_features_for_study(
            positive_terms=positive_terms,
            canonical_label_columns=canonical_cols,
            out_features=out_features,
        )

        self.assertEqual(out_features["label_Headache"], 0)
        self.assertEqual(out_features["label_Nausea"], 0)


if __name__ == "__main__":
    unittest.main()
