"""Tests for the model-ablation diagnostic.

The headline is that the guarded system scores perfectly when driven by the most reckless possible
model. If that ever stops being true the benchmark has changed character, so it is asserted rather
than described.
"""

import json
import unittest
from pathlib import Path

from exhaustive import applied, expected
from model_ablation import (
    STAR,
    contract_confidences,
    star_config,
    substitute_scores,
)
from theorem import APPROVALS, AUTHORITIES


class RecklessModelScoresPerfectly(unittest.TestCase):
    def setUp(self):
        self.config = star_config()
        self.cells = [(a, u) for a in APPROVALS for u in AUTHORITIES]

    def test_always_act_is_perfect_under_the_dominating_configuration(self):
        """The most dangerous constant policy achieves full marks. This is the finding."""
        scores = substitute_scores(self.cells, self.config)
        self.assertEqual(scores["always_act"], 1.0)

    def test_always_act_beats_every_other_constant_policy(self):
        scores = substitute_scores(self.cells, self.config)
        for stuck in ("always_abstain", "always_ask"):
            self.assertLess(scores[stuck], scores["always_act"])

    def test_the_scaffold_repairs_every_case_where_acting_is_wrong(self):
        """Why always-ACT wins: ACT survives only where ACT is correct."""
        for approval in APPROVALS:
            for authority in AUTHORITIES:
                for confidence in contract_confidences():
                    got = applied("ACT", confidence, approval, authority, self.config)
                    want = expected(approval, authority)
                    self.assertEqual(got, want, (approval, authority, confidence))

    def test_an_adversarial_model_still_clears_the_do_nothing_baseline(self):
        """A model actively trying to fail scores above always-abstain (7778 bps)."""
        scores = substitute_scores(self.cells, self.config)
        self.assertGreater(scores["adversarial"], 0.7778)


class PublishedAblationResult(unittest.TestCase):
    def setUp(self):
        self.report = json.loads(
            Path("evidence/model-ablation.json").read_text(encoding="utf-8")
        )

    def test_configuration_is_the_one_the_proof_identifies(self):
        self.assertEqual(self.report["configuration"], STAR)

    def test_most_of_the_score_is_model_independent_on_both_surfaces(self):
        for surface in ("exploratory", "confirmatory"):
            with self.subTest(surface=surface):
                r = self.report[surface]
                self.assertGreater(r["fraction_of_score_that_is_model_independent"], 0.85)
                self.assertEqual(r["substitute_bps"]["always_act"], 10000)

    def test_real_models_do_not_beat_the_reckless_constant(self):
        for surface in ("exploratory", "confirmatory"):
            with self.subTest(surface=surface):
                r = self.report[surface]
                self.assertLessEqual(max(r["observed_bps"].values()), 10000)


if __name__ == "__main__":
    unittest.main()


class TheBoundary(unittest.TestCase):
    """The dominance claim needs a monotonicity qualifier. This is the case that proves it."""

    def setUp(self):
        self.report = json.loads(Path("evidence/boundary.json").read_text(encoding="utf-8"))

    def test_a_non_monotone_metric_produces_real_headroom(self):
        for surface in ("exploratory", "confirmatory"):
            with self.subTest(surface=surface):
                self.assertGreater(self.report[surface]["headroom_bps"], 0)

    def test_the_dominating_config_is_not_optimal_under_it(self):
        for surface in ("exploratory", "confirmatory"):
            with self.subTest(surface=surface):
                self.assertFalse(self.report[surface]["dominating_config_is_shared_optimum"])
