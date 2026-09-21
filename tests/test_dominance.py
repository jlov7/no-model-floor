"""Tests for the dominance analysis, including the constructed positive control.

The published finding is a zero. A zero from a measurement that cannot return anything else is
worthless, so the central test here builds a case where headroom is nonzero by construction and
requires the same code to find it.
"""

import unittest

from dominance import dominating, headroom_across
from exhaustive import configurations, label


class DominanceOnConstructedMatrices(unittest.TestCase):
    """Small hand-built score matrices where the right answer is known by inspection."""

    def test_single_row_best_everywhere_is_the_only_dominator(self):
        # config 0 is weakly best on both cells; config 1 wins nowhere; config 2 ties on one cell.
        matrix = [[1.0, 1.0], [0.5, 0.5], [1.0, 0.5]]
        configs = configurations()[: len(matrix)]
        self.assertEqual(dominating(matrix, configs), {label(configs[0])})

    def test_no_dominator_when_optima_are_split_across_cells(self):
        # config 0 wins cell A, config 1 wins cell B, neither wins both.
        matrix = [[1.0, 0.0], [0.0, 1.0]]
        configs = configurations()[:2]
        self.assertEqual(dominating(matrix, configs), set())

    def test_ties_are_all_reported_not_an_arbitrary_argmax(self):
        matrix = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
        configs = configurations()[:3]
        self.assertEqual(
            dominating(matrix, configs), {label(configs[0]), label(configs[1])}
        )


class PositiveControl(unittest.TestCase):
    """Headroom that is present by construction must be recovered exactly."""

    def test_recovers_headroom_built_in_by_construction(self):
        # Model A peaks at config 0, model B at config 1, each scoring 1.0 at its own optimum and
        # 0.0 at the other's. Any shared config averages 0.5, so headroom is exactly 0.5.
        model_a = [[1.0, 1.0], [0.0, 0.0]]
        model_b = [[0.0, 0.0], [1.0, 1.0]]
        uniform = [0.5, 0.5]
        self.assertAlmostEqual(headroom_across([model_a, model_b], uniform), 0.5 * 10000, places=6)

    def test_reports_zero_when_a_shared_optimum_exists(self):
        model_a = [[1.0, 1.0], [0.0, 0.0]]
        model_b = [[1.0, 1.0], [0.5, 0.5]]
        uniform = [0.5, 0.5]
        self.assertAlmostEqual(headroom_across([model_a, model_b], uniform), 0.0, places=6)

    def test_a_dominating_config_is_optimal_under_every_weighting(self):
        # config 0 dominates cell by cell, so no weighting can make config 1 preferable.
        model_a = [[1.0, 0.8], [0.9, 0.1]]
        model_b = [[1.0, 0.8], [0.2, 0.7]]
        configs = configurations()[:2]
        self.assertIn(label(configs[0]), dominating(model_a, configs))
        self.assertIn(label(configs[0]), dominating(model_b, configs))
        for w in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.9, 0.1]):
            self.assertAlmostEqual(headroom_across([model_a, model_b], w), 0.0, places=6)


class PublishedDominanceResult(unittest.TestCase):
    """The committed evidence must still say what the README says it says."""

    def setUp(self):
        import json
        from pathlib import Path
        self.report = json.loads(Path("evidence/dominance.json").read_text(encoding="utf-8"))

    def test_a_shared_dominating_config_exists_on_both_surfaces(self):
        for surface, block in self.report.items():
            for metric, r in block.items():
                with self.subTest(surface=surface, metric=metric):
                    self.assertGreater(r["dominating_intersection_size"], 0)
                    self.assertTrue(r["headroom_is_zero_under_every_reweighting"])

    def test_random_reweightings_never_produce_headroom(self):
        for surface, block in self.report.items():
            for metric, r in block.items():
                with self.subTest(surface=surface, metric=metric):
                    self.assertAlmostEqual(
                        r["max_headroom_bps_over_random_reweightings"], 0.0, places=6
                    )

    def test_sensitivity_control_finds_nonzero_headroom(self):
        # If this ever passes with a zero, the sweep has stopped being able to detect anything.
        for surface, block in self.report.items():
            for metric, r in block.items():
                with self.subTest(surface=surface, metric=metric):
                    self.assertTrue(r["sensitivity_control_passes"])
                    self.assertGreater(r["sensitivity_control_guards_off_headroom_bps"], 0)


if __name__ == "__main__":
    unittest.main()


class TheSurvivalCheckCanActuallyFail(unittest.TestCase):
    """The previous version of this check could not, which is why it is tested now.

    It resampled whole cells from an already-computed score matrix. A configuration weakly best on
    every column of a fixed matrix is weakly best on any multiset of those columns, so it always
    reported 100% survival, whatever the data, while being cited as evidence that the dominance was
    not an accident of sampling. Resampling draws within cells re-scores from the responses, so
    accidental dominance can break. Both directions are asserted here.
    """

    def test_it_reports_failure_when_dominance_is_accidental(self):
        import random

        from dominance import dominance_survives_resampling
        from exhaustive import GUARDS, configurations

        guards_off = [c for c in configurations() if not any(c[g] for g in GUARDS)]
        rng = random.Random(11)
        borderline = lambda: [rng.choice([4900, 5100]) for _ in range(10)]
        cells = {
            "a": [{"approval": "APPROVED", "authority": "VALID", "decision": "ACT",
                   "confidence_bps": c} for c in borderline()],
            "b": [{"approval": "DENIED", "authority": "VALID", "decision": "ACT",
                   "confidence_bps": c} for c in borderline()],
        }
        survival = dominance_survives_resampling({"m": cells}, "accuracy", guards_off)
        self.assertLess(survival, 1.0, "a check that cannot report failure is not a check")

    def test_it_reports_survival_on_the_real_evidence(self):
        import json
        from pathlib import Path

        report = json.loads(Path("evidence/dominance.json").read_text(encoding="utf-8"))
        for surface, block in report.items():
            for metric, r in block.items():
                with self.subTest(surface=surface, metric=metric):
                    self.assertEqual(r["dominance_survives_draw_bootstrap_fraction"], 1.0)
