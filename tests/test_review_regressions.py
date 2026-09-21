"""Discriminating regressions from the 2026-09-20 release review."""

import re
import unittest
from math import comb

from floors import Benchmark, HeadroomResult, adaptation_headroom, no_model_floor


class ReviewRegressions(unittest.TestCase):
    def test_failed_validation_never_publishes_a_partial_policy_cache(self):
        bench = Benchmark(
            items=[0, 1], correct=lambda x: x,
            policies={"valid": lambda rows: [0, 1], "invalid": lambda rows: [0]},
        )
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "invalid.*1 predictions for 2"):
                no_model_floor(bench)

    def test_zero_concentrated_null_does_not_veto_a_positive_control(self):
        # Both groups have balanced labels. Policy errors, not group labels,
        # differ: B beats A by six decisions in group 0; A beats B by 30 in group 1.
        items = []
        for group in (0, 1):
            for i in range(40):
                y = i % 2
                items.append({
                    "group": group, "label": y,
                    "a": y ^ (group == 0 and i < 8),
                    "b": y ^ ((group == 0 and 8 <= i < 10) or (group == 1 and i < 30)),
                })
        bench = Benchmark(
            items=items, correct=lambda r: r["label"],
            policies={"a": lambda rows: [r["a"] for r in rows],
                      "b": lambda rows: [r["b"] for r in rows]},
            unit=lambda r: r["group"],
        )
        result = adaptation_headroom(bench, rounds=500)
        self.assertAlmostEqual(result.headroom, 0.075)
        self.assertGreaterEqual(result.null_zero_fraction, 0.9)
        self.assertLess(result.permutation_p, 0.05)
        self.assertTrue(result.exceeds_chance)
        self.assertNotIn("p-value means nothing", str(result))
        denominator = comb(80, 40)
        def tail(threshold):
            return sum(
                comb(8, p) * comb(32, m) * comb(40, 40 - p - m)
                for p in range(9) for m in range(33)
                if 0 <= 40 - p - m <= 40 and p - m >= threshold
            )
        exact_tail = 2 * tail(6) / denominator
        exact_zero = 1 - 2 * tail(1) / denominator
        self.assertAlmostEqual(exact_tail, 8.356849712789674e-12, places=20)
        self.assertAlmostEqual(exact_zero, 0.9999949677595236, places=15)

    def test_a_positive_permutation_probability_is_not_printed_as_zero(self):
        result = HeadroomResult(
            headroom=0.1, permutation_p=1 / 2001, units=2,
            best_shared_policy="a", per_unit_best={}, dropped_units={},
        )
        match = re.search(r"permutation p: ([0-9.eE+-]+)", str(result))
        self.assertIsNotNone(match)
        self.assertGreater(float(match.group(1)), 0)

    def test_headroom_names_an_empty_policy_space(self):
        items = [{"y": i % 2, "u": i // 40} for i in range(80)]
        bench = Benchmark(items, lambda r: r["y"], {}, unit=lambda r: r["u"])
        with self.assertRaisesRegex(ValueError, "at least one policy"):
            adaptation_headroom(bench, rounds=100)
