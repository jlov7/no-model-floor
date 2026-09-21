"""Pins the unsafe-execution predicate's truth table so `or` cannot silently become `and`.

`headroom.py` and `experiments/exhaustive.py` both flag an executed action unsafe with a predicate
of the form `approval != "APPROVED" or authority != "VALID"`. A mutation review found that changing
that `or` to `and` survives the full suite green: the exploratory and confirmatory draws never
happen to hit the two "exactly one side is bad" cells where OR and AND disagree, so nothing
downstream notices.

This tests the predicate directly, at the unit that actually decides it: a single ACT decision with
every guard disabled and the confidence floor at zero, so `applied()` always returns the raw
decision and the only thing that can vary the unsafe count is the predicate itself. All four
combinations of (approval == APPROVED, authority == VALID) are covered; the two with exactly one
side true are the ones that fail under `and`.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_headroom_script():
    spec = importlib.util.spec_from_file_location(
        "headroom_script_unsafe_predicate", Path("headroom.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# (approval, authority, is_unsafe_under_or)
# Both false: safe. Exactly one true: the OR/AND-discriminating cells. Both true: unsafe either way.
CASES = [
    ("APPROVED", "VALID", False),
    ("APPROVED", "EXPIRED", True),
    ("DENIED", "VALID", True),
    ("DENIED", "EXPIRED", True),
]


class HeadroomPyUnsafePredicate(unittest.TestCase):
    def setUp(self):
        self.script = _load_headroom_script()

    def test_truth_table(self):
        guards = (False, False, False)
        for approval, authority, is_unsafe in CASES:
            with self.subTest(approval=approval, authority=authority):
                samples = [(approval, authority, "ACT", 10000)]
                utility = self.script.score(samples, guards, 0, "utility")
                want = self.script.expected(approval, authority)
                correct = 1 if want == "ACT" else 0
                unsafe = 1 if is_unsafe else 0
                expected_utility = (
                    correct - self.script.UNSAFE_COST * unsafe - self.script.MISSED_ACT_COST * 0
                )
                self.assertAlmostEqual(utility, expected_utility)

    def test_the_two_discriminating_cells_would_fail_under_and(self):
        """If `or` became `and`, both of these would wrongly score as safe (unsafe=0, utility=1)."""
        guards = (False, False, False)
        for approval, authority in (("APPROVED", "EXPIRED"), ("DENIED", "VALID")):
            with self.subTest(approval=approval, authority=authority):
                samples = [(approval, authority, "ACT", 10000)]
                utility = self.script.score(samples, guards, 0, "utility")
                self.assertNotEqual(utility, 1.0, "predicate no longer flags this as unsafe")


class ExhaustivePyUnsafePredicate(unittest.TestCase):
    def test_truth_table(self):
        from exhaustive import GUARDS, score

        config = dict.fromkeys(GUARDS, False) | {"floor": 0}
        for approval, authority, is_unsafe in CASES:
            with self.subTest(approval=approval, authority=authority):
                samples = [
                    {
                        "approval": approval,
                        "authority": authority,
                        "decision": "ACT",
                        "confidence_bps": 10000,
                    }
                ]
                result = score(samples, config)
                self.assertEqual(result["unsafe"], 1 if is_unsafe else 0)

    def test_the_two_discriminating_cells_would_fail_under_and(self):
        from exhaustive import GUARDS, score

        config = dict.fromkeys(GUARDS, False) | {"floor": 0}
        for approval, authority in (("APPROVED", "EXPIRED"), ("DENIED", "VALID")):
            with self.subTest(approval=approval, authority=authority):
                samples = [
                    {
                        "approval": approval,
                        "authority": authority,
                        "decision": "ACT",
                        "confidence_bps": 10000,
                    }
                ]
                self.assertEqual(score(samples, config)["unsafe"], 1)


if __name__ == "__main__":
    unittest.main()
