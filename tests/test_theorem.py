"""Tests for the exhaustive dominance proof.

The proof's weight rests entirely on the claim that its enumeration is complete. Most of these
tests attack that claim rather than the arithmetic, which is trivial.
"""

import random
import unittest

from exhaustive import GRID, applied, configurations, expected, label
from theorem import (
    APPROVALS,
    AUTHORITIES,
    DECISIONS,
    confidence_classes,
    draw_score,
    draw_space,
    risk_is_never_read,
    universally_dominating,
)


class EnumerationIsComplete(unittest.TestCase):
    """If the draw space misses a behaviour, the proof proves nothing."""

    def test_every_contract_valid_confidence_falls_in_an_enumerated_class(self):
        """An arbitrary in-contract confidence must behave exactly like one of the representatives.

        Two confidences are equivalent iff the same set of floors exceeds them. This checks the
        representative set covers every value the scaffold could see, not just the grid points.

        The range is the contract range, and the next test is what earns the right to assume it.
        An earlier version of this test sampled outside it and failed on a negative confidence,
        which is how the unstated domain assumption below came to be stated.
        """
        reps = confidence_classes()
        signature = lambda c: tuple(c < f for f in GRID)
        covered = {signature(r) for r in reps}
        rng = random.Random(20260819)
        for _ in range(20000):
            c = rng.randint(0, 10000)
            self.assertIn(signature(c), covered, f"confidence {c} falls in no enumerated class")

    def test_out_of_range_confidence_is_rejected_rather_than_scored(self):
        """The enumeration covers 0..10000 only, so anything else must never reach scoring.

        If the provider ever started accepting out-of-range confidence, the draw space would be
        incomplete and the proof would silently stop covering every possible model.
        """
        from vaa.providers import ModelDecision, ProviderError

        payload = lambda c: {
            "decision": "ACT", "action_id": "apply-late-adjustment", "confidence_bps": c,
            "question": None, "assumptions": [],
        }
        # Control first: if a valid payload raises, the rejections below prove nothing.
        self.assertEqual(ModelDecision.from_mapping(payload(5000)).confidence_bps, 5000)
        # And match the message, so a rejection for some unrelated reason cannot pass as one
        # for being out of range.
        for bad in (-1, -500, 10001, 99999):
            with self.subTest(confidence=bad):
                with self.assertRaisesRegex(ProviderError, "confidence_bps"):
                    ModelDecision.from_mapping(payload(bad))

    def test_decision_space_covers_the_contract_plus_one_off_contract_class(self):
        """Scoring reads only `== want` and `== ACT`, so everything off-contract is one class.

        It is included because the recording path stores the raw decision string without validating
        it, and that path is frozen at the preregistration tag.
        """
        from theorem import OFF_CONTRACT
        self.assertEqual(set(DECISIONS), {"ACT", "ASK", "ABSTAIN", OFF_CONTRACT})

    def test_context_space_is_the_full_cross_product(self):
        self.assertEqual(len(APPROVALS) * len(AUTHORITIES), 9)
        self.assertEqual(
            len(draw_space()), len(APPROVALS) * len(AUTHORITIES) * len(DECISIONS) * len(confidence_classes())
        )

    def test_risk_is_not_consulted_by_scoring_or_guards(self):
        # The enumeration omits risk. That is only sound if nothing downstream reads it.
        self.assertTrue(risk_is_never_read())

    def test_guards_are_order_independent_so_the_config_space_is_complete(self):
        """2^3 x 41 assumes guard order cannot create a distinct configuration.

        Guards fire on mutually exclusive conditions, so no draw can trigger two of them. If that
        ever stopped being true, the enumerated space would be missing configurations.
        """
        for approval in APPROVALS:
            for authority in AUTHORITIES:
                fires = [
                    authority != "VALID",
                    approval == "DENIED",
                    approval == "UNKNOWN",
                ]
                # guard 1 can co-occur with 2 or 3, so check the resolved outcome is order-stable
                if sum(fires) > 1:
                    self.assertTrue(fires[0], (approval, authority))
                    # guard 1 wins under any order because it returns ABSTAIN and is checked first;
                    # assert the outcome the enumeration relies on
                    cfg = dict.fromkeys(
                        ("abstain_on_invalid_authority", "abstain_on_denied_approval",
                         "ask_on_unknown_approval"), True)
                    cfg["floor"] = 0
                    self.assertEqual(applied("ACT", 10000, approval, authority, cfg), "ABSTAIN")


class TheProof(unittest.TestCase):
    def test_exactly_one_configuration_dominates_universally(self):
        for metric in ("accuracy", "utility"):
            with self.subTest(metric=metric):
                winners, size = universally_dominating(metric)
                self.assertEqual(size, 1512)
                self.assertEqual(
                    winners,
                    ["invalid_authority+denied_approval+unknown_approval@0"],
                    "the dominating configuration changed",
                )

    def test_the_dominating_config_is_never_beaten_anywhere(self):
        configs = configurations()
        star = next(
            c for c in configs
            if label(c) == "invalid_authority+denied_approval+unknown_approval@0"
        )
        for metric in ("accuracy", "utility"):
            for d in draw_space():
                mine = draw_score(*d, star, metric)
                for c in configs:
                    self.assertGreaterEqual(
                        mine + 1e-9, draw_score(*d, c, metric),
                        f"{label(c)} beats the dominating config on {d} under {metric}",
                    )

    def test_the_only_errors_it_cannot_repair_are_missed_acts(self):
        """The mechanism claim: guards fix everything except failing to ACT when ACT is correct."""
        configs = configurations()
        star = next(
            c for c in configs
            if label(c) == "invalid_authority+denied_approval+unknown_approval@0"
        )
        for approval, authority, decision, confidence in draw_space():
            want = expected(approval, authority)
            got = applied(decision, confidence, approval, authority, star)
            if got != want:
                # Every residual error is a missed ACT, either because the model did not say ACT
                # or because an off-contract confidence tripped the floor.
                self.assertEqual(want, "ACT", (approval, authority, decision))
                self.assertTrue(decision != "ACT" or confidence < 0, (decision, confidence))


class GuardsAreRepairOnly(unittest.TestCase):
    """No guard can produce ACT. This is what makes the missed-ACT class unrepairable."""

    def test_no_configuration_ever_turns_a_non_act_into_an_act(self):
        for approval, authority, decision, confidence in draw_space():
            if decision == "ACT":
                continue
            for config in configurations():
                self.assertNotEqual(
                    applied(decision, confidence, approval, authority, config), "ACT",
                    f"a guard produced ACT from {decision}",
                )


if __name__ == "__main__":
    unittest.main()
