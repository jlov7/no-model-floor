"""Soundness of the guards, stated as it actually is rather than as the paper once claimed.

PAPER.md asserted that each of the three guards fires only where its own output is correct. That is
false, and no test caught it because none existed; worse, the ad-hoc check written when the claim
was added carried a carve-out that excluded precisely the failing cells.

What holds is compositional: the guards in the order the scaffold fixes never override with a wrong
answer, because the invalid-authority guard takes precedence on the cells where the ask guard alone
would be wrong. Both halves are asserted here, the false one negatively, so neither can drift again.
"""

import unittest

from exhaustive import GUARDS, applied, configurations, expected, label
from theorem import APPROVALS, AUTHORITIES, DECISIONS, OFF_CONTRACT, confidence_classes

CONTEXTS = [(a, u) for a in APPROVALS for u in AUTHORITIES]
OUTPUT = {
    "abstain_on_invalid_authority": "ABSTAIN",
    "abstain_on_denied_approval": "ABSTAIN",
    "ask_on_unknown_approval": "ASK",
}


def only(guard):
    config = {g: (g == guard) for g in GUARDS}
    config["floor"] = 0
    return config


class IndividualGuardsAreNotAllSound(unittest.TestCase):
    """The claim the paper used to make, asserted in the negative so it cannot come back."""

    def test_the_ask_guard_is_unsound_in_isolation(self):
        wrong = [
            (a, u)
            for a, u in CONTEXTS
            if applied("ACT", 10000, a, u, only("ask_on_unknown_approval")) == "ASK"
            and expected(a, u) != "ASK"
        ]
        self.assertEqual(
            sorted(wrong), [("UNKNOWN", "EXPIRED"), ("UNKNOWN", "MISSING")],
            "the ask guard fires on unknown approval regardless of authority",
        )

    def test_the_other_two_guards_are_sound_in_isolation(self):
        for guard in ("abstain_on_invalid_authority", "abstain_on_denied_approval"):
            with self.subTest(guard=guard):
                wrong = [
                    (a, u) for a, u in CONTEXTS
                    if applied("ACT", 10000, a, u, only(guard)) == OUTPUT[guard]
                    and expected(a, u) != OUTPUT[guard]
                ]
                self.assertEqual(wrong, [])


class TheCompositionIsSound(unittest.TestCase):
    """What the paper may claim: wherever the ordered composition overrides, it is right."""

    def setUp(self):
        self.star = next(
            c for c in configurations()
            if label(c) == "invalid_authority+denied_approval+unknown_approval@0"
        )

    def test_no_override_is_ever_wrong(self):
        """Over contract-valid inputs. The one exception is the floor, not a guard: an
        out-of-contract negative confidence trips the floor even at zero, which overrides a correct
        ACT. That is a contract violation reaching a control, not the guards being unsound."""
        for approval, authority, decision, confidence in (
            (a, u, d, c)
            for a, u in CONTEXTS
            for d in DECISIONS
            for c in confidence_classes()
            if 0 <= c <= 10000
        ):
            if decision == OFF_CONTRACT:
                continue
            got = applied(decision, confidence, approval, authority, self.star)
            if got != decision:
                self.assertEqual(
                    got, expected(approval, authority),
                    f"scaffold overrode {decision} with {got} on {(approval, authority)}",
                )

    def test_precedence_is_what_rescues_the_ask_guard(self):
        """If the invalid-authority guard stopped taking precedence, soundness would break."""
        config = {g: True for g in GUARDS}
        config["floor"] = 0
        for authority in ("EXPIRED", "MISSING"):
            self.assertEqual(applied("ACT", 10000, "UNKNOWN", authority, config), "ABSTAIN")
            self.assertEqual(
                applied("ACT", 10000, "UNKNOWN", authority, only("ask_on_unknown_approval")), "ASK")


class OrderIsPartOfTheFamily(unittest.TestCase):
    """Same guards, same predicates and outputs, different order: dominance fails."""

    @staticmethod
    def ask_first(decision, approval, authority, enabled):
        if approval == "UNKNOWN" and enabled:
            return "ASK"
        if authority != "VALID" and enabled:
            return "ABSTAIN"
        if approval == "DENIED" and enabled:
            return "ABSTAIN"
        return decision

    def test_reordering_makes_enabling_everything_no_longer_optimal(self):
        cells = [(a, u) for a, u in CONTEXTS for _ in (0, 1)]  # two risk tiers, unread
        def score(model, enabled):
            return sum(
                self.ask_first(model, a, u, enabled) == expected(a, u) for a, u in cells
            ) / len(cells)
        # An always-abstaining model does strictly worse with the reordered guards enabled.
        self.assertLess(score("ABSTAIN", True), score("ABSTAIN", False))
        self.assertGreater(score("ACT", True), score("ACT", False))


if __name__ == "__main__":
    unittest.main()
