"""The reimplementation used by the proof must match the scaffold the experiments actually ran.

`experiments/theorem.py` exhausts the behaviour space of `exhaustive.applied` and
`exhaustive.expected`, which are standalone reimplementations. Everything the proof claims about
"every possible model" is a claim about the real scaffold only if those two agree with
`vaa.agent.apply_scaffold_rules` and `vaa.environment.expected_decision_for` everywhere.

That bridge was previously asserted by inspection rather than tested, which is the same failure
mode this repository has already found in itself twice. So it is checked exhaustively here, over
the entire enumerated draw space crossed with the entire configuration space, including the
off-contract decision and negative-confidence classes the enumeration was widened to cover.
"""

import unittest

from exhaustive import GUARDS, applied, configurations, expected
from theorem import OFF_CONTRACT, draw_space

from vaa.agent import apply_scaffold_rules
from vaa.environment import expected_decision_for
from vaa.scaffold import ScaffoldSpec
from vaa.types import DecisionKind


def as_spec(config):
    return ScaffoldSpec(
        scaffold_id="parity-check",
        version=1,
        abstain_on_invalid_authority=config["abstain_on_invalid_authority"],
        abstain_on_denied_approval=config["abstain_on_denied_approval"],
        ask_on_unknown_approval=config["ask_on_unknown_approval"],
        min_confidence_bps=config["floor"],
        # Neither of these can change a decision; they are outside the guard set the proof
        # enumerates and only affect resource accounting.
        require_world_model_prediction=False,
        verify_outcome=False,
        max_model_calls=1,
        max_tool_calls=3,
    )


class ReimplementationMatchesTheRealScaffold(unittest.TestCase):
    def test_expected_decision_agrees_on_every_context(self):
        for approval in ("APPROVED", "DENIED", "UNKNOWN"):
            for authority in ("VALID", "EXPIRED", "MISSING"):
                with self.subTest(approval=approval, authority=authority):
                    self.assertEqual(
                        expected(approval, authority),
                        expected_decision_for(authority, approval).value,
                    )

    def test_applied_agrees_over_the_whole_draw_and_configuration_space(self):
        """Every enumerated draw against every configuration. This is the proof's load-bearing bridge."""
        specs = [(c, as_spec(c)) for c in configurations()]
        mismatches = []
        for approval, authority, decision, confidence in draw_space():
            if decision == OFF_CONTRACT:
                continue  # not representable as a DecisionKind; covered separately below
            kind = DecisionKind(decision)
            for config, spec in specs:
                mine = applied(decision, confidence, approval, authority, config)
                theirs = apply_scaffold_rules(
                    spec,
                    authority_status=authority,
                    approval_status=approval,
                    decision=kind,
                    confidence_bps=confidence,
                ).value
                if mine != theirs:
                    mismatches.append(
                        (approval, authority, decision, confidence, config, mine, theirs)
                    )
        self.assertEqual(mismatches[:3], [], f"{len(mismatches)} disagreements")

    def test_off_contract_decisions_cannot_be_produced_by_the_real_scaffold(self):
        """The real scaffold is typed, so this class cannot be bridged, and does not need to be.

        `apply_scaffold_rules` takes a DecisionKind, so an off-contract string cannot reach it, and
        the recorder raises on an unrecognised decision before writing anything. The class is in the
        enumeration for safety, not because it is reachable. What the proof needs from it is only
        that no configuration turns such a value into ACT, which is asserted on the reimplementation
        below, plus the fact that the typed boundary exists at all, asserted here.
        """
        from vaa.types import DecisionKind

        with self.assertRaises(ValueError):
            DecisionKind(OFF_CONTRACT)

        for approval, authority, decision, confidence in draw_space():
            if decision != OFF_CONTRACT:
                continue
            for config in configurations():
                self.assertNotEqual(
                    applied(decision, confidence, approval, authority, config), "ACT"
                )

    def test_the_guard_names_match_the_scaffold_fields(self):
        spec = as_spec({g: True for g in GUARDS} | {"floor": 0})
        for guard in GUARDS:
            self.assertTrue(hasattr(spec, guard), guard)


if __name__ == "__main__":
    unittest.main()


class HeadroomScriptMatchesTheAnalysis(unittest.TestCase):
    """`headroom.py` reimplements the scoring a fourth time, and had no test at all.

    It is the file the README asks a sceptical reader to run first, and mutation testing showed that its
    ground truth could be corrupted and its cost constants zeroed while every test passed and the
    script exited zero. It now asserts its own claims, which catches the first mutation; this
    catches the rest by requiring its logic to agree with the analysis it summarises.
    """

    def setUp(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location("headroom_script", Path("headroom.py"))
        self.script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.script)

    def test_expected_answers_agree(self):
        for approval in ("APPROVED", "DENIED", "UNKNOWN"):
            for authority in ("VALID", "EXPIRED", "MISSING"):
                with self.subTest(approval=approval, authority=authority):
                    self.assertEqual(
                        self.script.expected(approval, authority), expected(approval, authority)
                    )

    def test_applied_decisions_agree_over_the_whole_space(self):
        for approval, authority, decision, confidence in draw_space():
            if decision == OFF_CONTRACT:
                continue
            for config in configurations():
                guards = tuple(config[g] for g in GUARDS)
                mine = self.script.applied(
                    decision, confidence, approval, authority, guards, config["floor"]
                )
                theirs = applied(decision, confidence, approval, authority, config)
                if mine != theirs:
                    self.fail(
                        f"headroom.py disagrees on {(approval, authority, decision, confidence)}: "
                        f"{mine} vs {theirs}"
                    )

    def test_the_cost_constants_agree(self):
        """Zeroing these in the script is invisible in the headline, so pin them directly."""
        from exhaustive import MISSED_ACT_COST, UNSAFE_COST

        self.assertEqual(self.script.UNSAFE_COST, UNSAFE_COST)
        self.assertEqual(self.script.MISSED_ACT_COST, MISSED_ACT_COST)

    def test_out_of_domain_values_fail_loudly(self):
        """A corrupted draw must raise rather than score as a silent ABSTAIN.

        The previous `.get(approval, "ABSTAIN")` silently classified anything outside
        {APPROVED, DENIED, UNKNOWN} as ABSTAIN, and any authority outside {VALID, EXPIRED,
        MISSING} as non-VALID. That is the exact failure class this repository exists to
        detect: a wrong number that no check catches because no check looks.
        """
        for approval in ("BANANA", "", None, "approved", 3):
            with self.subTest(approval=approval):
                with self.assertRaises(ValueError):
                    self.script.expected(approval, "VALID")
        for authority in ("MAYBE", "", None, "valid", 3):
            with self.subTest(authority=authority):
                with self.assertRaises(ValueError):
                    self.script.expected("APPROVED", authority)

    def test_empty_samples_fail_loudly(self):
        """An empty sample list must raise, not divide by zero."""
        with self.assertRaises(ValueError):
            self.script.score([], (False, False, False), 0, "accuracy")

    def test_empty_surfaces_fail_loudly(self):
        """main() must report missing evidence as a failure, not crash with a traceback."""
        import contextlib
        import io
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            self.script.ROOT = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.script.main(), 1)
