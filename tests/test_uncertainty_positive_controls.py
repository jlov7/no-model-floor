"""Positive controls for the headline paired-equality claim in `experiments/uncertainty.py`.

CI asserts `decisions_differ == 0` for the derived-vs-best-fixed comparison, and only that. Two
mutations exploit this and survive the full suite green because the committed value genuinely is
zero, and nothing ever exercises a case where it should NOT be zero:

  (a) `paired_disagreement` can have its counter changed from `differ += 1` to `differ += 0`, so it
      can never report a disagreement no matter what it is fed.
  (b) `derived_config` can be made to return an all-guards config for every model -- exactly the
      historical bug the function was rewritten to kill, where the comparison was passed the same
      policy as both arguments and so could never disagree with itself.

Both tests below use real committed draws and real evidence manifests, not synthetic data, so they
also stand as a check that the committed evidence still has the shape these controls assume.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from exhaustive import GUARDS, load
from uncertainty import derived_config, paired_disagreement

ABLATION_ROOT = Path("evidence/ablation")


def _explore_cell(r):
    approval, authority, _risk = r["scenario_id"].split("-")
    return {
        "decision": r["decision"],
        "confidence_bps": r["confidence_bps"],
        "approval": approval.upper(),
        "authority": authority.upper(),
    }


class PairedDisagreementHasAPositiveControl(unittest.TestCase):
    """Two genuinely different fixed policies, over real draws, must disagree on some of them.

    This is the control that a mutated `differ += 0` fails: it always reports zero disagreements
    regardless of input, whereas real behaviour must report a nonzero count for policies known to
    differ.
    """

    def test_all_guards_vs_guards_off_disagree_on_some_draws(self):
        per_model = load(ABLATION_ROOT, _explore_cell)
        all_guards = dict.fromkeys(GUARDS, True) | {"floor": 0}
        guards_off = dict.fromkeys(GUARDS, False) | {"floor": 0}
        self.assertTrue(per_model, "no evidence runs loaded; test setup is broken")
        for model, samples in per_model.items():
            with self.subTest(model=model):
                result = paired_disagreement(samples, guards_off, all_guards)
                self.assertGreater(
                    result["decisions_differ"],
                    0,
                    "all-guards and guards-off are known to differ on this benchmark; "
                    "a zero count here means the counter cannot report disagreement",
                )
                self.assertEqual(result["draws"], len(samples))

    def test_the_same_policy_against_itself_never_differs(self):
        """The negative case: this is the one comparison that SHOULD be zero, to bound the control."""
        per_model = load(ABLATION_ROOT, _explore_cell)
        all_guards = dict.fromkeys(GUARDS, True) | {"floor": 0}
        for model, samples in per_model.items():
            with self.subTest(model=model):
                result = paired_disagreement(samples, all_guards, all_guards)
                self.assertEqual(result["decisions_differ"], 0)


class DerivedConfigIsNotAlwaysAllGuards(unittest.TestCase):
    """The historical bug: `derived_config` returning all-guards for every model.

    That would make the paired comparison test a policy against itself no matter which model was
    asked for, which is exactly what made `decisions_differ` structurally zero before this function
    was rewritten. Find models that derived only two of the three guards directly from the
    committed evidence, rather than hardcoding a guess about which ones, so this stays correct if
    the evidence is regenerated with different models.
    """

    def _models_with_partial_guards(self):
        import json

        partial = []
        for run_dir in sorted(ABLATION_ROOT.glob("*")):
            manifest_path = run_dir / "ablation.json"
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            guards = manifest.get("results", {}).get("derived", {}).get("guards")
            if guards is not None and len(guards) < len(GUARDS):
                partial.append(manifest["model"])
        return partial

    def test_the_evidence_contains_at_least_one_partial_guard_model(self):
        """Guards this test's premise: if this ever stops holding, the test below is vacuous."""
        self.assertTrue(
            self._models_with_partial_guards(),
            "no model in evidence/ablation derived fewer than all guards; "
            "the positive control below cannot exercise anything",
        )

    def test_a_partial_guard_models_derived_config_is_not_all_guards(self):
        all_guards = dict.fromkeys(GUARDS, True) | {"floor": 0}
        for model in self._models_with_partial_guards():
            with self.subTest(model=model):
                config = derived_config(ABLATION_ROOT, model)
                self.assertNotEqual(
                    config,
                    all_guards,
                    f"{model} derived fewer than all guards per the committed evidence, "
                    "but derived_config returned the all-guards policy",
                )


if __name__ == "__main__":
    unittest.main()
