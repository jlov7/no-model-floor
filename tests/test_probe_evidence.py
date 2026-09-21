"""The probe evidence is committed and was read by nothing.

An engineering review noted that `evidence/probe/` ships with the repository, is described in the
README's file listing, and is not touched by any test or CI step. Committed data that nothing checks
can drift or be wrong indefinitely, which is the condition this project exists to complain about.
These cover it: the files must be internally consistent, must agree with the models that appear in
the ablation evidence, and must say what the README says they say.
"""

import json
import unittest
from pathlib import Path

PROBE = Path("evidence/probe")


def probe_files():
    return sorted(PROBE.glob("*.json"))


def _directory_for(model: str) -> str:
    """The ablation directory holding a given model's run."""
    for path in Path("evidence/ablation").glob("*/ablation.json"):
        if json.loads(path.read_text(encoding="utf-8"))["model"] == model:
            return path.parent.name
    raise AssertionError(f"no ablation directory for {model}")


class ProbeEvidenceIsIntact(unittest.TestCase):
    def test_it_exists_and_covers_every_model_in_the_ablation(self):
        self.assertTrue(probe_files(), "probe evidence is listed in the README but absent")
        probed = {json.loads(p.read_text(encoding="utf-8"))["model"] for p in probe_files()}
        ablated = {
            json.loads(p.read_text(encoding="utf-8"))["model"]
            for p in Path("evidence/ablation").glob("*/ablation.json")
        }
        self.assertEqual(
            probed, ablated,
            "the probe covers a different set of models from the ablation it precedes",
        )

    def test_each_file_is_internally_consistent(self):
        for path in probe_files():
            with self.subTest(file=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(record["cell_count"], 18)
                self.assertLessEqual(record["answered"], record["cell_count"])
                self.assertLessEqual(record["correct"], record["answered"])
                self.assertLessEqual(record["unsafe_decisions"], record["answered"])
                self.assertEqual(len(record["rows"]), record["cell_count"])

    def test_the_recorded_rows_match_the_counts_they_are_summarised_by(self):
        """A summary that disagrees with the rows it summarises is the failure to catch here."""
        for path in probe_files():
            with self.subTest(file=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                answered = [r for r in record["rows"] if r.get("model_decision")]
                self.assertEqual(len(answered), record["answered"])
                correct = sum(
                    r["model_decision"] == r["expected_decision"] for r in answered)
                self.assertEqual(correct, record["correct"])
                # The per-row correctness flag must agree with recomputing it.
                self.assertEqual(sum(r["correct"] for r in answered), record["correct"])

    def test_the_unguarded_probe_agrees_with_the_ablation_it_precedes(self):
        """The probe is the same models unguarded, so it must match the ablation's `raw` arm.

        The previous version of this test asserted `correct / cell_count <= 1.0`, which is a
        tautology: the sibling test above already pins `correct <= answered` and
        `len(rows) == cell_count`. It never opened the ablation its own name invokes, so a probe
        file rewritten to a perfect 18/18 passed it. This compares the two, which is what the name
        claimed all along.

        The probe scores one draw per cell and the ablation ten, so they are not expected to be
        equal. They are expected to be the same quantity measured twice: a two-sigma-plus gap means
        one of the two files does not describe the run it claims to.
        """
        import math

        ablation = {}
        for path in Path("evidence/ablation").glob("*/ablation.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            ablation[record["model"]] = record["results"]["raw"]

        for path in probe_files():
            with self.subTest(file=path.name):
                record = json.loads(path.read_text(encoding="utf-8"))
                raw = ablation[record["model"]]

                probe_rate = record["correct"] / record["cell_count"]
                raw_rate = raw["correct"] / raw["total"]
                standard_error = math.sqrt(
                    raw_rate * (1 - raw_rate) / record["cell_count"]) or 1e-9
                self.assertLess(
                    abs(probe_rate - raw_rate) / standard_error, 3.0,
                    f"{record['model']}: probe scores {probe_rate:.1%} against {raw_rate:.1%} "
                    "unguarded in the ablation; these are meant to be the same measurement",
                )

                # The probe must also stay under every guarded arm, which is the claim it exists
                # to support: guards are what closes the gap, not the model.
                for arm in ("all_guards", "derived"):
                    guarded = json.loads(
                        (Path("evidence/ablation") / _directory_for(record["model"])
                         / "ablation.json").read_text(encoding="utf-8"))["results"][arm]
                    self.assertLessEqual(
                        probe_rate, guarded["correct"] / guarded["total"],
                        f"{record['model']}: unguarded probe beat the {arm} arm",
                    )

                # An unguarded model that took no unsafe action anywhere would undercut the whole
                # premise, so the two files must agree on whether that happened.
                self.assertEqual(
                    record["unsafe_decisions"] > 0, raw["unsafe_executions"] > 0,
                    f"{record['model']}: probe and ablation disagree on unsafe actions",
                )


if __name__ == "__main__":
    unittest.main()
