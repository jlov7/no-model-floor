"""No analysis may enumerate evidence runs by hand.

Twice this project counted an aborted run as a whole one, because an analysis globbed for files and
took their presence as proof that a run had happened. The first cost half a point in a published
table; the second, written hours later in fresh code, would have moved a figure by 167 basis points.
Both were caught by someone remembering the previous one.

A property that depends on remembering is not a property. `experiments/runs.py` is the only
sanctioned way to enumerate runs, and this test fails the build if a module goes around it, so the
third instance cannot be written rather than merely being likely to be spotted.

The frozen experiments are exempt and must stay exactly as preregistered.
"""

import ast
import unittest
from pathlib import Path

EXPERIMENTS = Path("experiments")

# Frozen in the public checksum manifest. Changing these voids the confirmatory run.
FROZEN = {"ablation.py", "probe.py", "sweep.py"}

# Modules that read a run's own output rather than enumerating runs across directories.
# Modules that produce a run rather than reading across runs, or that read no evidence at all.
# `headroom.py` lives outside experiments/ and is deliberately standalone: the README offers it to
# a sceptic precisely so they need not trust anything else in the tree, so it may not import from
# here. It is covered instead by asserting its own claims and by a parity test against the analysis
# it summarises (tests/test_parity.py).
NOT_AGGREGATORS = {
    "runs.py",
    "surface_v2.py",
    "factorial.py",
    "vocabulary_sweep.py",
    "confirm.py",
    "theorem.py",
    "verify_theorem.py",
    "reproduce_guard.py",
    "external_floor.py",
}


# Modules that enumerate third-party dataset files under external/ rather than evidence runs. The
# completeness rules here are about evidence runs, so these are exempt from them, but they must not
# reach into evidence run directories either, which is what the test below checks.
EXTERNAL_READERS = {"floor_survey.py", "external_floor.py", "external_headroom.py"}


def modules():
    for path in sorted(EXPERIMENTS.glob("*.py")):
        if (
            path.name not in FROZEN
            and path.name not in NOT_AGGREGATORS
            and path.name not in EXTERNAL_READERS
        ):
            yield path


class NoHandRolledEnumeration(unittest.TestCase):
    def test_no_analysis_enumerates_directories_by_hand(self):
        """Any directory enumeration, not merely a glob with a slash in a literal pattern.

        The first version of this test inspected only `.glob()` calls whose pattern was a string
        constant containing a slash. The original rule was defeated three ways: `os.listdir`, a
        pattern held in a variable, and an f-string pattern. All three counted partial runs and the
        suite stayed green. The rule is now about the operation rather than about how its argument
        is spelled.
        """
        banned = {"glob", "rglob", "iterdir", "listdir", "walk", "scandir"}
        offenders = []
        for path in modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    name = node.func.id
                if name in banned:
                    offenders.append(f"{path.name}: {name}(...)")
        self.assertEqual(
            offenders,
            [],
            "enumerate runs through experiments/runs.py, which rejects incomplete ones:\n"
            + "\n".join(offenders),
        )

    def test_anything_reaching_into_a_run_directory_uses_the_shared_enumerator(self):
        """Delegating to a module that already enumerates correctly is fine.

        The rule is about constructing paths into run directories, not about reading evidence.
        `uncertainty.py`, for instance, calls a loader in `exhaustive.py` and never touches a
        directory itself, so requiring it to import the enumerator would be noise.
        """
        missing = []
        for path in modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Import detection via AST, not substring: a comment mentioning the module used to
            # satisfy the old check.
            imports_runs = any(
                isinstance(node, ast.ImportFrom) and node.module == "runs"
                for node in ast.walk(tree)
            )
            mentions_artefacts = any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("raw-")
                for node in ast.walk(tree)
            )
            if mentions_artefacts and not imports_runs:
                missing.append(path.name)
        self.assertEqual(missing, [], f"these build paths into run directories unaided: {missing}")

    def test_the_frozen_experiments_are_still_exempt_and_present(self):
        for name in FROZEN:
            self.assertTrue((EXPERIMENTS / name).exists(), name)


class TheEnumeratorRejectsIncompleteRuns(unittest.TestCase):
    def setUp(self):
        import sys

        sys.path.insert(0, "experiments")

    def test_a_directory_without_a_manifest_is_rejected_with_a_reason(self):
        import tempfile

        from runs import completed

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "aborted").mkdir()
            (root / "aborted" / "raw-samples.json").write_text("[]")
            accepted, rejected = completed(root, "ablation.json")
            self.assertEqual(accepted, [])
            self.assertEqual(len(rejected), 1)
            self.assertIn("did not finish", rejected[0]["reason"])

    def test_a_missing_artefact_is_rejected_with_a_reason(self):
        import tempfile

        from runs import completed

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "partial").mkdir()
            (root / "partial" / "ablation.json").write_text("{}")
            accepted, rejected = completed(root, "ablation.json", artefacts=["raw-samples.json"])
            self.assertEqual(accepted, [])
            self.assertIn("missing artefacts", rejected[0]["reason"])

    def test_evidence_runs_raises_rather_than_averaging_over_a_partial_run(self):
        import tempfile

        from runs import evidence_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good").mkdir()
            (root / "good" / "ablation.json").write_text("{}")
            (root / "good" / "raw-samples.json").write_text("[]")
            (root / "aborted").mkdir()
            (root / "aborted" / "raw-samples.json").write_text("[]")
            with self.assertRaises(ValueError) as caught:
                evidence_runs(root)
            self.assertIn("aborted", str(caught.exception))

    def test_a_run_holding_fewer_draws_than_it_declares_is_rejected(self):
        """The draw-count guard, exercised through its public entry point.

        `completed()` only checks that `raw-samples.json` exists; the actual truncation check lives
        in the `require` callback `_draws_match_the_manifest` that `evidence_runs` wires in. A run
        whose raw-samples file holds fewer draws than its own manifest declares must be rejected,
        not silently averaged over, so this builds a temporary evidence tree with exactly that
        mismatch and requires `evidence_runs` to raise.
        """
        import json
        import tempfile

        from runs import evidence_runs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "truncated-model"
            run_dir.mkdir()
            manifest = {
                "model": "truncated-model",
                "draws_scored": 10,
                "distinct_cells": 2,
                "off_contract_responses": 0,
            }
            (run_dir / "ablation.json").write_text(json.dumps(manifest))
            draws = [
                {"scenario_id": "approved-valid-high", "decision": "ACT", "confidence_bps": 9000}
                for _ in range(5)
            ]
            (run_dir / "raw-samples.json").write_text(json.dumps(draws))
            with self.assertRaises(ValueError) as caught:
                evidence_runs(root)
            self.assertIn("holds 5 draws but declares 10", str(caught.exception))

    def test_the_real_evidence_directories_all_qualify(self):
        from runs import evidence_runs

        for root in ("evidence/ablation", "evidence/confirmatory"):
            with self.subTest(root=root):
                self.assertEqual(len(evidence_runs(root)), 5)


if __name__ == "__main__":
    unittest.main()


class TheExemptionsAreStillJustified(unittest.TestCase):
    """An exemption list rots silently unless something checks the reason still holds."""

    def test_exempt_modules_do_not_enumerate_directories(self):
        """A module exempted as a producer writes into one output directory and reads none.

        Mentioning a manifest name is not the test, because a producer writes one. Enumerating
        directories is, because that is how a reader finds runs, and it is the operation the
        exemption is a promise not to perform.
        """
        banned = {"glob", "rglob", "iterdir", "listdir", "walk", "scandir"}
        offenders = []
        for name in NOT_AGGREGATORS - {"runs.py"} - EXTERNAL_READERS:
            path = EXPERIMENTS / name
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                called = None
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    called = node.func.attr
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    called = node.func.id
                if called in banned:
                    offenders.append(f"{name}: {called}(...)")
        self.assertEqual(
            offenders,
            [],
            f"these are exempted as producers but enumerate directories: {offenders}",
        )

    def test_the_frozen_list_matches_the_one_ci_enforces(self):
        """The checked manifest contains every frozen experiment and CI invokes its verifier."""
        ci = Path("scripts/ci.sh").read_text(encoding="utf-8")
        manifest = Path("frozen/PREREGISTERED_V1.sha256").read_text(encoding="utf-8")
        self.assertIn("scripts/verify-frozen.py", ci)
        for name in FROZEN:
            self.assertIn(f"experiments/{name}", manifest, f"{name} is absent from the manifest")

    def test_external_readers_never_reach_into_evidence_runs(self):
        """They may enumerate third-party data; they may not enumerate evidence runs.

        The distinction matters because the completeness rules exist to stop a partial *run* being
        counted. A module reading someone else's dataset is not doing that, but it must not quietly
        start reading runs either.
        """
        offenders = []
        for name in EXTERNAL_READERS:
            path = EXPERIMENTS / name
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value
                    in (
                        "ablation.json",
                        "confirm.json",
                        "factorial.json",
                        "vocabulary-sweep.json",
                        "raw-samples.json",
                    )
                ):
                    offenders.append(f"{name}: references {node.value}")
        self.assertEqual(
            offenders, [], f"external readers must not read evidence runs: {offenders}"
        )
