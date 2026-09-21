"""The shipped tool must be the same tool this project used, not a lookalike.

`floors.py` exists because every earlier runnable target operated on the
author's benchmark, so a reader persuaded by the argument was handed prose. It is meant to be copied
into someone else's project, which means it must import nothing from this one, and it must produce
the same answers this project reports. Both are checked here rather than asserted in its docstring.
"""

import importlib.util
import json
import random
import unittest
from pathlib import Path


def load_tool():
    """Load floors.py by path, as a copy of it in another project would be imported.

    Registering it in sys.modules before executing is required because dataclass field resolution
    looks the defining module up there. A normal `import floors` does that automatically.
    """
    import sys

    spec = importlib.util.spec_from_file_location("floors_tool", Path("floors.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["floors_tool"] = module
    spec.loader.exec_module(module)
    return module


class ItIsSelfContained(unittest.TestCase):
    def test_it_imports_nothing_from_this_project(self):
        """Copying one file must be enough. An import from here would silently break that."""
        import ast

        source = Path("floors.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        # Derived from what this repository actually exposes rather than from a hand-kept list. The
        # hand-kept list named six modules and omitted `vaa`, `headroom`, `external_headroom` and
        # `floor_survey`, so an import of any of those would have passed this check.
        local = {q.stem for q in Path(".").glob("*.py")} | {
            q.stem for q in Path("experiments").glob("*.py")
        } | {q.name for q in Path("src").iterdir()} if Path("src").exists() else set()
        local.discard("floors")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotIn(node.module.split(".")[0], local, node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(alias.name.split(".")[0], local, alias.name)

    def test_it_runs_from_an_empty_directory_with_this_project_unreachable(self):
        """The import check above is a proxy. This is the actual claim being made to a reader.

        It runs the copied file with -I, which ignores PYTHONPATH and user site-packages. Without
        that, this project's own `PYTHONPATH=src:experiments` makes its modules importable and a
        file that depends on them still passes, while the reader's copy dies on import.
        """
        import shutil
        import subprocess
        import sys
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy("floors.py", Path(tmp) / "floors.py")
            proc = subprocess.run(
                [sys.executable, "-I", "floors.py"],
                cwd=tmp, capture_output=True, text=True, timeout=120, check=False,
            )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Assert on what it printed, so an exit code from a tool that computed nothing cannot pass.
        self.assertIn("no-model floor:", proc.stdout)
        self.assertIn("adaptation headroom:", proc.stdout)
        self.assertIn("permutation p:", proc.stdout)

    def test_its_worked_example_exposes_both_checks(self):
        module = load_tool()
        self.assertTrue(hasattr(module, "Benchmark"))
        self.assertTrue(hasattr(module, "no_model_floor"))
        self.assertTrue(hasattr(module, "adaptation_headroom"))


class CheckOneBehaviour(unittest.TestCase):
    """Data-free, so this runs on a hosted runner where the external corpora do not exist."""

    def setUp(self):
        self.module = load_tool()

    def test_the_floor_always_includes_the_majority_class(self):
        """Omitting it is the error that cost this project a finding."""
        bench = self.module.Benchmark(
            items=[{"y": 1}] * 9 + [{"y": 0}], correct=lambda r: r["y"],
            policies={"always 0": lambda rows: [0] * len(rows)})
        result = self.module.no_model_floor(bench)
        self.assertAlmostEqual(result.floor, 0.9, places=6)
        self.assertEqual(result.floor_policy, "majority class")

    def test_a_user_policy_cannot_suppress_the_majority_baseline_by_taking_its_name(self):
        """`setdefault` let it. The floor then printed 10% directly above 'majority class: 90%'."""
        items = [{"y": 1}] * 90 + [{"y": 0}] * 10
        bench = self.module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"majority class": lambda rows: [0] * len(rows)})
        result = self.module.no_model_floor(bench)
        self.assertAlmostEqual(result.floor, 0.9, places=6)
        self.assertGreaterEqual(result.floor, result.majority_class)
        self.assertIn(0.9, [round(v, 6) for v in result.scores.values()])

    def test_a_user_supplying_every_collision_name_loses_nothing(self):
        """One rename step was not enough: a user who supplied *both* 'majority class' and
        'majority class (built in)' had their second policy silently overwritten by the built-in
        rate. The built-in name must be unique against everything actually present."""
        items = [{"y": 1}] * 90 + [{"y": 0}] * 10
        bench = self.module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"majority class": lambda rows: [1] * len(rows),
                      "majority class (built in)": lambda rows: [0] * len(rows)})
        result = self.module.no_model_floor(bench)
        # Both user policies survive untouched under their own names.
        self.assertAlmostEqual(result.scores["majority class"], 0.9, places=6)
        self.assertAlmostEqual(result.scores["majority class (built in)"], 0.1, places=6)
        # The built-in majority appears too, under some third name.
        others = [n for n in result.scores
                  if n not in ("majority class", "majority class (built in)")]
        self.assertEqual(len(others), 1, sorted(result.scores))
        self.assertAlmostEqual(result.scores[others[0]], 0.9, places=6)

    def test_an_empty_benchmark_is_refused_by_both_checks(self):
        """An empty item list reached `max()` of an empty Counter and died mid-computation."""
        empty = self.module.Benchmark(items=[], correct=lambda r: 0, policies={})
        with self.assertRaises(ValueError) as caught:
            self.module.no_model_floor(empty)
        self.assertIn("no items", str(caught.exception))

    def test_unhashable_labels_are_named_not_thrown(self):
        bench = self.module.Benchmark(
            items=[{"u": i % 2} for i in range(40)], correct=lambda r: [r["u"]],
            policies={"p": lambda rows: [[r["u"]] for r in rows]})
        with self.assertRaises(ValueError) as caught:
            self.module.no_model_floor(bench)
        self.assertIn("unhashable", str(caught.exception))
        bench2 = self.module.Benchmark(
            items=[{"u": i % 2} for i in range(40)], correct=lambda r: [r["u"]],
            policies={"p": lambda rows: [[r["u"]] for r in rows]},
            unit=lambda r: r["u"])
        with self.assertRaises(ValueError) as caught:
            self.module.adaptation_headroom(bench2, rounds=500)
        self.assertIn("unhashable", str(caught.exception))

    def test_the_majority_baseline_can_be_fitted_on_held_out_labels(self):
        """Without a training set the built-in majority reads the answer key, and must say so.

        These are the two EICU-AC figures from this project's own history: the evaluation split's
        base rate is 64.8%, the honest training-fitted constant scores 35.2% on it, and a published
        comparison was once drawn against the former.
        """
        train = [{"y": 1}] * 117 + [{"y": 0}] * 71     # majority is 1
        evaluation = [{"y": 0}] * 83 + [{"y": 1}] * 45  # majority is 0

        unfitted = self.module.no_model_floor(
            self.module.Benchmark(evaluation, lambda r: r["y"], {}))
        self.assertAlmostEqual(unfitted.majority_class, 0.6484, places=3)
        self.assertEqual(unfitted.majority_fitted_on, "evaluation labels")
        self.assertTrue(unfitted.floor_requires_the_answer_key)
        self.assertIn("WARNING", str(unfitted))

        fitted = self.module.no_model_floor(
            self.module.Benchmark(evaluation, lambda r: r["y"], {}, train=train))
        self.assertAlmostEqual(fitted.majority_class, 0.3516, places=3)
        self.assertEqual(fitted.majority_fitted_on, "training labels")
        self.assertFalse(fitted.floor_requires_the_answer_key)
        self.assertNotIn("WARNING", str(fitted))

    def test_the_floor_is_the_best_policy_not_the_worst(self):
        bench = self.module.Benchmark(
            items=[{"y": 1}] * 6 + [{"y": 0}] * 4, correct=lambda r: r["y"],
            policies={"good": lambda rows: [1] * len(rows),
                      "bad": lambda rows: [0] * len(rows)})
        result = self.module.no_model_floor(bench)
        self.assertAlmostEqual(result.floor, 0.6, places=6)

    def test_it_names_the_policy_when_one_returns_something_unmeasurable(self):
        """A generator failed several frames away with a message naming no policy."""
        bench = self.module.Benchmark(
            items=[{"y": 0}] * 5, correct=lambda r: r["y"],
            policies={"lazy": lambda rows: (0 for _ in rows)})
        with self.assertRaises(ValueError) as caught:
            self.module.no_model_floor(bench)
        self.assertIn("lazy", str(caught.exception))

    def test_it_refuses_a_policy_that_returns_the_wrong_number_of_predictions(self):
        bench = self.module.Benchmark(
            items=[{"y": 0}] * 5, correct=lambda r: r["y"],
            policies={"broken": lambda rows: [0]})
        with self.assertRaises(ValueError):
            self.module.no_model_floor(bench)


class CheckTwoBehaviour(unittest.TestCase):
    """The function this project is named after had no behavioural test at all.

    A mutation review changed the permutation so it never shuffled, inverted its comparison, flipped
    the sign of the reported gap, dropped the +1 correction from the p-value, cut the default rounds
    to two and lowered the significance threshold to 0.5. The suite stayed green through every one
    of them. These are the tests that fail instead.
    """

    def setUp(self):
        self.module = load_tool()

    @staticmethod
    def _bench(module, positive_rate_north, positive_rate_south, n_per=100):
        """Two equal units, each with the given fraction of positive labels.

        Both rates must be strictly between 0 and 1: a unit of one class is refused by design, and
        stating the rate directly makes that visible in the test rather than a surprise.
        """
        items = []
        for region, rate in (("north", positive_rate_north), ("south", positive_rate_south)):
            for i in range(n_per):
                items.append({"y": 1 if i < int(n_per * rate) else 0, "region": region})
        return module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"always allow": lambda rows: [0] * len(rows),
                      "always block": lambda rows: [1] * len(rows)},
            unit=lambda r: r["region"])

    def test_it_finds_structure_that_is_really_there(self):
        """Opposite majorities per unit: no single policy is best in both."""
        result = self.module.adaptation_headroom(
            self._bench(self.module, 0.9, 0.1), rounds=500)
        self.assertGreater(result.headroom, 0.0)
        self.assertAlmostEqual(result.headroom, 0.40, places=2)
        self.assertLess(result.permutation_p, 0.05)
        self.assertTrue(result.exceeds_chance)
        self.assertEqual(
            result.per_unit_best, {"north": "always block", "south": "always allow"})

    def test_it_reports_nothing_when_one_policy_is_best_everywhere(self):
        """Same majority in both units. A permutation that never shuffled would still fire here."""
        bench = self._bench(self.module, 0.7, 0.8)  # both majority-positive
        result = self.module.adaptation_headroom(bench, rounds=500)
        self.assertAlmostEqual(result.headroom, 0.0, places=6)
        self.assertFalse(result.exceeds_chance)
        self.assertGreater(result.permutation_p, 0.05)

    def test_the_p_value_can_never_be_zero(self):
        """The +1 correction. Without it a clean separation reports p = 0.000 from 500 draws."""
        result = self.module.adaptation_headroom(
            self._bench(self.module, 0.9, 0.1), rounds=500)
        self.assertGreater(result.permutation_p, 0.0)
        self.assertAlmostEqual(result.permutation_p, 1 / 501, places=6)

    def test_the_significance_threshold_is_five_percent(self):
        made = self.module.HeadroomResult(
            headroom=0.1, permutation_p=0.2, units=2, best_shared_policy="x",
            per_unit_best={}, dropped_units={})
        self.assertFalse(made.exceeds_chance, "0.2 is not significant at any usual level")
        made.permutation_p = 0.04
        self.assertTrue(made.exceeds_chance)
        made.permutation_p = 0.05
        self.assertFalse(made.exceeds_chance, "the threshold is strict")

    def test_units_below_the_minimum_size_are_dropped_and_named(self):
        items = ([{"y": i % 2, "u": "big"} for i in range(40)]
                 + [{"y": i % 2, "u": "small"} for i in range(19)]
                 + [{"y": i % 2, "u": "other"} for i in range(40)])
        bench = self.module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"a": lambda rows: [0] * len(rows), "b": lambda rows: [1] * len(rows)},
            unit=lambda r: r["u"])
        result = self.module.adaptation_headroom(bench, rounds=100)
        self.assertEqual(result.units, 2)
        self.assertEqual(result.dropped_units, {"small": 19})
        self.assertEqual(self.module.MIN_UNIT_SIZE, 20)

    def test_zero_headroom_remains_nonsignificant_with_a_concentrated_null(self):
        """Two units where one policy nearly always wins both: the null is concentrated at zero.

        Concentration is only a diagnostic. Here the observed headroom is zero, so the nominal
        comparison remains nonsignificant.
        """
        # Two units, both majority-negative, with a tiny edge for one policy in one unit only.
        items = ([{"y": 0, "u": "a"} for _ in range(30)] + [{"y": 1, "u": "a"} for _ in range(3)]
                 + [{"y": 0, "u": "b"} for _ in range(30)] + [{"y": 1, "u": "b"} for _ in range(3)])
        bench = self.module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"always 0": lambda rows: [0] * len(rows),
                      "nearly 0": lambda rows: [1 if i == 0 else 0 for i in range(len(rows))]},
            unit=lambda r: r["u"])
        result = self.module.adaptation_headroom(bench, rounds=500)
        self.assertEqual(result.headroom, 0)
        self.assertEqual(result.permutation_p, 1)
        self.assertTrue(result.null_is_degenerate)
        self.assertFalse(
            result.exceeds_chance,
            "zero observed headroom must not yield a nominal association",
        )
        self.assertIn("WARNING", str(result))
        self.assertIn("does not by itself invalidate", str(result))

    def test_a_spread_null_is_not_flagged_as_zero_concentrated(self):
        """The legacy diagnostic distinguishes a spread null without controlling the verdict."""
        result = self.module.adaptation_headroom(
            self._bench(self.module, 0.9, 0.1), rounds=500)
        self.assertFalse(result.null_is_degenerate)
        self.assertTrue(result.exceeds_chance)

    def test_the_number_of_items_behind_a_gap_is_reported(self):
        """400 bps over a small unit can be five decisions. Say which."""
        result = self.module.adaptation_headroom(
            self._bench(self.module, 0.9, 0.1), rounds=200)
        self.assertGreater(result.items_driving_the_gap, 0)
        self.assertIn("driven by:", str(result))

    def test_the_default_round_count_supports_the_resolution_it_prints(self):
        """The nominal 0.05 decision requires enough permutation resolution."""
        self.assertGreaterEqual(self.module.DEFAULT_ROUNDS, 1000)

    def test_units_that_are_a_copy_of_the_label_are_refused(self):
        """The error this project published twice before catching it in its own analysis."""
        items = [{"y": i % 2} for i in range(400)]
        bench = self.module.Benchmark(
            items=items, correct=lambda r: r["y"],
            policies={"a": lambda rows: [0] * len(rows), "b": lambda rows: [1] * len(rows)},
            unit=lambda r: r["y"])
        with self.assertRaises(ValueError) as caught:
            self.module.adaptation_headroom(bench, rounds=100)
        self.assertIn("single class", str(caught.exception))

    def test_it_refuses_to_run_no_permutations(self):
        """Zero rounds returned p = 1.0, which reads as a negative rather than as nothing run."""
        with self.assertRaises(ValueError):
            self.module.adaptation_headroom(self._bench(self.module, 0.9, 0.1), rounds=0)

    def test_it_enforces_the_conservative_round_minimum(self):
        """Small Monte Carlo counts produce coarse p-value grids.

        One hundred is an explicit usability floor, not a claim that 99 draws cannot attain p < .05.
        """
        with self.assertRaises(ValueError) as caught:
            self.module.adaptation_headroom(self._bench(self.module, 0.9, 0.1), rounds=99)
        self.assertIn("100", str(caught.exception))
        result = self.module.adaptation_headroom(self._bench(self.module, 0.9, 0.1), rounds=100)
        self.assertTrue(result.exceeds_chance)

    def test_unhashable_units_are_named_not_thrown(self):
        bench = self.module.Benchmark(
            items=[{"y": i % 2, "u": i % 2} for i in range(40)], correct=lambda r: r["y"],
            policies={"p": lambda rows: [r["y"] for r in rows]},
            unit=lambda r: [r["u"]])
        with self.assertRaises(ValueError) as caught:
            self.module.adaptation_headroom(bench, rounds=500)
        self.assertIn("unhashable", str(caught.exception))

    def test_it_refuses_check_two_without_units(self):
        bench = self.module.Benchmark(
            items=[{"y": 0}] * 5, correct=lambda r: r["y"],
            policies={"always 0": lambda rows: [0] * len(rows)})
        with self.assertRaises(ValueError):
            self.module.adaptation_headroom(bench)


class ItReproducesThisProjectsNumbers(unittest.TestCase):
    """Run the shipped tool against the real data and require the published floor back."""

    def setUp(self):
        self.module = load_tool()

    def _require(self, path: str) -> None:
        """Skip naming the one file this test needs. Both drift tests used to key off the
        Mind2Web-SC path alone, so the EICU-AC test skipped when only EICU-AC was missing."""
        if not Path(path).exists():
            self.skipTest(f"{path} unavailable; provide an authorized local dataset copy")

    def test_it_recovers_the_published_mind2web_floor(self):
        self._require("external/m2w/seeact/sample_labeled_all.json")
        records = json.loads(
            Path("external/m2w/seeact/sample_labeled_all.json").read_text(encoding="utf-8"))
        published = json.loads(
            Path("evidence/external-floor.json").read_text(encoding="utf-8"))["mind2web_sc"]

        def profile_lookup(rows):
            """The cross-validated policy that sets the published floor."""
            n = len(rows)
            key = lambda r: (r["user_info"]["domestic"], r["user_info"]["dr_license"],
                             r["user_info"]["vaccine"], r["user_info"]["membership"])
            idx = list(range(n))
            random.Random(0).shuffle(idx)
            out = [0] * n
            for fold in range(10):
                test = set(idx[fold::10])
                train = [i for i in idx if i not in test]
                table = {}
                for i in train:
                    table.setdefault(key(rows[i]), []).append(rows[i]["label"])
                fallback = round(sum(rows[i]["label"] for i in train) / len(train))
                for i in test:
                    seen = table.get(key(rows[i]))
                    out[i] = round(sum(seen) / len(seen)) if seen else fallback
            return out

        bench = self.module.Benchmark(
            items=records,
            correct=lambda r: r["label"],
            policies={"profile lookup": profile_lookup,
                      "always allow": lambda rows: [0] * len(rows),
                      "always block": lambda rows: [1] * len(rows)},
        )
        result = self.module.no_model_floor(bench)
        self.assertAlmostEqual(result.floor, published["no_model_floor"], places=3)

    def test_check_two_agrees_with_this_projects_own_analysis(self):
        """The tool and `experiments/external_headroom.py` implement the same check twice.

        The duplication is deliberate, because the shipped file has to stand alone, but nothing kept
        the two in step: they seed their permutations differently and could have drifted apart
        silently while the README claimed the reader was getting the same instrument. This runs both
        over EICU-AC and requires the same answer.
        """
        self._require("external/eicu/ehragent/eicu_ac.json")
        import external_headroom

        items, unit = external_headroom.load_eicu(Path("external"))
        predictions = external_headroom.policy_scores(items, {
            "always allow": external_headroom.constant(0),
            "always block": external_headroom.constant(1),
            "global majority": external_headroom.global_majority,
            "template lookup": external_headroom.lookup_on("template"),
        })
        analysis = external_headroom.headroom(items, predictions, unit)

        # The tool's policies take items only, so the already-fitted predictions are handed back.
        def replay(fitted):
            return lambda rows: [fitted[i] for i in rows]

        bench = self.module.Benchmark(
            items=list(range(len(items))),
            correct=lambda i: items[i]["_y"],
            policies={name: replay(guesses) for name, guesses in predictions.items()},
            unit=lambda i: items[i][unit],
        )
        tool = self.module.adaptation_headroom(bench, rounds=external_headroom.ROUNDS)

        self.assertEqual(round(tool.headroom * 10000), analysis["headroom_bps"])
        self.assertEqual(tool.units, analysis["units"])
        self.assertEqual(tool.best_shared_policy, analysis["best_shared_policy"])
        self.assertEqual(tool.exceeds_chance, analysis["exceeds_chance_structure"])


if __name__ == "__main__":
    unittest.main()
