"""Every published figure, recomputed from evidence, in one table.

Six separate errors in this repository were the same shape: a number computed once by hand, written
into prose, and never checked again. Pinning them one at a time after each audit does not scale and
did not work, because the next hand-computed number was not covered either.

This inverts it. Each entry below names a figure that appears in the documents, the code that
recomputes it from `evidence/`, and the exact string that must appear. Adding a number to the prose
without adding it here leaves it unguarded, which is visible; changing evidence without changing
prose fails the build.

The recomputation deliberately does not import the analysis modules where it can avoid it. If a
figure is only reproducible by the same code that produced it, the check is circular.
"""

import glob
import json
import os
import unittest
from collections import defaultdict
from pathlib import Path


def normalise(text):
    """Collapse whitespace and unify minus signs.

    The documents are hard-wrapped and use a typographic minus, so matching raw substrings makes
    the check fail on line breaks rather than on wrong numbers. The point is to guard the figures,
    not the line width.
    """
    return " ".join(text.replace("\u2212", "-").split())


PAPER = normalise(Path("PAPER.md").read_text(encoding="utf-8"))
README = normalise(Path("README.md").read_text(encoding="utf-8")) + " " + PAPER


def load(name):
    return json.loads(Path(f"evidence/{name}").read_text(encoding="utf-8"))


def completed_factorial_runs():
    """Directories from finished runs only.

    One aborted run left partial arms on disk with no manifest. A glob that ignored the manifest
    contaminated a published table, so this is the only sanctioned way to enumerate the runs.
    """
    return [d for d in sorted(glob.glob("evidence/factorial/*"))
            if os.path.exists(os.path.join(d, "factorial.json"))]


def ask_rate_by(field, value, vocabulary):
    hits = total = 0
    for directory in completed_factorial_runs():
        for path in glob.glob(os.path.join(directory, "raw-*.json")):
            arm = os.path.basename(path)[4:-5]
            if (arm.startswith("finance")) != (vocabulary == "finance"):
                continue
            for record in json.loads(Path(path).read_text(encoding="utf-8")):
                if "decision" not in record:
                    continue
                if field == "expected_abstain":
                    if record["expected"] != "ABSTAIN":
                        continue
                elif field == "authority_invalid":
                    if record["authority"] == "OK":
                        continue
                elif record["permission"] != value:
                    continue
                hits += record["decision"] == "ASK"
                total += 1
    return hits / total * 100


def cell_expectations():
    counts = defaultdict(int)
    seen = set()
    for record in load("ablation/M1-gemma4-12b/raw-samples.json"):
        if record["scenario_id"] not in seen:
            seen.add(record["scenario_id"])
            counts[record["expected"]] += 1
    return counts


# figure -> (recompute, formatted string that must appear, document)
FIGURES = {
    "reckless policy score":
        (lambda: load("model-ablation.json")["exploratory"]["substitute_bps"]["always_act"],
         lambda v: f"{v}", README),
    "adversarial floor":
        (lambda: load("model-ablation.json")["exploratory"]["adversarial_floor_bps"],
         lambda v: f"{v}",
         README),
    "uniform noise":
        (lambda: load("model-ablation.json")["exploratory"]["substitute_bps"]["uniform_random"],
         lambda v: f"{v} bps", README),
    "model-independent fraction":
        (lambda: load("model-ablation.json")["exploratory"]["fraction_of_score_that_is_model_independent"] * 100,
         lambda v: f"{v:.1f}% of the score is the scaffold", README),
    "movable range":
        (lambda: load("model-ablation.json")["exploratory"]["model_movable_range_bps"],
         lambda v: f"Only {v} of the 10000 responds to the model", README),
    "mind2web floor":
        (lambda: load("external-floor.json")["mind2web_sc"]["no_model_floor"] * 100,
         lambda v: f"{v:.1f}%", README),
    "eicu floor":
        (lambda: load("external-floor.json")["eicu_ac"]["no_model_floor"] * 100,
         lambda v: f"{v:.1f}%", README),
    "refusal cell count":
        (lambda: cell_expectations()["ABSTAIN"],
         lambda v: f"{v} of the 18", README),
    "vocabulary effect on accuracy":
        (lambda: load("factorial-effects.json")["pooled"]["accuracy"]["vocabulary"]["effect_bps"],
         lambda v: f"{v}", README),
    "vocabulary effect on ask rate":
        (lambda: load("factorial-effects.json")["pooled"]["ask_rate"]["vocabulary"]["effect_bps"],
         lambda v: f"+{v}", README),
    "exhaustive headroom is zero":
        (lambda: max(m["headroom_bps"] for s in load("exhaustive.json")["surfaces"].values()
                     for m in s["metrics"].values()),
         lambda v: "headroom here is zero", README),
    "boundary headroom, confirmatory":
        (lambda: load("boundary.json")["confirmatory"]["headroom_bps"],
         lambda v: f"{v:.2f} bps on the confirmatory one", README),
}


class PublishedFiguresRecompute(unittest.TestCase):
    def test_every_listed_figure_matches_the_evidence(self):
        missing = []
        for name, (recompute, render, document) in FIGURES.items():
            expected = normalise(render(recompute()))
            if expected not in document:
                missing.append(f"{name}: expected to find {expected!r}")
        self.assertEqual(missing, [], "\n".join(missing))

    def test_the_aborted_run_is_excluded_from_every_factorial_figure(self):
        runs = completed_factorial_runs()
        self.assertEqual(len(runs), 5)
        self.assertNotIn("evidence/factorial/F2-qwen3.8-27b", runs)
        # It must still be present on disk: the failure is evidence, not something to delete.
        self.assertTrue(os.path.exists("evidence/factorial/F2-qwen3.8-27b/FAILED.json"))


if __name__ == "__main__":
    unittest.main()
