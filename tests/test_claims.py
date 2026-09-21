"""Claim-boundary tests: the README may not drift from the evidence or reintroduce old overclaims.

These fail the build rather than relying on manual inspection.
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
PAPER = (ROOT / "PAPER.md").read_text()
DOCS = README + PAPER + (ROOT / "docs" / "limitations.md").read_text()


class ClaimBoundaryTests(unittest.TestCase):
    # Denials are the point of these documents, so only affirmative constructions may fail.
    BANNED_AFFIRMATIVE = (
        r"\bis a self-improving", r"\bbuilt a self-improving", r"\bdemonstrates self-improv",
        r"\blearns to be safe", r"\bthe model became safer", r"\bteaches itself",
        r"\bcannot be influenced by", r"\bis fully isolated",
    )

    def test_no_affirmative_self_improvement_claim(self) -> None:
        for pattern in self.BANNED_AFFIRMATIVE:
            self.assertNotRegex(DOCS.lower(), pattern, f"affirmative claim reintroduced: {pattern}")

    def test_the_denials_are_still_present(self) -> None:
        low = " ".join(DOCS.lower().split())
        self.assertIn("not a self-improving system", low)
        self.assertRegex(low, r"scoring independence, not containment|not containment")

    def test_novelty_is_not_claimed_for_the_metric(self) -> None:
        self.assertIn("virtual-best-solver", README.lower().replace(" ", "-") + README.lower())
        self.assertRegex(README, r"(?i)not a new metric|already exists|not new")

    def test_degenerate_baseline_is_stated_near_the_headline(self) -> None:
        self.assertIn("7778", README)

    def test_configuration_space_is_stated_in_full(self) -> None:
        self.assertIn("328", README)
        self.assertIn("328", PAPER)

    def test_reported_headroom_matches_the_evidence(self) -> None:
        data = json.loads((ROOT / "evidence" / "exhaustive.json").read_text())
        for surface, block in data["surfaces"].items():
            for metric, a in block["metrics"].items():
                self.assertEqual(a["headroom_bps"], 0, f"{surface}/{metric} moved off zero")
                self.assertTrue(
                    a["shared_config_is_optimal_for_every_model"],
                    f"{surface}/{metric}: a shared optimum no longer covers every model",
                )

    def test_every_model_figure_in_the_readme_appears_in_the_evidence(self) -> None:
        """Every four-digit basis-point figure in a README table must exist in evidence.

        The sources are enumerated rather than globbed for any integer, so that adding a table
        without adding its evidence fails here instead of passing by accident.
        """
        committed = set()
        for f in (ROOT / "evidence").rglob("*.json"):
            if f.name in ("ablation.json", "confirm.json"):
                for policy in json.loads(f.read_text())["results"].values():
                    committed.add(policy["bps"])
            elif f.name == "vocabulary-sweep.json":
                for arm in json.loads(f.read_text())["vocabularies"].values():
                    if arm["accuracy_bps"] is not None:
                        committed.add(arm["accuracy_bps"])
            elif f.name == "vocabulary-effects.json":
                pooled = json.loads(f.read_text())["pooled"]
                for kind in pooled.values():
                    committed.update(kind["per_vocabulary_bps"].values())
            elif f.name == "floor-survey.json":
                # Item counts, not basis points, but they are four digits in a table and this
                # check cannot tell the difference from the prose alone.
                for entry in json.loads(f.read_text())["benchmarks"]:
                    committed.add(entry["items"])
            elif f.name == "model-ablation.json":
                for surface in ("exploratory", "confirmatory"):
                    block = json.loads(f.read_text())[surface]
                    committed.update(block["substitute_bps"].values())
                    committed.update(block["observed_bps"].values())
        quoted = {
            int(n) for n in re.findall(r"\|\s*(\d{4})\s*\|", README)
            if 1000 <= int(n) <= 10000
        }
        missing = quoted - committed - {7778}  # 7778 is the always-refuse reference
        self.assertFalse(missing, f"README quotes figures absent from evidence: {sorted(missing)}")

    def test_preregistration_states_its_custody_limits(self) -> None:
        prereg = " ".join((ROOT / "PREREGISTRATION.md").read_text().lower().split())
        self.assertIn("not third-party verification", prereg)
        self.assertIn("no external service timestamped", prereg)
        self.assertIn("not by an independent person", prereg)


if __name__ == "__main__":
    unittest.main()
