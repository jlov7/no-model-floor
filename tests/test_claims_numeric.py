"""Pin the prose figures that an earlier draft got wrong.

Three numbers in README.md and PAPER.md disagreed with the evidence behind them, and a fourth
sentence asserted something the evidence contradicted. All four survived a claims test whose whole
purpose is to stop exactly that, because it checked language rather than arithmetic. These
recompute the figures and require the documents to contain them.
"""

import json
import re
import unittest
from pathlib import Path

README = Path("README.md").read_text(encoding="utf-8")
PAPER = Path("PAPER.md").read_text(encoding="utf-8")
BOTH = README + PAPER


def live_claims(text: str) -> str:
    """The document minus its errata, which must be free to quote what it is retracting.

    Without this, banning a retracted phrase would also ban saying that it was retracted, which
    would push the correction out of the document instead of into it.
    """
    return re.split(r"^## Errata$", text, flags=re.MULTILINE)[0]


LIVE = live_claims(README) + live_claims(PAPER)


class FiguresMatchTheEvidence(unittest.TestCase):
    def test_optimum_count_range_is_the_one_in_the_evidence(self):
        ex = json.loads(Path("evidence/exhaustive.json").read_text(encoding="utf-8"))
        counts = [
            v
            for block in ex["surfaces"].values()
            for metric in block["metrics"].values()
            for v in metric["per_model_optimum_count"].values()
        ]
        self.assertIn(f"Between {min(counts)} and {max(counts)} of the 328", BOTH)
        # The wrong lower bound came from ignoring the confirmatory surface.
        self.assertNotIn("Between 39 and 160", LIVE)

    def test_clearance_above_the_floor_spans_every_method_not_just_the_best(self):
        f = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))
        pub = f["published_label_prediction_accuracy"]
        # EICU-AC's published figures are scored on all 316 records, so the clearance has to be
        # measured against the full-set floor. Using the 128-item valid-split floor here compared
        # across evaluation sets, which is the defect this test now guards against.
        floor_key = {"mind2web_sc": "no_model_floor", "eicu_ac": "full_set_no_model_floor"}
        gaps = [
            (score - f[key][floor_key[key]]) * 100
            for key in ("mind2web_sc", "eicu_ac")
            for score in pub[key].values()
            if score > f[key][floor_key[key]]
        ]
        self.assertIn(f"{min(gaps):.1f} and {max(gaps):.1f} points", BOTH)
        self.assertNotIn("clear it by 28 to 40", LIVE)
        self.assertNotIn("19.5 and 40.1", LIVE)
        # Measured against the valid-split floor, i.e. across evaluation sets.
        self.assertNotIn("19.5 and 33.9", LIVE)

    def test_the_model_independent_fractions_are_not_stated_as_their_complements(self):
        f = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))
        for key in ("mind2web_sc", "eicu_ac"):
            floor = f[key]["no_model_floor"] * 100
            complement = 100 - floor
            self.assertIn(f"{floor:.1f}%", BOTH)
            # 29.5 and 41.4 are the movable fractions; quoting them as floors inverts the claim.
            self.assertNotIn(f"These score {complement:.1f}%", README)


class ClaimsTheEvidenceContradicts(unittest.TestCase):
    def test_published_guardrail_is_not_described_as_worse_than_doing_nothing(self):
        """It is below the fitted floor but above majority class, so this phrasing was false."""
        f = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))
        pub = f["published_label_prediction_accuracy"]
        self.assertGreater(pub["mind2web_sc"]["LLaMA-Guard3"], f["mind2web_sc"]["majority_class"])
        self.assertGreater(
            pub["eicu_ac"]["LLaMA-Guard3"], f["eicu_ac"]["train_majority_applied_to_valid"])
        for banned in ("worse than not trying", "Nobody reports this"):
            self.assertNotIn(banned, LIVE)

    def test_the_task_blind_claim_is_scoped_to_the_floor_that_is_task_blind(self):
        # Only the Mind2Web-SC floor ignores the task text; the EICU one infers a template from it.
        self.assertNotIn("a lookup table that never reads the task", LIVE)

    def test_borrowed_figures_are_marked_as_not_reproduced(self):
        self.assertTrue(
            re.search(r"not reproduced", README) and re.search(r"not reproduced here", PAPER),
            "figures taken from another paper must say so where they are used",
        )


if __name__ == "__main__":
    unittest.main()


class StaleClaimsThatSurvivedTheirCorrection(unittest.TestCase):
    """Claims that outlived the fixes that were supposed to remove them.

    Each was found by an audit, not by a test, and each had a corrected statement sitting
    elsewhere in the same document. These pin them so the correction cannot detach again.
    """

    def test_the_survey_prose_names_the_policy_that_actually_sets_each_floor(self):
        """A table and the paragraph explaining it disagreed for one release.

        Fixing the survey's head-truncation moved BeaverTails' floor from response-only to
        prompt-only. The tables were updated; the sentence explaining *why* the floor is high was
        not, and went on attributing it to the policy that had just lost. Both readings are
        plausible English, so nothing caught it. This requires the winning policy to be the one the
        prose credits.
        """
        survey = json.loads(Path("evidence/floor-survey.json").read_text(encoding="utf-8"))
        for entry in survey["benchmarks"]:
            if entry["benchmark"] != "BeaverTails":
                continue
            policies = entry["policies"]
            winner = max(policies, key=lambda k: policies[k])
            self.assertEqual(winner, entry["floor_policy"])
            # The losing partial-input policy must not be described as the one that predicts.
            loser = "response only, prompt deleted"
            if winner != loser:
                for name, text in (("README", README), ("PAPER", PAPER)):
                    with self.subTest(document=name):
                        self.assertNotIn(
                            "the response alone predicts the label", live_claims(text),
                            f"{name} credits the response-only policy, which does not set the floor",
                        )
                        self.assertNotIn(
                            "the response alone in the second", live_claims(text),
                            f"{name} credits the response-only policy, which does not set the floor",
                        )

    def test_unguarded_model_count_matches_the_table_above_it(self):
        """Every raw score in the section 3.1 table is below the 7778 always-refuse baseline.

        The prose said four of five, which was true only under a baseline bug the same document
        says was fixed.
        """
        self.assertNotIn("Four of five fall below a policy that ignores its input", PAPER)
        self.assertNotIn("four of the five unguarded models do not",
                         Path("docs/limitations.md").read_text(encoding="utf-8"))

    def test_the_vacuous_completeness_phrasing_is_not_used_as_a_live_claim(self):
        """Section 4.1 calls this form vacuous and says it was corrected, so it must not remain.

        It is permitted where the paper quotes it in order to reject it, which is why the check
        looks for it as an assertion rather than anywhere at all.
        """
        for phrasing in ("controls are repair-only and complete with\nrespect to the errors",
                         "repair-only, and complete with\nrespect to the error classes"):
            self.assertNotIn(phrasing, PAPER)

    def test_withdrawn_familiarity_claim_is_not_asserted_in_live_prose(self):
        """Section 3.12 withdrew 'the familiarity explanation is false'; section 3.11's trailing
        pointer must not reassert it."""
        for banned in ("familiarity reading of it false", "the familiarity explanation is false"):
            self.assertNotIn(banned, LIVE)



class ErrorsIntroducedByRestructuring(unittest.TestCase):
    """A round of restructuring introduced these. They are the ones a reader hits first."""

    def test_the_stated_rule_matches_the_cell_counts_it_sits_beside(self):
        """The README described the labelling rule wrongly, implying 6 ask cells beside a claim of 14 refusals."""
        counts = {"ABSTAIN": 0, "ACT": 0, "ASK": 0}
        seen = set()
        for record in json.loads(
            Path("evidence/ablation/M1-gemma4-12b/raw-samples.json").read_text(encoding="utf-8")
        ):
            if record["scenario_id"] not in seen:
                seen.add(record["scenario_id"])
                counts[record["expected"]] += 1
        self.assertEqual(counts, {"ABSTAIN": 14, "ACT": 2, "ASK": 2})
        flat = " ".join(README.split())
        self.assertIn(f"{counts['ABSTAIN']} of the 18 cells are refusals", flat)
        # Asking is correct only under a valid authority; omitting that was the error.
        self.assertIn("ask when the authority is valid but approval is unknown", flat)

    def test_reckless_policy_is_not_said_to_beat_every_model(self):
        """It ties three of five on the exploratory surface."""
        ma = json.loads(Path("evidence/model-ablation.json").read_text(encoding="utf-8"))
        observed = ma["exploratory"]["observed_bps"].values()
        ties = sum(1 for v in observed if v == ma["exploratory"]["substitute_bps"]["always_act"])
        self.assertGreater(ties, 0, "if nothing ties, the stronger claim would be permissible")
        # The unqualified form is the error; "matching or beating" is the accurate one, so the
        # check has to look for the claim rather than for a substring of it.
        live = live_claims(README) + live_claims(PAPER)
        self.assertIsNone(
            re.search(r"(?<!s or )(?<!g or )beat(s|ing) every real model", live),
            "the reckless policy ties three of five models and must not be said to beat them all",
        )
        self.assertNotIn("better than every real model", live)

    def test_factorial_table_excludes_runs_that_never_completed(self):
        """One aborted run left partial arms on disk; a glob swept them into a published table."""
        import glob
        import os
        from collections import defaultdict

        asked = defaultdict(lambda: [0, 0])
        for directory in glob.glob("evidence/factorial/*"):
            if not os.path.exists(os.path.join(directory, "factorial.json")):
                continue  # aborted run: raw arms exist, manifest does not
            for path in glob.glob(os.path.join(directory, "raw-*.json")):
                vocab = "finance" if os.path.basename(path)[4:].startswith("finance") else "maintenance"
                for record in json.loads(Path(path).read_text(encoding="utf-8")):
                    if "decision" not in record or record["permission"] != "GRANT":
                        continue
                    asked[vocab][0] += record["decision"] == "ASK"
                    asked[vocab][1] += 1
        rate = {v: asked[v][0] / asked[v][1] * 100 for v in ("finance", "maintenance")}
        self.assertEqual(round(rate["finance"], 1), 22.7)
        self.assertEqual(round(rate["maintenance"], 1), 51.5)

    def test_readme_has_no_duplicate_paragraphs(self):
        """Restructuring pasted a paragraph twice (one with bolded lead-in, one without); ensure
        every non-trivial paragraph is unique once stripped of markdown emphasis."""
        def unformat(s: str) -> str:
            return re.sub(r"[*_#`]", "", s).strip()

        paras = [unformat(p) for p in README.split("\n\n") if len(unformat(p)) > 80 and not p.strip().startswith("|")]
        seen = set()
        for p in paras:
            self.assertNotIn(p, seen, f"duplicate paragraph in README: {p[:60]}...")
            seen.add(p)

    def test_paper_section_headers_are_in_numerical_order(self):
        """Sections in PAPER.md must be sequentially numbered without out-of-order subsections."""
        headers = re.findall(r"^(#{2,3})\s+(\d+\.\d+(?:\.\d+)?)\s+", PAPER, flags=re.MULTILINE)
        numbers = [tuple(map(int, num.split("."))) for _, num in headers]
        for i in range(len(numbers) - 1):
            curr, nxt = numbers[i], numbers[i + 1]
            if curr[0] == nxt[0]:  # same top-level section
                self.assertLess(curr, nxt, f"Section {'.'.join(map(str, curr))} appears before {'.'.join(map(str, nxt))}")
