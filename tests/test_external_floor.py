"""Tests for the third-party benchmark floors.

These run against the committed aggregate statistics, not the datasets, which are deliberately not
redistributed. Recomputing from raw data is a separate opt-in target.
"""

import json
import re
import unittest
from pathlib import Path

from external_floor import PUBLISHED, majority, tokens


def live_claims(text: str) -> str:
    """The document minus its errata, which must be free to quote what it retracts."""
    return re.split(r"^## Errata$", text, flags=re.MULTILINE)[0]


class Helpers(unittest.TestCase):
    def test_majority_picks_the_modal_label(self):
        self.assertEqual(majority([1, 1, 0]), 1)
        self.assertEqual(majority([0, 0, 1]), 0)

    def test_tokens_are_lowercased_words(self):
        self.assertEqual(tokens("What is Patient 027's LOS?"), {"what", "is", "patient", "s", "los"})


class CommittedFloors(unittest.TestCase):
    def setUp(self):
        self.r = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))

    def test_floors_are_well_above_chance(self):
        # If a floor collapsed to chance the comparison below would stop meaning anything.
        self.assertGreater(self.r["mind2web_sc"]["no_model_floor"], 0.60)
        self.assertGreater(self.r["eicu_ac"]["no_model_floor"], 0.55)

    def test_a_published_guardrail_scores_below_the_no_model_floor(self):
        for key in ("mind2web_sc", "eicu_ac"):
            with self.subTest(benchmark=key):
                self.assertIn("LLaMA-Guard3", self.r[key]["published_below_no_model_floor"])

    def test_the_strong_methods_are_genuinely_above_the_floor(self):
        # The diagnostic has to discriminate. If it condemned everything it would be useless.
        for key, name in (("mind2web_sc", "AGrail (GPT-4o)"), ("eicu_ac", "GuardAgent (GPT-4)")):
            with self.subTest(benchmark=key):
                self.assertNotIn(name, self.r[key]["published_below_no_model_floor"])
                self.assertGreater(PUBLISHED[key][name] - self.r[key]["no_model_floor"], 0.25)

    def test_mind2web_blocking_always_needs_a_disqualifying_attribute(self):
        # Necessary but not sufficient, which is why the task text is still doing work.
        self.assertEqual(self.r["mind2web_sc"]["blocked_with_no_disqualifying_attribute"], 0)
        self.assertGreater(self.r["mind2web_sc"]["allowed_despite_disqualifying_attribute"], 50)

    def test_published_figures_are_recorded_for_both_benchmarks(self):
        for key in ("mind2web_sc", "eicu_ac"):
            self.assertEqual(
                set(self.r["published_label_prediction_accuracy"][key]), set(PUBLISHED[key]))


class FactorialEffects(unittest.TestCase):
    """The 2x2 that separates vocabulary from distractor fields."""

    def setUp(self):
        self.r = json.loads(Path("evidence/factorial-effects.json").read_text(encoding="utf-8"))

    def test_vocabulary_effect_is_real_and_negative(self):
        acc = self.r["pooled"]["accuracy"]["vocabulary"]
        self.assertTrue(acc["excludes_zero"])
        self.assertLess(acc["effect_bps"], 0)

    def test_vocabulary_dominates_distractors(self):
        for kind in ("accuracy", "ask_rate"):
            with self.subTest(kind=kind):
                p = self.r["pooled"][kind]
                self.assertGreater(
                    abs(p["vocabulary"]["effect_bps"]), 3 * abs(p["distractors"]["effect_bps"]))

    def test_vocabulary_shifts_behaviour_towards_asking(self):
        ask = self.r["pooled"]["ask_rate"]["vocabulary"]
        self.assertTrue(ask["excludes_zero"])
        self.assertGreater(ask["effect_bps"], 0)

    def test_the_run_is_marked_exploratory(self):
        # It was built after seeing the result it explains, and must not read as preregistered.
        self.assertFalse(self.r["preregistered"])

    def test_five_models_contributed(self):
        self.assertEqual(len(self.r["models"]), 5)

    def test_vocabulary_interval_excludes_zero_for_most_models(self):
        hits = sum(
            1 for m in self.r["models"].values() if m["ask_rate"]["vocabulary"]["excludes_zero"])
        self.assertEqual(hits, 5, "the ask-rate shift should hold for every model")

    def test_the_joint_band_exists_and_is_discriminating(self):
        """The simultaneous band over all 36 intervals must be present, non-trivial, and honest.

        An earlier version of PAPER.md quoted a joint band (critical value 3.07, 14 of 36
        surviving) that no committed script computed. These assertions keep a replacement from
        silently disappearing again: the band must exist, exceed every individual interval's
        reach, leave at least one interval standing but not swallow the family, and record
        survivors consistent with its own count.
        """
        band = self.r["joint_band"]
        self.assertEqual(band["intervals"], 36)
        self.assertGreater(band["band_bps"], 0)
        self.assertEqual(band["surviving_count"], len(band["surviving"]))
        self.assertGreater(band["surviving_count"], 0, "a band that rejects everything says nothing")
        self.assertLess(band["surviving_count"], 36, "a band nothing fails is not an adjustment")
        # Every recorded survivor must be an interval whose own observed magnitude clears the
        # band, so the survivor list cannot drift away from the numbers it summarises.
        for name in band["surviving"]:
            scope, kind, effect = (name.split("/") + [None, None])[:3]
            block = self.r["pooled"] if scope == "pooled" else self.r["models"].get(scope)
            self.assertIsNotNone(block, f"survivor names an unknown scope: {name}")
            estimate = block[kind][effect]["effect_bps"]
            self.assertGreater(abs(estimate), band["band_bps"], name)

    def test_no_pooled_effect_clears_the_joint_band(self):
        """The unflattering reading is pinned: joint adjustment over the stated family of 36
        leaves none of the six pooled intervals standing."""
        for kind in ("accuracy", "ask_rate"):
            for effect in ("vocabulary", "distractors", "interaction"):
                name = f"pooled/{kind}/{effect}"
                self.assertNotIn(name, self.r["joint_band"]["surviving"], name)


class TheEicuComparisonWasWithdrawn(unittest.TestCase):
    """It was the weaker half on interval alone; running the model removed it entirely."""

    def setUp(self):
        self.r = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))

    def test_the_floor_carries_an_interval(self):
        lo, hi = self.r["eicu_ac"]["no_model_floor_ci"]
        self.assertLess(lo, self.r["eicu_ac"]["no_model_floor"] < hi or hi)
        self.assertEqual(self.r["eicu_ac"]["evaluation_items"], 128)

    def test_the_published_baseline_is_below_the_floor_interval(self):
        lo, _ = self.r["eicu_ac"]["no_model_floor_ci"]
        score = self.r["published_label_prediction_accuracy"]["eicu_ac"]["LLaMA-Guard3"]
        self.assertLess(score, lo)

    def test_the_documents_record_the_retraction_of_the_withdrawal(self):
        docs = Path("PAPER.md").read_text(encoding="utf-8") + Path("docs/ERRATA.md").read_text(encoding="utf-8")
        self.assertRegex(docs, r"withdrawn against a floor that was wrong|withdrawal against a miscomputed floor")


class TheReproducedBaseline(unittest.TestCase):
    """Running LLaMA-Guard3 rather than citing it withdrew half the external finding."""

    def setUp(self):
        self.r = json.loads(
            Path("evidence/guard-reproduction/reproduction.json").read_text(encoding="utf-8"))
        self.f = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))

    def test_mind2web_reproduces_below_the_floor(self):
        run = self.r["mind2web_sc"]["accuracy"]
        self.assertLess(run, self.f["mind2web_sc"]["no_model_floor"])
        # And lands near the published figure, which is what makes it a reproduction.
        self.assertLess(abs(run - self.r["published_for_comparison"]["mind2web_sc"]), 0.05)

    def test_eicu_is_below_the_corrected_floor_but_inside_its_interval(self):
        """The withdrawal was made against a floor that omitted the majority-class policy.

        Both halves are asserted: the run is below the corrected floor's point estimate, and is not
        below its interval's lower bound. The prose has to say both, and if either flips the prose
        is stale.
        """
        run = self.r["eicu_ac"]["accuracy"]
        block = self.f["eicu_ac"]
        self.assertLess(run, block["no_model_floor"])
        self.assertGreater(run, block["no_model_floor_ci"][0])
        # The published figure clears both readings, which is why the finding stands.
        published = self.r["published_for_comparison"]["eicu_ac"]
        self.assertLess(published, block["no_model_floor_ci"][0])

    def test_the_floor_includes_the_majority_class_policy(self):
        """Omitting it understated the floor by six points and cost a correct finding."""
        block = self.f["eicu_ac"]
        self.assertGreaterEqual(block["no_model_floor"], block["majority_class_valid_split"])

    def test_the_runs_were_clean(self):
        for name in ("mind2web_sc", "eicu_ac"):
            with self.subTest(benchmark=name):
                self.assertEqual(self.r[name]["off_contract"], 0)
                self.assertEqual(self.r[name]["transport_errors"], 0)
                self.assertEqual(self.r[name]["scored"], self.r[name]["items"])


class TheTheoremSurvivesSearch(unittest.TestCase):
    def test_no_counterexample_was_found(self):
        s = json.loads(Path("evidence/theorem-search.json").read_text(encoding="utf-8"))
        self.assertTrue(s["conditions_sufficient_over_this_space"])
        self.assertEqual(s["counterexamples_found"], 0)
        # A search that qualified nothing would pass vacuously.
        self.assertGreater(s["pairs_satisfying_all_three_conditions"], 1000)


class RuleSpaceHeadroom(unittest.TestCase):
    """Check two over the benchmark's own rule library, discretised into a portfolio.

    This is the named missing experiment (PAPER section 4.3/section 7). These pins keep its
    committed answer honest: no nominal association, best fixed entry is all-rules-on, and the
    predicate validation that qualifies the whole result stays recorded beside it.
    """

    def setUp(self):
        data = Path("external/m2w/seeact/sample_labeled_all.json")
        if not data.exists():
            self.skipTest("Mind2Web-SC unavailable; provide an authorized local dataset copy")
        self.r = json.loads(Path("evidence/rule-space-headroom.json").read_text(encoding="utf-8"))

    def test_no_nominal_association(self):
        self.assertFalse(self.r["exceeds_chance_structure"])
        self.assertGreaterEqual(self.r["permutation_p"], 0.05)
        self.assertFalse(self.r["null_is_degenerate"], "the null here has enough units to move")

    def test_the_best_fixed_entry_is_all_rules_on(self):
        self.assertEqual(
            self.r["best_shared_policy"],
            "rules[license+country+member+minor18+vaccine+minor15]",
        )

    def test_the_portfolio_is_the_full_subset_space_plus_constants(self):
        self.assertEqual(self.r["portfolio_size"], 2 ** 6 + 2)

    def test_predicate_quality_stays_disclosed(self):
        """The result is only as good as its label-free predicates; they must stay on record."""
        v = self.r["predicate_validation_against_shipped_violations"]
        self.assertGreater(v["violations_recovered"], v["violations_missed"])
        self.assertEqual(v["violations_recovered"] + v["violations_missed"], 100)

    def test_the_oracle_diagnostic_stays_fenced_and_zero(self):
        """The annotation-oracle run is a direction measurement, not a verdict.

        Its firing signal is label-derived, so all-rules-on is perfect in every domain by
        construction and the family must be exactly zero. Pinning the fence keeps anyone --
        including a future me -- from quoting the diagnostic as a check-two result.
        """
        d = self.r["annotation_oracle_diagnostic"]
        self.assertIn("not_a_verdict", d)
        self.assertIn("label-derived", d["not_a_verdict"])
        self.assertEqual(d["headroom_bps"], 0)
        self.assertEqual(d["permutation_p"], 1.0)


class TheVocabularySweep(unittest.TestCase):
    """Seven surfaces for one rule, and what they say about the pair section 3.11 used."""

    def setUp(self):
        self.r = json.loads(Path("evidence/vocabulary-effects.json").read_text(encoding="utf-8"))

    def test_the_contrast_offered_against_familiarity_does_not_exclude_zero(self):
        """It was once bolded as a falsification. It is directional evidence at most.

        Asserted in the direction that keeps the documents honest: if this contrast ever did
        exclude zero the prose would be understating, which is the safe way round, but the test
        records why the strong claim was withdrawn.
        """
        acc = self.r["pooled"]["accuracy"]
        self.assertGreater(acc["finance_rank"], 1, "some vocabulary must outscore finance")
        contrast = acc["pairs"]["finance->clinical"]
        self.assertFalse(
            contrast["excludes_zero"],
            "if this now excludes zero, the withdrawal in 3.12 should be revisited",
        )
        self.assertFalse(contrast["excludes_zero_simultaneously"])

    def test_the_prespecified_familiarity_signal_is_reported_not_buried(self):
        """The sweep predicted nonsense would be worst if familiarity drove the effect. It is."""
        self.assertEqual(self.r["pooled"]["accuracy"]["ranking_best_first"][-1], "nonsense")
        paper = Path("PAPER.md").read_text(encoding="utf-8")
        self.assertIn("The familiarity explanation is not falsified", paper)

    def test_multiplicity_is_handled(self):
        for kind in ("accuracy", "ask_rate"):
            with self.subTest(kind=kind):
                block = self.r["pooled"][kind]
                self.assertGreater(block["simultaneous_band_bps"], 0)
                # Adjustment must be a real restriction, not a relabelling.
                self.assertLessEqual(
                    block["pairs_whose_simultaneous_interval_excludes_zero"],
                    block["pairs_whose_interval_excludes_zero"],
                )

    def test_the_rank_carries_an_interval(self):
        for kind in ("accuracy", "ask_rate"):
            with self.subTest(kind=kind):
                lo, hi = self.r["pooled"][kind]["original_pair_rank_95ci"]
                self.assertLessEqual(lo, self.r["pooled"][kind]["original_pair_rank_among_21"])
                self.assertGreaterEqual(hi, self.r["pooled"][kind]["original_pair_rank_among_21"])

    def test_the_meaningless_vocabulary_is_worst_on_accuracy(self):
        self.assertEqual(self.r["pooled"]["accuracy"]["ranking_best_first"][-1], "nonsense")

    def test_the_original_pair_is_typical_on_accuracy(self):
        acc = self.r["pooled"]["accuracy"]
        self.assertGreater(acc["original_pair_rank_among_21"], 3, "a top-3 gap would not be typical")

    def test_the_original_pair_is_atypical_on_ask_rate_and_the_docs_say_so(self):
        ask = self.r["pooled"]["ask_rate"]
        self.assertLessEqual(ask["original_pair_rank_among_21"], 3)
        paper = Path("PAPER.md").read_text(encoding="utf-8")
        self.assertIn("3rd of 21", paper)

    def test_the_spread_exceeds_the_original_gap(self):
        acc = self.r["pooled"]["accuracy"]
        self.assertGreater(acc["spread_bps"], acc["original_pair_gap_bps"])

    def test_all_five_models_completed_and_any_exclusion_is_named(self):
        """The 27B sweep failed once and was re-run with a fresh engine per vocabulary.

        Whatever the enumerator rejects must be named. An empty exclusion list is only acceptable
        alongside a complete set of models, which is what makes this assertion meaningful rather
        than vacuous.
        """
        self.assertEqual(len(self.r["models"]), 5)
        self.assertTrue(
            any("27b" in m for m in self.r["models"]),
            "the model whose runtime failed must be present, not quietly missing",
        )
        for record in self.r["excluded_runs"]:
            self.assertIn("reason", record)
            self.assertTrue(record["reason"])


class ThePairedComparison(unittest.TestCase):
    """The surviving external claim compares two systems on identical items, so it needs a test."""

    def setUp(self):
        self.r = json.loads(Path("evidence/external-floor.json").read_text(encoding="utf-8"))
        self.paired = self.r["mind2web_paired_vs_reproduced_guardrail"]

    def test_the_floor_beats_the_guardrail_on_a_paired_test(self):
        self.assertEqual(self.paired["items"], 200)
        self.assertGreater(
            self.paired["floor_correct_guard_wrong"], self.paired["guard_correct_floor_wrong"])
        self.assertLess(self.paired["mcnemar_exact_two_sided_p"], 0.01)

    def test_the_pairing_uses_the_policy_that_sets_the_floor(self):
        """Pairing against a weaker policy than the one quoted would flatter the comparison."""
        floor = self.r["mind2web_sc"]["no_model_floor"]
        self.assertEqual(floor, self.r["mind2web_sc"]["user_profile_lookup_crossvalidated"])

    def test_the_documents_report_the_p_value(self):
        p = self.paired["mcnemar_exact_two_sided_p"]
        docs = Path("README.md").read_text(encoding="utf-8") + Path("PAPER.md").read_text(encoding="utf-8")
        self.assertIn(f"{p}", docs)

    def test_both_floors_carry_intervals_and_both_readings_are_reported(self):
        """A point comparison against a floor that has an interval overstates what it settles."""
        for key in ("mind2web_sc", "eicu_ac"):
            with self.subTest(benchmark=key):
                block = self.r[key]
                lo, hi = block["no_model_floor_ci"]
                self.assertLess(lo, block["no_model_floor"])
                self.assertGreater(hi, block["no_model_floor"])
                # The conservative reading must be present, and agree here.
                self.assertIn("LLaMA-Guard3", block["published_below_floor_interval_lower_bound"])
                self.assertEqual(block["point_comparison_overstates"], [])


class TheFloorSurvey(unittest.TestCase):
    """Five benchmarks rather than two, and the distribution is the result."""

    def setUp(self):
        self.s = json.loads(Path("evidence/floor-survey.json").read_text(encoding="utf-8"))

    def test_the_survey_covers_more_than_the_original_two(self):
        self.assertGreaterEqual(len(self.s["benchmarks"]), 3)

    def test_it_reports_benchmarks_that_pass_as_well_as_fail(self):
        """A diagnostic that condemned everything would be useless; this must show both."""
        gaps = [b["floor_above_majority_points"] for b in self.s["benchmarks"]]
        self.assertTrue(any(g > 5 for g in gaps), "no benchmark has a floor above chance")
        self.assertTrue(any(g == 0 for g in gaps), "no benchmark passes cleanly")

    def test_benchmarks_it_cannot_assess_are_named_not_dropped(self):
        """AgentHarm has no allow/block label, which is a scope limit worth recording."""
        self.assertIsInstance(self.s["not_surveyed"], list)

    def test_every_surveyed_benchmark_names_its_floor_policy(self):
        for entry in self.s["benchmarks"]:
            with self.subTest(benchmark=entry["benchmark"]):
                self.assertIn(entry["floor_policy"], entry["policies"])
                self.assertAlmostEqual(
                    entry["no_model_floor"], entry["policies"][entry["floor_policy"]], places=4)


class TheHeadroomTestOutsideThisRepository(unittest.TestCase):
    """The check this project is named after, run on benchmarks it did not construct.

    These assertions encode mixed, bounded results. An earlier version of this class asserted that
    the analysis found real headroom on three of five benchmarks, which was wrong: two of those
    three were grouped by a field that is a copy of the label. A later blanket-negative reading was
    also wrong. The tests below pin each result separately and preserve the refusal that stops the
    original label-derived-grouping error recurring silently.
    """

    def setUp(self):
        blob = json.loads(Path("evidence/external-headroom.json").read_text(encoding="utf-8"))
        self.h = blob["benchmarks"]
        self.refused = {e["benchmark"]: e for e in blob["not_run"]}

    def test_benchmarks_whose_units_are_the_label_are_refused_not_reported(self):
        """The error this analysis made, now a failing condition rather than a headline."""
        for name in ("BeaverTails", "ToxicChat"):
            with self.subTest(benchmark=name):
                self.assertIn(name, self.refused, f"{name} is not assessable and must not be scored")
                self.assertNotIn(name, self.h, f"{name} must not appear among reported results")
                entry = self.refused[name]
                self.assertTrue(entry["label_derived_units"])
                self.assertTrue(entry["single_label_units"], "must name the offending units")

    def test_external_results_and_public_summaries_remain_mixed_and_bounded(self):
        """Pin each result and prevent the superseded blanket-negative claim from returning."""
        eicu = self.h["EICU-AC"]
        self.assertTrue(eicu["exceeds_chance_structure"])
        self.assertLess(eicu["permutation_p"], 0.05)
        self.assertGreater(eicu["sensitivity_to_min_unit_size"]["10"]["permutation_p"], 0.05)
        self.assertFalse(self.h["R-Judge"]["exceeds_chance_structure"])
        self.assertEqual(self.h["Mind2Web-SC"]["headroom_bps"], 0)
        for path in ("README.md", "PAPER.md", "CITATION.cff", "docs/ERRATA.md"):
            with self.subTest(path=path):
                text = Path(path).read_text(encoding="utf-8")
                normalized = " ".join(text.split())
                self.assertNotIn(
                    "no adaptation headroom that survives scrutiny on any of them", normalized
                )
                self.assertNotIn("no positive result outside this repository", normalized)
                self.assertNotIn("has no positive result anywhere", normalized)
                self.assertNotIn("| 0.729 | no structure |", text)

    def test_concentration_is_a_diagnostic_not_an_automatic_inference_veto(self):
        """Preserve the recorded p-value without claiming confirmed adaptive benefit."""
        eicu = self.h["EICU-AC"]
        self.assertTrue(eicu["permutation_null_is_degenerate"])
        self.assertGreater(eicu["permutation_null_zero_fraction"], 0.9)
        self.assertLess(eicu["permutation_p"], 0.05)
        self.assertTrue(eicu["exceeds_chance_structure"])
        self.assertIn("does not by itself invalidate", eicu["permutation_null_note"])
        self.assertEqual(eicu["items_driving_the_gap"]["total_items_driving_the_gap"], 5)
        self.assertGreater(eicu["sensitivity_to_min_unit_size"]["10"]["permutation_p"], 0.05)

    def test_zero_concentration_diagnostic_varies_across_benchmarks(self):
        """The legacy diagnostic records concentration without controlling the verdict."""
        for name in ("Mind2Web-SC", "R-Judge"):
            with self.subTest(benchmark=name):
                self.assertFalse(self.h[name]["permutation_null_is_degenerate"])

    def test_the_null_mean_is_reported_as_a_shuffled_reference(self):
        """The null mean is recorded without treating it as an unbiased estimate of selection bias."""
        for name, v in self.h.items():
            with self.subTest(benchmark=name):
                self.assertIn("permutation_null_mean_bps", v)
                self.assertGreaterEqual(v["permutation_null_mean_bps"], 0)
                self.assertEqual(
                    v["headroom_above_null_mean_bps"],
                    v["headroom_bps"] - v["permutation_null_mean_bps"],
                )

    def test_rjudge_observed_gap_has_no_nominal_association(self):
        """R-Judge's positive point gap is below its shuffled null mean and nonsignificant."""
        rjudge = self.h["R-Judge"]
        self.assertGreater(rjudge["headroom_bps"], 0)
        self.assertFalse(rjudge["exceeds_chance_structure"])
        self.assertGreater(rjudge["permutation_p"], 0.05)
        self.assertLess(
            rjudge["headroom_above_null_mean_bps"], 0,
            "R-Judge's observed gap is smaller than its shuffled null mean",
        )

    def test_nonnegative_headroom_and_nominal_permutation_p_are_both_reported(self):
        """Support alone does not settle interval validity or establish adaptive benefit."""
        for name, v in self.h.items():
            with self.subTest(benchmark=name):
                self.assertGreaterEqual(v["headroom_bps"], 0)
                self.assertIn("permutation_p", v)

    def test_a_degenerate_interval_is_withheld_rather_than_shipped(self):
        """With a handful of units the resampled gap takes a handful of values. That is not an
        interval, and printing one invites a reader to do inference with it."""
        for name, v in self.h.items():
            with self.subTest(benchmark=name):
                if v["units"] < 5:
                    self.assertIsNone(v["headroom_ci_bps"])

    def test_the_benchmark_with_zero_headroom_has_one_policy_best_everywhere(self):
        m2w = self.h["Mind2Web-SC"]
        self.assertEqual(m2w["headroom_bps"], 0)
        self.assertEqual(m2w["units_whose_best_differs_from_shared"], [])


if __name__ == "__main__":
    unittest.main()
