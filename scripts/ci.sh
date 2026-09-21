#!/usr/bin/env bash
# Every check this project makes, in one place.
#
# GitHub Actions runs this file and nothing else, and `make ci` runs the same file, so a green
# local run and a green remote run mean the same thing. There is no second copy of the checks
# to drift out of step.
#
# Runs offline against committed evidence. Calls no model. Needs Python 3.13 and no packages.
set -uo pipefail

# Find an interpreter of at least 3.13. Honours $PYTHON if it is set and usable, so callers can
# pin one, and otherwise tries the usual names. Hosted runners generally only have `python3`;
# some local setups shadow it, which is why this looks past the first candidate.
for candidate in "${PYTHON:-}" python3.13 python3 python; do
  [ -n "$candidate" ] || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)' \
     > /dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "${PYTHON:-}" ]; then
  echo "No Python 3.13 or newer found. Set PYTHON to one." >&2
  exit 1
fi
printf 'Using %s (%s)\n' "$PYTHON" "$("$PYTHON" --version 2>&1)"
export PYTHONPATH="src:experiments"

# Scratch outputs go in a private directory. These were fixed /tmp/ci-*.json paths, so two runs at
# once (a pre-push hook while a `make ci` is going) overwrote each other's intermediates and could
# diff a file the other run had just written.
CI_TMP="$(mktemp -d "${TMPDIR:-/tmp}/vaa-ci.XXXXXXXX")"
trap 'rm -rf "$CI_TMP"' EXIT
export CI_TMP

failed=0
step() {
  printf '\n\033[1m== %s ==\033[0m\n' "$1"; shift
  if "$@"; then printf '   ok\n'; else printf '   FAILED\n'; failed=1; fi
}

step "Tests" $PYTHON -m unittest discover -s tests -p 'test_*.py'
# A stale count in prose is the exact defect this project exists to detect, and it recurred:
# the README's "212 tests" outlived the commit that made it 213, and no check diffed the
# sentence against the suite. Bind every "N tests" claim in the README to the discovered count.
step "README test count matches the suite" bash -c '
  set -e
  actual="$("$0" -m unittest discover -s tests -p "test_*.py" 2>&1 | sed -n "s/^Ran \([0-9][0-9]*\) tests.*/\1/p" | tail -1)"
  [ -n "$actual" ] || { echo "could not determine the test count"; exit 1; }
  for claim in $(grep -oE "[0-9]+ tests" README.md | sed "s/ tests//" | sort -u); do
    if [ "$claim" != "$actual" ]; then
      echo "README claims $claim tests but the suite runs $actual."
      exit 1
    fi
  done
  echo "   README test count ($actual) matches the suite"' "$PYTHON"

# Lint is advisory only when ruff is absent, because this project has no dependencies and a
# contributor without it should still be able to run every other check.
step "Lint" bash -c '
  if command -v ruff > /dev/null 2>&1; then
    ruff check .
  else
    echo "   ruff not installed, skipping"
  fi'

step "Compile" $PYTHON -m compileall -q src experiments headroom.py docs
# Running it is not the check. A tool that computed nothing also exits 0, so the printed output has
# to contain both results.
step "The tool offered for copying still runs" \
  bash -c 'set -e
    out="$("$0" floors.py)"
    printf %s "$out" | grep -q "no-model floor:" || { echo "no floor in output"; exit 1; }
    printf %s "$out" | grep -q "adaptation headroom:" || { echo "no headroom in output"; exit 1; }
    printf %s "$out" | grep -q "permutation p:" || { echo "no p-value in output"; exit 1; }' "$PYTHON"

step "Headline result reproduces from committed evidence" \
  bash -c '$0 headroom.py > /dev/null' "$PYTHON"

step "Exhaustive sweep still reports zero headroom" bash -c 'set -e
  "$0" experiments/exhaustive.py --exploratory evidence/ablation \
    --confirmatory evidence/confirmatory --output $CI_TMP/ci-exhaustive.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/exhaustive.json"))
assert a == json.load(open("$CI_TMP/ci-exhaustive.json")), "exhaustive drifted from committed evidence"
for surface, block in a["surfaces"].items():
    for metric, r in block["metrics"].items():
        assert r["headroom_bps"] == 0, (surface, metric, r["headroom_bps"])
assert a["configuration_space"]["total"] == 328
PY' "$PYTHON"

# The two floor sweeps behind sections 3.3 and 3.4 were recomputable via `make verify` but never
# compared to anything in CI, so their published numbers (9966/6607/6718; 9589 vs 5978/6045) could
# have drifted from the code unchecked.
step "Floor sweeps reproduce their published figures" bash -c 'set -e
  "$0" experiments/sweep.py --ablation-root evidence/ablation \
    --output $CI_TMP/ci-sweep.json > /dev/null
  "$0" experiments/confirm_sweep.py --root evidence/confirmatory \
    --output $CI_TMP/ci-confirm-sweep.json > /dev/null
  "$0" - <<PY
import json
assert json.load(open("evidence/sweep.json")) == json.load(open("$CI_TMP/ci-sweep.json")), \\
    "exploratory floor sweep drifted from committed evidence"
c = json.load(open("$CI_TMP/ci-confirm-sweep.json"))["shared_floor"]
on = c["guards_on"]; off = c["guards_off"]
assert on["adaptation_headroom_bps"] == 0, on
assert on["best_single_floor"] == 0 and on["mean_bps_at_best_single_floor"] == 9589, on
assert on["mean_bps_if_each_model_uses_its_own_best"] == 9589, on
assert off["adaptation_headroom_bps"] == 67, off
assert off["mean_bps_at_best_single_floor"] == 5978, off
assert off["mean_bps_if_each_model_uses_its_own_best"] == 6045, off
PY' "$PYTHON"

step "Dominance proved for every possible model, over every admissible floor" bash -c 'set -e
  "$0" experiments/theorem.py --output $CI_TMP/ci-theorem.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/theorem.json"))
assert a == json.load(open("$CI_TMP/ci-theorem.json")), "the proof drifted from committed evidence"
assert a["risk_never_read_by_scoring_or_guards"]
assert a["dominating_under_both_metrics"] == [
    "invalid_authority+denied_approval+unknown_approval@0"], a["dominating_under_both_metrics"]
for metric in ("accuracy", "utility"):
    assert a[metric]["holds_for_every_possible_model"], metric
    assert a[metric]["attains_ceiling_over_every_integer_floor"], metric
PY' "$PYTHON"

step "No reweighting rescues it, and the sweep can still detect headroom" bash -c 'set -e
  "$0" experiments/dominance.py --exploratory evidence/ablation \
    --confirmatory evidence/confirmatory --output $CI_TMP/ci-dominance.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/dominance.json"))
assert a == json.load(open("$CI_TMP/ci-dominance.json")), "dominance drifted from committed evidence"
for surface, block in a.items():
    for metric, r in block.items():
        assert r["dominating_intersection_size"] > 0, (surface, metric)
        assert r["max_headroom_bps_over_random_reweightings"] < 1e-6, (surface, metric)
        # A sweep that always answers zero would make the headline claim meaningless.
        assert r["sensitivity_control_passes"], (surface, metric)
PY' "$PYTHON"

step "The reckless model still scores perfectly, and the boundary still bites" bash -c 'set -e
  "$0" experiments/model_ablation.py --exploratory evidence/ablation \
    --confirmatory evidence/confirmatory --output $CI_TMP/ci-ablation.json > /dev/null
  "$0" experiments/boundary.py --exploratory evidence/ablation \
    --confirmatory evidence/confirmatory --output $CI_TMP/ci-boundary.json > /dev/null
  "$0" - <<PY
import json
abl = json.load(open("evidence/model-ablation.json"))
assert abl == json.load(open("$CI_TMP/ci-ablation.json")), "model ablation drifted"
bnd = json.load(open("evidence/boundary.json"))
assert bnd == json.load(open("$CI_TMP/ci-boundary.json")), "boundary drifted"
for surface in ("exploratory", "confirmatory"):
    assert abl[surface]["substitute_bps"]["always_act"] == 10000, surface
    assert abl[surface]["fraction_of_score_that_is_model_independent"] > 0.85, surface
    # The monotonicity qualifier is load-bearing; this is what makes it so.
    assert bnd[surface]["headroom_bps"] > 0, surface
PY' "$PYTHON"

step "Surface factorial reproduces from committed draws" bash -c 'set -e
  "$0" experiments/factorial_report.py --root evidence/factorial \
    --output $CI_TMP/ci-factorial.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/factorial-effects.json"))
assert a == json.load(open("$CI_TMP/ci-factorial.json")), "factorial effects drifted"
assert a["preregistered"] is False, "this run was exploratory and must stay labelled so"
acc = a["pooled"]["accuracy"]; ask = a["pooled"]["ask_rate"]
assert acc["vocabulary"]["excludes_zero"] and acc["vocabulary"]["effect_bps"] < 0
assert ask["vocabulary"]["excludes_zero"] and ask["vocabulary"]["effect_bps"] > 0
# The whole point of the 2x2 is that vocabulary, not distractors, carries the effect.
for kind in (acc, ask):
    assert abs(kind["vocabulary"]["effect_bps"]) > 3 * abs(kind["distractors"]["effect_bps"])
PY' "$PYTHON"

step "Seven-vocabulary sweep reproduces from committed draws" bash -c 'set -e
  "$0" experiments/vocabulary_report.py --root evidence/vocabulary \
    --output $CI_TMP/ci-vocabulary.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/vocabulary-effects.json"))
assert a == json.load(open("$CI_TMP/ci-vocabulary.json")), "vocabulary effects drifted"
assert a["preregistered"] is False
acc = a["pooled"]["accuracy"]
# The familiarity explanation is falsified only while some vocabulary outscores finance.
assert acc["finance_rank"] > 1, "finance top would revive the familiarity reading"
# The nonsense control is the load-bearing one; if it stopped being worst, say so rather than drift.
assert acc["ranking_best_first"][-1] == "nonsense"
PY' "$PYTHON"

step "The section 4.1 condition still has no counterexample" bash -c 'set -e
  "$0" experiments/verify_theorem.py --max-controls 3 --output $CI_TMP/ci-theorem-search.json \
    > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/theorem-search.json"))
assert a == json.load(open("$CI_TMP/ci-theorem-search.json")), "theorem search drifted"
assert a["counterexamples_found"] == 0
# A search qualifying nothing would pass vacuously.
assert a["pairs_satisfying_all_three_conditions"] > 1000
PY' "$PYTHON"

step "The reproduced baseline still says what the documents say it says" bash -c 'set -e
  "$0" - <<PY
import json
r = json.load(open("evidence/guard-reproduction/reproduction.json"))
f = json.load(open("evidence/external-floor.json"))
# Mind2Web reproduces below the floor; EICU does not reproduce below it and is withdrawn.
assert r["mind2web_sc"]["accuracy"] < f["mind2web_sc"]["no_model_floor"]
# Below the corrected floor point estimate, inside its interval. Both halves, because the
# documents claim exactly that and either flipping makes them stale.
assert r["eicu_ac"]["accuracy"] < f["eicu_ac"]["no_model_floor"]
assert r["eicu_ac"]["accuracy"] > f["eicu_ac"]["no_model_floor_ci"][0]
assert r["published_for_comparison"]["eicu_ac"] < f["eicu_ac"]["no_model_floor_ci"][0]
for name in ("mind2web_sc", "eicu_ac"):
    assert r[name]["off_contract"] == 0 and r[name]["transport_errors"] == 0
PY' "$PYTHON"

step "The headroom test still discriminates outside this repository" bash -c 'set -e
  "$0" - <<PY
import json
r = json.load(open("evidence/external-headroom.json"))["benchmarks"]
blob = json.load(open("evidence/external-headroom.json"))
refused = {e["benchmark"] for e in blob["not_run"]}
degenerate = {k for k, v in r.items() if v.get("permutation_null_is_degenerate")}
positive = {k for k, v in r.items() if v.get("exceeds_chance_structure")}

# Preserve distinct recorded outcomes without requiring a positive result.
# Null concentration is a diagnostic; it is not a validity veto. The synthetic
# positive control in the unit suite tests that this distinction is load-bearing.
verdicts = {"nominal association" if name in positive else "no nominal association" for name in r}
if refused:
    verdicts.add("refused")
assert len(verdicts) > 1, ("the test returns a single verdict everywhere", sorted(verdicts))
assert refused, "the label-derived refusal never fires, so that guard is untested here"
# Numeric consistency is distinct from the exchangeability assumption required
# to interpret a permutation p-value for an observational or fitted-policy dataset.
for name, v in r.items():
    assert v["headroom_bps"] >= 0, name
    assert 0.0 < v["permutation_p"] <= 1.0, name
    assert v["exceeds_chance_structure"] == (v["permutation_p"] < 0.05), name
PY' "$PYTHON"

step "The floor survey still discriminates" bash -c 'set -e
  "$0" - <<PY
import json
s = json.load(open("evidence/floor-survey.json"))
gaps = [b["floor_above_majority_points"] for b in s["benchmarks"]]
# The survey is only worth reporting if it separates benchmarks rather than condemning all of them.
assert any(g > 5 for g in gaps), gaps
assert any(g == 0 for g in gaps), gaps
for entry in s["benchmarks"]:
    assert entry["floor_policy"] in entry["policies"], entry["benchmark"]
PY' "$PYTHON"

step "Third-party benchmark floors stay internally consistent" bash -c 'set -e
  "$0" - <<PY
import json
r = json.load(open("evidence/external-floor.json"))
pub = r["published_label_prediction_accuracy"]
# The EICU-AC published figures are scored on all 316 records, not the 128-item valid split, so the
# comparison below uses the full-set floor. Reading the valid-split floor here compared across two
# different evaluation sets, which is what made the earlier version of this claim unsound.
FLOOR_KEY = {"mind2web_sc": "no_model_floor", "eicu_ac": "full_set_no_model_floor"}
for key in ("mind2web_sc", "eicu_ac"):
    floor = r[key][FLOOR_KEY[key]]
    below = sorted(n for n, v in pub[key].items() if v < floor)
    assert below == r[key]["published_below_no_model_floor"], (key, below)
    # A published guardrail scoring under a lookup table is the finding; keep it asserted.
    assert "LLaMA-Guard3" in below, key
    # And the test has to discriminate, or it says nothing.
    # Discrimination margin. This was 0.25, set when the EICU-AC floor was computed on the wrong
    # evaluation set and came out 20 points too low. Against the corrected floor the best published
    # method clears it by 14.2 points, so the old threshold encoded the error rather than the claim.
    assert max(pub[key].values()) - floor > 0.10, key
PY' "$PYTHON"

step "Uncertainty analysis reproduces" bash -c 'set -e
  "$0" experiments/uncertainty.py --exhaustive evidence/exhaustive.json \
    --exploratory evidence/ablation --confirmatory evidence/confirmatory \
    --output $CI_TMP/ci-uncertainty.json > /dev/null
  "$0" - <<PY
import json
a = json.load(open("evidence/uncertainty.json"))
assert a == json.load(open("$CI_TMP/ci-uncertainty.json")), "uncertainty drifted from committed evidence"
# The paired comparison must be against each model.s derived policy, not a policy against itself.
for surface, block in a["surfaces"].items():
    for model, r in block.items():
        assert r["paired_vs_best_fixed"]["decisions_differ"] == 0, (surface, model)
PY' "$PYTHON"

step "Frozen confirmatory logic matches its checksum manifest" \
  "$PYTHON" scripts/verify-frozen.py

step "Figures have not drifted from the evidence" bash -c 'set -e
  generated="$CI_TMP/figures"
  FIGURE_OUTPUT_DIR="$generated" PYTHONPATH=docs "$0" -c \
    "import make_figure; make_figure.main(); make_figure.headroom_figure(); make_figure.model_ablation_figure()" \
    > /dev/null
  for name in ablation.svg headroom.svg model-ablation.svg; do
    cmp "docs/assets/$name" "$generated/$name" || {
      echo "docs/assets/$name does not match the committed evidence."
      exit 1
    }
  done' "$PYTHON"

step "No secrets, absolute paths or unexpected binary files" \
  "$PYTHON" scripts/scan-release.py .

step "Claims match the evidence" \
  $PYTHON -m unittest discover -s tests -p 'test_claims*.py'

printf '\n'
if [ "$failed" -ne 0 ]; then printf '\033[31mFAILED\033[0m\n'; exit 1; fi
printf '\033[32mAll checks passed.\033[0m\n'
