"""The no-model floor of two published third-party guardrail benchmarks.

Everything else in this repository is measured on a benchmark I built, which makes the central
finding easy to dismiss: of course a toy of my own construction turned out to be broken. The test
that produced that finding is not specific to my benchmark though. It asks one question, and it
asks it of anything:

    How well does something with no model in it do?

Whatever score a trivial policy already reaches was never evidence about the model, and any
improvement should be read against that floor rather than against chance. Papers overwhelmingly
report against chance, or against nothing at all.

Applied here to the two benchmarks used by GuardAgent (arXiv 2406.09187) and AGrail
(arXiv 2502.11448):

  Mind2Web-SC   200 web tasks, balanced allow/block, six safety rules gating actions on user
                attributes such as age, driving licence, vaccination and membership.
  EICU-AC       316 clinical questions with role-based access control over ten databases, shipping
                an official train/valid split.

The floors computed are deliberately stupid. No model is called and no learned representation is
used: token overlap against training questions, and lookup tables over metadata. If a published
guardrail scores below these, that is worth knowing.

The datasets are not redistributed here. EICU-AC derives from eICU, whose source is credentialed,
and both are other people's work. This script reads local copies and commits only aggregate
statistics. `--help` prints where to obtain them.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

SEEDS = 5
FOLDS = 10
# `floor_survey.py` and `external_headroom.py` keep apostrophes in their own tokenisers. The
# difference was measured rather than reconciled: on this corpus both give the inferred
# template policy 0.5859, and the floor is set by the majority class at 0.6484 either way.
# The three analyses stay independently runnable rather than sharing a helper.
TOKEN = re.compile(r"[a-z]+")

# Shared with floor_survey.py and external_headroom.py's own bootstraps.
BOOTSTRAP_SEED = 20260819
BOOTSTRAP_ROUNDS = 2000


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for two systems scored on the same items.

    Comparing a floor against a published guardrail on identical items is a paired comparison, and
    reporting the two accuracies without a paired test leaves the difference unquantified. Only the
    discordant pairs carry information: under the null they split evenly, so the p-value is an exact
    two-sided binomial test on them. Exact rather than chi-square because the discordant counts here
    are small enough that the asymptotic form is unsafe.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, k) for k in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / (2**n))


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used because the floors are proportions on small evaluation sets.

    The EICU-AC comparison rests on 128 validation items, so quoting a point estimate against a
    published number without an interval overstates what it can settle. Draws within this set are
    independent items rather than repeated samples of one item, so no clustering is needed here.
    """
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def majority(labels: list[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def lookup_predict(table: dict[Any, list[int]], key: Any, fallback: int) -> int:
    got = table.get(key)
    return majority(got) if got else fallback


# ---------------------------------------------------------------- Mind2Web-SC

USER_SIDE = {
    "under18": lambda u: u["age"] < 18,
    "under15": lambda u: u["age"] < 15,
    "no_licence": lambda u: not u["dr_license"],
    "not_domestic": lambda u: not u["domestic"],
    "unvaccinated": lambda u: not u["vaccine"],
    "non_member": lambda u: not u["membership"],
}


def mind2web_floor_predictions(records: list[dict]) -> list[int]:
    """Held-out per-item predictions of the policy that actually sets the floor.

    The floor reported for Mind2Web-SC is the cross-validated profile lookup, not the best single
    attribute, so pairing against the single-attribute rule would compare the guardrail with a
    weaker policy than the one quoted. Predictions come from the held-out fold in each split, which
    is what makes them comparable with a zero-shot model's.
    """
    n = len(records)
    def profile(r):
        return (
            r["user_info"]["domestic"],
            r["user_info"]["dr_license"],
            r["user_info"]["vaccine"],
            r["user_info"]["membership"],
        )
    predictions = [0] * n
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    for fold in range(FOLDS):
        test = set(idx[fold::FOLDS])
        train = [i for i in idx if i not in test]
        table: dict[Any, list[int]] = {}
        for i in train:
            table.setdefault(profile(records[i]), []).append(records[i]["label"])
        fallback = majority([records[i]["label"] for i in train])
        for i in test:
            predictions[i] = lookup_predict(table, profile(records[i]), fallback)
    return predictions


def mind2web_no_model_floor(records: list[dict], seed: int) -> float:
    """Recomputes the Mind2Web-SC no-model floor for one bootstrap replicate.

    Mirrors the floor logic inside `mind2web()` (any_flag, best_single, profile-CV, majority) so a
    cluster bootstrap can refit the whole pipeline, including the max over candidate policies, on a
    resampled set of tasks. Kept as its own implementation rather than shared with `mind2web()`,
    matching how the CV logic is already duplicated between that function and
    `mind2web_floor_predictions`.
    """
    n = len(records)
    if n == 0:
        return 0.0
    labels = [r["label"] for r in records]

    def accuracy(predict) -> float:
        return sum(predict(r) == r["label"] for r in records) / n

    any_flag = accuracy(lambda r: int(any(f(r["user_info"]) for f in USER_SIDE.values())))
    best_single = max(accuracy(lambda r, f=f: int(f(r["user_info"]))) for f in USER_SIDE.values())

    def profile_key(r: dict) -> tuple:
        return (
            r["user_info"]["domestic"],
            r["user_info"]["dr_license"],
            r["user_info"]["vaccine"],
            r["user_info"]["membership"],
        )

    totals = []
    for s in range(SEEDS):
        idx = list(range(n))
        random.Random(seed + s).shuffle(idx)
        correct = 0
        for fold in range(FOLDS):
            test = set(idx[fold::FOLDS])
            train = [i for i in idx if i not in test]
            if not train:
                continue
            table: dict[Any, list[int]] = {}
            for i in train:
                table.setdefault(profile_key(records[i]), []).append(records[i]["label"])
            fallback = majority([records[i]["label"] for i in train])
            correct += sum(
                lookup_predict(table, profile_key(records[i]), fallback) == records[i]["label"]
                for i in test
            )
        totals.append(correct / n)
    profile = sum(totals) / len(totals)
    return max(any_flag, best_single, profile, max(Counter(labels).values()) / n)


def mind2web_cluster_bootstrap(
    records: list[dict],
    task_key: str = "annotation_id",
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Cluster bootstrap 95% interval for the Mind2Web-SC no-model floor.

    The 200 records are 150 distinct tasks; 50 tasks appear twice with opposite labels (the same
    task, once with and once without the disqualifying attribute in play). Wilson treats every
    record as an independent draw, which is too narrow once half the label-50 pairs are correlated
    by task. Whole tasks are resampled with replacement here, carrying every record that belongs to
    them, and the entire floor pipeline -- including the max over candidate policies -- is refit on
    each resample, so the interval reflects task-level rather than record-level uncertainty.
    """
    clusters: dict[Any, list[dict]] = {}
    for r in records:
        clusters.setdefault(r[task_key], []).append(r)
    cluster_ids = list(clusters.keys())
    m = len(cluster_ids)
    rng = random.Random(seed)
    floors = []
    for b in range(rounds):
        chosen = [cluster_ids[rng.randrange(m)] for _ in range(m)]
        resampled = [rec for cid in chosen for rec in clusters[cid]]
        floors.append(mind2web_no_model_floor(resampled, seed=seed + 1 + b))
    floors.sort()
    lo = floors[int(0.025 * rounds)]
    hi = floors[int(0.975 * rounds) - 1]
    return {
        "cluster_key": task_key,
        "clusters": m,
        "rounds": rounds,
        "seed": seed,
        "ci": [round(lo, 4), round(hi, 4)],
        "mean": round(sum(floors) / len(floors), 4),
    }


def mind2web(records: list[dict]) -> dict[str, Any]:
    """No official split ships with this set, so floors are cross-validated."""
    labels = [r["label"] for r in records]
    n = len(records)

    def accuracy(predict) -> float:
        return sum(predict(r) == r["label"] for r in records) / n

    # Every rule pairs a user-side condition with a task-side one. The user side is free.
    any_flag = accuracy(lambda r: int(any(f(r["user_info"]) for f in USER_SIDE.values())))
    best_single = max(
        (accuracy(lambda r, f=f: int(f(r["user_info"]))), name) for name, f in USER_SIDE.items()
    )

    def cv(keyfn) -> float:
        totals = []
        for seed in range(SEEDS):
            idx = list(range(n))
            random.Random(seed).shuffle(idx)
            correct = 0
            for fold in range(FOLDS):
                test = set(idx[fold::FOLDS])
                train = [i for i in idx if i not in test]
                table: dict[Any, list[int]] = {}
                for i in train:
                    table.setdefault(keyfn(records[i]), []).append(records[i]["label"])
                fallback = majority([records[i]["label"] for i in train])
                correct += sum(
                    lookup_predict(table, keyfn(records[i]), fallback) == records[i]["label"]
                    for i in test
                )
            totals.append(correct / n)
        return sum(totals) / len(totals)

    profile = cv(
        lambda r: (
            r["user_info"]["domestic"],
            r["user_info"]["dr_license"],
            r["user_info"]["vaccine"],
            r["user_info"]["membership"],
        )
    )
    return {
        "records": n,
        "label_counts": dict(Counter(labels)),
        "always_allow": round(accuracy(lambda r: 0), 4),
        "always_block": round(accuracy(lambda r: 1), 4),
        "block_if_any_user_precondition": round(any_flag, 4),
        "best_single_user_attribute": {
            "accuracy": round(best_single[0], 4),
            "attribute": best_single[1],
        },
        "user_profile_lookup_crossvalidated": round(profile, 4),
        "majority_class": round(max(Counter(labels).values()) / n, 4),
        "no_model_floor": round(
            max(any_flag, best_single[0], profile, max(Counter(labels).values()) / n), 4
        ),
        "no_model_floor_ci": [
            round(v, 4)
            for v in wilson(
                round(
                    max(any_flag, best_single[0], profile, max(Counter(labels).values()) / n) * n
                ),
                n,
            )
        ],
        "evaluation_items": n,
        "blocked_with_no_disqualifying_attribute": sum(
            1
            for r in records
            if r["label"] == 1 and not any(f(r["user_info"]) for f in USER_SIDE.values())
        ),
        "allowed_despite_disqualifying_attribute": sum(
            1
            for r in records
            if r["label"] == 0 and any(f(r["user_info"]) for f in USER_SIDE.values())
        ),
        "no_model_floor_cluster_bootstrap": mind2web_cluster_bootstrap(records),
    }


# -------------------------------------------------------------------- EICU-AC


def eicu_full_set_cv(
    records: list[dict], seed: int = BOOTSTRAP_SEED, folds: int = 5
) -> tuple[float, float, float]:
    """5-fold CV over all 316 EICU-AC records, keyed on (identity, q_tag).

    The published AGrail/GuardAgent figures for EICU-AC are scored on all 316 records, not the
    128-item `valid` split `eicu()` below evaluates -- their reported scores (e.g. 154/316, 312/316)
    are only reachable on the full set. Comparing them against a floor computed on 128 items crosses
    evaluation sets, so this recomputes the floor on the same 316 the published numbers use.
    Grouped on `q_tag`, the shipped template annotation, rather than `template`, which is
    near-unique per item and unusable as a lookup key.

    Returns (majority, role_x_shipped_q_tag, role_x_inferred_q_tag).
    """
    n = len(records)
    labels = [r["label"] for r in records]
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    bags = [tokens(r["question"]) for r in records]
    correct_shipped = 0
    correct_inferred = 0
    for fold in range(folds):
        test = [i for j, i in enumerate(idx) if j % folds == fold]
        train = [i for j, i in enumerate(idx) if j % folds != fold]
        table: dict[Any, list[int]] = {}
        for i in train:
            table.setdefault((records[i]["identity"], records[i]["q_tag"]), []).append(labels[i])
        fallback = majority([labels[i] for i in train])
        train_bags = [(bags[i], records[i]["q_tag"]) for i in train]
        for i in test:
            correct_shipped += (
                lookup_predict(table, (records[i]["identity"], records[i]["q_tag"]), fallback)
                == labels[i]
            )
            q = bags[i]
            best_tag = max(train_bags, key=lambda p: len(q & p[0]) / max(1, len(q | p[0])))[1]
            correct_inferred += (
                lookup_predict(table, (records[i]["identity"], best_tag), fallback) == labels[i]
            )
    majority_rate = max(Counter(labels).values()) / n
    return majority_rate, correct_shipped / n, correct_inferred / n


def eicu_full_set_bootstrap(
    records: list[dict], rounds: int = BOOTSTRAP_ROUNDS, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    """Out-of-bag bootstrap 95% interval for the full-set EICU-AC no-model floor.

    A plain item bootstrap -- resample 316 items with replacement, then 5-fold-split that resample
    -- leaks: a duplicated item lands on both sides of a fold, and the near-unique question text
    lets the inferred-q_tag lookup match a duplicate of itself, inflating accuracy well past the
    point estimate (verified: it pushes the mean above 0.91 against a 0.845 point estimate, the
    wrong direction for a statistic that also carries max-selection optimism). Training on each
    bootstrap draw and scoring only the items that draw left out (the out-of-bag set) avoids that,
    because no item can then appear on both sides. The shipped-annotation policy is excluded from
    the max every replicate, matching `eicu_full_set_cv`.
    """
    n = len(records)
    labels = [r["label"] for r in records]
    bags = [tokens(r["question"]) for r in records]
    rng = random.Random(seed)
    floors = []
    for b in range(rounds):
        train_idx = [rng.randrange(n) for _ in range(n)]
        oob_idx = [i for i in range(n) if i not in set(train_idx)]
        if not oob_idx:
            continue
        oob_labels = [labels[i] for i in oob_idx]
        majority_oob = max(Counter(oob_labels).values()) / len(oob_idx)
        table: dict[Any, list[int]] = {}
        for i in train_idx:
            table.setdefault((records[i]["identity"], records[i]["q_tag"]), []).append(labels[i])
        fallback = majority([labels[i] for i in train_idx])
        train_bags = [(bags[i], records[i]["q_tag"]) for i in train_idx]
        correct = 0
        for i in oob_idx:
            q = bags[i]
            best_tag = max(train_bags, key=lambda p: len(q & p[0]) / max(1, len(q | p[0])))[1]
            correct += (
                lookup_predict(table, (records[i]["identity"], best_tag), fallback) == (labels[i])
            )
        inferred_oob = correct / len(oob_idx)
        floors.append(max(majority_oob, inferred_oob))
    floors.sort()
    m = len(floors)
    lo = floors[int(0.025 * m)]
    hi = floors[int(0.975 * m) - 1]
    return {
        "method": "out-of-bag: refits majority and role x inferred q_tag each round on a "
        "bootstrap draw, scores on the items the draw left out, excludes the shipped-annotation "
        "policy from the max",
        "rounds": m,
        "seed": seed,
        "ci": [round(lo, 4), round(hi, 4)],
        "mean": round(sum(floors) / m, 4),
    }


def eicu(records: list[dict]) -> dict[str, Any]:
    """Uses the official train/valid split, which is how a published number would be obtained."""
    train = [r for r in records if r.get("split") == "train"]
    valid = [r for r in records if r.get("split") == "valid"]

    fallback = majority([r["label"] for r in train])
    table: dict[Any, list[int]] = {}
    for r in train:
        table.setdefault((r["identity"], r["q_tag"]), []).append(r["label"])

    shipped = sum(
        lookup_predict(table, (r["identity"], r["q_tag"]), fallback) == r["label"] for r in valid
    ) / len(valid)

    # The template annotation would not exist at inference, so infer it from raw text instead.
    prototypes = [(tokens(r["question"]), r["q_tag"]) for r in train]
    inferred = 0
    for r in valid:
        q = tokens(r["question"])
        tag = max(prototypes, key=lambda p: len(q & p[0]) / max(1, len(q | p[0])))[1]
        inferred += lookup_predict(table, (r["identity"], tag), fallback) == r["label"]
    inferred /= len(valid)

    # The simplest no-model policy of all, and the one this analysis originally omitted: predict
    # the commonest label. On this split that is "allow", and it scores higher than the template
    # lookup, so it sets the floor. Leaving it out understated the floor by six points and caused a
    # finding to be withdrawn that the corrected floor supports.
    constant = max(Counter(r["label"] for r in valid).values()) / len(valid)
    floor = max(inferred, constant)
    lo, hi = wilson(round(floor * len(valid)), len(valid))

    # The published AGrail/GuardAgent/LLaMA-Guard3 figures for this benchmark are scored on all 316
    # records (e.g. GuardAgent's 312/316), not the 128-item valid split above, so the comparison
    # against them has to use a floor computed on the same 316 -- see `eicu_full_set_cv`.
    majority_full, shipped_full, inferred_full = eicu_full_set_cv(records)
    floor_full = max(majority_full, inferred_full)
    floor_full_policy = (
        "majority class" if majority_full >= inferred_full else "role x inferred q_tag"
    )
    bootstrap_full = eicu_full_set_bootstrap(records)

    return {
        "majority_class_valid_split": round(constant, 4),
        "floor_policy": "majority class" if constant >= inferred else "role x inferred template",
        "records": len(records),
        "train": len(train),
        "valid": len(valid),
        "no_model_floor_ci": [round(lo, 4), round(hi, 4)],
        "evaluation_items": len(valid),
        "label_counts_train": dict(Counter(r["label"] for r in train)),
        "label_counts_valid": dict(Counter(r["label"] for r in valid)),
        "train_majority_applied_to_valid": round(
            sum(r["label"] == fallback for r in valid) / len(valid), 4
        ),
        "role_x_shipped_template": round(shipped, 4),
        "role_x_template_inferred_from_text": round(inferred, 4),
        "no_model_floor": round(floor, 4),
        "full_set_note": (
            "The published figures for this benchmark are scored on all 316 records, not the "
            "128-item valid split above (their reported scores, e.g. 154/316 and 312/316, are "
            "only reachable on the full set), so `published_below_no_model_floor` and "
            "`published_below_floor_interval_lower_bound` are computed against "
            "full_set_no_model_floor below rather than against no_model_floor. "
            "full_set_role_x_shipped_q_tag_cv uses the shipped q_tag annotation, which would not "
            "exist at inference; it is reported for reference only and excluded from the max that "
            "sets full_set_no_model_floor."
        ),
        "full_set_records": len(records),
        "full_set_cv_seed": BOOTSTRAP_SEED,
        "full_set_cv_folds": 5,
        "full_set_majority_class": round(majority_full, 4),
        "full_set_role_x_shipped_q_tag_cv": round(shipped_full, 4),
        "full_set_role_x_inferred_q_tag_cv": round(inferred_full, 4),
        "full_set_no_model_floor": round(floor_full, 4),
        "full_set_no_model_floor_policy": floor_full_policy,
        "full_set_no_model_floor_bootstrap": bootstrap_full,
    }


# Published figures, for comparison against the floors above. Label prediction accuracy as
# reported in AGrail table 2 (arXiv 2502.11448), which also restates the GuardAgent results.
PUBLISHED = {
    "mind2web_sc": {
        "AGrail (GPT-4o)": 0.984,
        "AGrail (Claude-3.5)": 0.940,
        "GuardAgent (GPT-4)": 0.900,
        "LLaMA-Guard3": 0.560,
    },
    "eicu_ac": {
        "GuardAgent (GPT-4)": 0.987,
        "AGrail (Claude-3.5)": 0.984,
        "AGrail (GPT-4o)": 0.978,
        "LLaMA-Guard3": 0.487,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Both datasets are linked from https://github.com/guardagent/dataset "
        "(Mind2Web-SC: seeact/sample_labeled_all.json, EICU-AC: ehragent/eicu_ac.json).",
    )
    parser.add_argument("--mind2web", required=True, help="path to sample_labeled_all.json")
    parser.add_argument("--eicu", required=True, help="path to eicu_ac.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Pair the Mind2Web-SC floor against the reproduced guardrail run on identical items.
    paired: dict[str, Any] = {}
    reproduction = Path("evidence/guard-reproduction/raw-mind2web_sc.json")
    if reproduction.exists():
        m2w_records = json.loads(Path(args.mind2web).read_text(encoding="utf-8"))
        floor_pred = mind2web_floor_predictions(m2w_records)
        guard = json.loads(reproduction.read_text(encoding="utf-8"))
        labels = [r["label"] for r in m2w_records]
        guard_pred = [r["predicted"] for r in guard]
        only_floor = sum(
            f == y and g != y for f, g, y in zip(floor_pred, guard_pred, labels, strict=True)
        )
        only_guard = sum(
            g == y and f != y for f, g, y in zip(floor_pred, guard_pred, labels, strict=True)
        )
        paired = {
            "items": len(labels),
            "floor_correct_guard_wrong": only_floor,
            "guard_correct_floor_wrong": only_guard,
            "mcnemar_exact_two_sided_p": round(mcnemar_exact(only_floor, only_guard), 6),
            "note": "Best task-blind policy against the LLaMA-Guard3 run in "
            "evidence/guard-reproduction, scored on identical items.",
        }

    report = {
        "note": "Aggregate statistics only. The datasets themselves are not redistributed here.",
        "mind2web_paired_vs_reproduced_guardrail": paired,
        "mind2web_sc": mind2web(json.loads(Path(args.mind2web).read_text(encoding="utf-8"))),
        "eicu_ac": eicu(json.loads(Path(args.eicu).read_text(encoding="utf-8"))),
        "published_label_prediction_accuracy": PUBLISHED,
    }
    for key in ("mind2web_sc", "eicu_ac"):
        # eicu_ac's published figures are scored on all 316 records, not the 128-item valid split
        # `no_model_floor` covers (see `full_set_note` on that block), so the comparison against
        # them has to use the full-set floor. mind2web_sc's 200 items already match the published
        # set, so its comparison is unchanged.
        if key == "eicu_ac":
            floor = report[key]["full_set_no_model_floor"]
            bounds = report[key]["full_set_no_model_floor_bootstrap"]["ci"]
        else:
            floor = report[key]["no_model_floor"]
            bounds = report[key].get("no_model_floor_ci")
        report[key]["published_below_no_model_floor"] = sorted(
            name for name, score in PUBLISHED[key].items() if score < floor
        )
        # A point comparison against a floor that carries an interval overstates what it settles.
        # Where the interval exists, also report which figures fall below its lower bound, which is
        # the conservative reading and the one the prose should use.
        if bounds:
            report[key]["published_below_floor_interval_lower_bound"] = sorted(
                name for name, score in PUBLISHED[key].items() if score < bounds[0]
            )
            report[key]["point_comparison_overstates"] = sorted(
                set(report[key]["published_below_no_model_floor"])
                - set(report[key]["published_below_floor_interval_lower_bound"])
            )
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for key, title in (("mind2web_sc", "Mind2Web-SC"), ("eicu_ac", "EICU-AC")):
        r = report[key]
        print(f"\n{title}: no-model floor {r['no_model_floor']:.1%}")
        for name, score in sorted(PUBLISHED[key].items(), key=lambda kv: -kv[1]):
            mark = "  BELOW THE FLOOR" if score < r["no_model_floor"] else ""
            print(f"    {name:22s} {score:6.1%}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
