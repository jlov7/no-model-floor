"""Adaptation headroom on published benchmarks, so the second diagnostic finally leaves this repo.

Section 4.3 concedes the sharpest structural criticism this work received: of the two checks it
proposes, only the first has ever been applied outside the benchmark it was built to invalidate.
The headroom test, which the project is named after, had been computed exactly once, on a benchmark
where section 3.6 proves it must return zero. A measurement instrument validated only against its
own calibration target has no external standing.

This computes it on five published benchmarks.

**The obstacle, and what is done about it.** The test as stated in section 4 enumerates a *declared
control space*, and the adaptive guardrails this work points at do not have one: they adapt over
open-ended, model-generated rule sets. That mismatch is real and is not solved here. What is done
instead is to apply the test to a control space that these benchmarks genuinely do have, and that
anyone evaluating on them has to choose from: the set of trivial policies available for scoring
them. Choosing a different one per subgroup is adaptation in exactly the algorithm-selection sense,
with subgroups as instances and policies as solvers.

**Units.** Each benchmark ships a natural grouping that is not the label: the scenario in R-Judge,
the domain in Mind2Web-SC, the requester role in EICU-AC, the harm category in BeaverTails, whether
the turn is a jailbreak attempt in ToxicChat. These are the adaptation units, and they are the
closest available analogue to "per model" in the synthetic study.

**What a nonzero result means.** That the best trivial policy differs by subgroup, so a single fixed
choice leaves measurable score on the table. That is the condition under which an adaptive method
has something to win, and its absence is the condition under which an adaptive result carries no
information. Either answer is worth having; the point is that the instrument now returns an answer
about somebody else's benchmark.

Offline once the datasets are fetched. Calls no model.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

SEED = 20260819
ROUNDS = 2000
TOKEN = re.compile(r"[a-z']+")


def tokens(text: Any) -> frozenset[str]:
    return frozenset(TOKEN.findall(str(text).lower()))


def _majority(labels: list[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def policy_scores(
    items: list[dict], policies: dict[str, Callable[[list[dict], list[int]], list[int]]]
) -> dict[str, list[int]]:
    """Per-item predictions of every policy, fitted where fitting is needed."""
    labels = [i["_y"] for i in items]
    return {name: fn(items, labels) for name, fn in policies.items()}


def label_dependence(items: list[dict], unit_key: str) -> dict[str, Any]:
    """How much of the label a unit already determines, which decides whether it can be a unit.

    This exists because the first version of this analysis reported two headline results that were
    artefacts of exactly this. BeaverTails was grouped by harm category, which is annotated only on
    unsafe items, so the group "none" is precisely the safe set and per-unit accuracy is 1.0 by
    construction. ToxicChat was grouped by a jailbreak flag that is never set on a non-toxic item.
    In both cases the unit was a second copy of the label, the per-unit best was trivially perfect,
    and the permutation test duly returned p < 0.001 for a structure that no method could exploit at
    inference time because the annotation does not exist then.

    A unit that partitions the labels perfectly is not a covariate. Reported here so the analysis
    refuses it rather than celebrating it.
    """
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item[unit_key]].append(index)
    degenerate = []
    for unit, indices in groups.items():
        rates = Counter(items[i]["_y"] for i in indices)
        if len(rates) == 1:
            degenerate.append(str(unit))
    return {
        "units": len(groups),
        "units_with_a_single_label": sorted(degenerate),
        "fraction_of_items_in_single_label_units": round(
            sum(len(groups[u]) for u in groups if len({items[i]["_y"] for i in groups[u]}) == 1)
            / len(items), 4),
        # Any single-class unit contributes a guaranteed 1.0 to the virtual best, so it inflates
        # headroom whether or not the other units are mixed. Requiring every unit to be degenerate
        # was too weak a rule: ToxicChat has exactly one such unit, its jailbreak flag, which is
        # never set on a non-toxic item, and that alone manufactured a significant result.
        "unit_determines_label": bool(degenerate),
        "every_unit_is_single_label": len(degenerate) == len(groups),
    }


def _items_driving_the_gap(
    kept: dict[Any, list[int]], predictions: dict[str, list[int]], items: list[dict]
) -> dict[str, Any]:
    """How many item-level decisions produce the headroom, per unit.

    A gap expressed in basis points over a small unit can rest on a handful of items. Reporting the
    count alongside it stops a reader inferring a population effect from five decisions.
    """
    truth = [i["_y"] for i in items]

    def score(name: str, idx: list[int]) -> float:
        return sum(predictions[name][i] == truth[i] for i in idx) / len(idx)

    shared = max(
        predictions, key=lambda n: sum(score(n, ix) for ix in kept.values()) / len(kept)
    )
    out: dict[str, Any] = {"best_shared_policy": shared, "per_unit": {}}
    total = 0
    for unit, idx in kept.items():
        best = max(predictions, key=lambda n: score(n, idx))
        extra = round((score(best, idx) - score(shared, idx)) * len(idx))
        total += extra
        out["per_unit"][str(unit)] = {
            "items": len(idx),
            "unit_best_policy": best,
            "items_the_unit_best_gets_right_that_the_shared_best_does_not": extra,
        }
    out["total_items_driving_the_gap"] = total
    return out


def headroom(
    items: list[dict],
    predictions: dict[str, list[int]],
    unit_key: str,
    min_unit_size: int = 20,
) -> dict[str, Any]:
    """Virtual-best minus single-best over the declared policy set, with subgroups as units."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[item[unit_key]].append(index)
    # Units too small to estimate anything are dropped, and the drop is reported.
    kept = {u: idx for u, idx in groups.items() if len(idx) >= min_unit_size}
    dropped = {str(u): len(idx) for u, idx in groups.items() if len(idx) < min_unit_size}
    if len(kept) < 2:
        return {"units": len(kept), "insufficient": True, "dropped_units": dropped}

    # A retained unit containing a single class contributes a guaranteed 1.0 to the virtual best,
    # so it manufactures headroom regardless of what the policies do. A dropped one cannot, which
    # is why this is checked here rather than over every group.
    single_label = sorted(
        str(u) for u, idx in kept.items() if len({items[i]["_y"] for i in idx}) == 1
    )
    if single_label:
        return {
            "units": len(kept),
            "label_derived_units": True,
            "single_label_units": single_label,
            "dropped_units": dropped,
            "reason": (
                "retained units contain a single class, so per-unit accuracy there is 1.0 by "
                "construction and any headroom measured is annotation redundancy rather than "
                "structure a method could exploit at inference time"
            ),
        }

    def accuracy(name: str, indices: list[int]) -> float:
        preds = predictions[name]
        return sum(preds[i] == items[i]["_y"] for i in indices) / len(indices)

    per_unit_best = {u: max(accuracy(n, idx) for n in predictions) for u, idx in kept.items()}
    shared = {n: sum(accuracy(n, idx) for idx in kept.values()) / len(kept) for n in predictions}
    best_shared_name = max(shared, key=lambda n: shared[n])
    virtual_best = sum(per_unit_best.values()) / len(kept)
    gap = virtual_best - shared[best_shared_name]

    # Cluster bootstrap over units, so the interval reflects which subgroups were seen.
    rng = random.Random(SEED)
    unit_names = list(kept)
    draws = []
    for _ in range(ROUNDS):
        picked = [unit_names[rng.randrange(len(unit_names))] for _ in unit_names]
        vb = sum(per_unit_best[u] for u in picked) / len(picked)
        sh = max(sum(accuracy(n, kept[u]) for u in picked) / len(picked) for n in predictions)
        draws.append(vb - sh)
    draws.sort()

    # Nonnegative support does not force an interval to include zero. Per-unit
    # maximization creates selection bias, so use an explicit permutation null
    # for association, conditional on an appropriate exchangeability design.
    observed_gap = gap
    sizes = [len(idx) for idx in kept.values()]
    flat = [i for idx in kept.values() for i in idx]
    permutation_rng = random.Random(SEED + 1)
    at_least_as_large = 0
    null_gaps = []
    for _ in range(ROUNDS):
        shuffled = flat[:]
        permutation_rng.shuffle(shuffled)
        blocks, cursor = [], 0
        for size in sizes:
            blocks.append(shuffled[cursor:cursor + size])
            cursor += size
        vb = sum(max(accuracy(n, b) for n in predictions) for b in blocks) / len(blocks)
        sh = max(sum(accuracy(n, b) for b in blocks) / len(blocks) for n in predictions)
        null_gaps.append(vb - sh)
        at_least_as_large += (vb - sh) >= observed_gap
    permutation_p = (at_least_as_large + 1) / (ROUNDS + 1)
    null_mean = sum(null_gaps) / len(null_gaps)

    # A zero-concentrated discrete null is not automatically invalid. Retain
    # this diagnostic (including its legacy field name), but do not use an
    # arbitrary 90% threshold to veto the nominal permutation comparison.
    null_zero_fraction = sum(abs(g) < 1e-12 for g in null_gaps) / len(null_gaps)
    null_distinct_values = len({round(g, 10) for g in null_gaps})
    null_is_degenerate = null_zero_fraction >= 0.90

    # Item-weighted, because equal unit weights let a small unit dominate.
    total = sum(len(idx) for idx in kept.values())
    per_unit_best_weighted = sum(
        per_unit_best[u] * len(kept[u]) for u in kept
    ) / total
    shared_weighted = max(
        sum(accuracy(n, kept[u]) * len(kept[u]) for u in kept) / total for n in predictions
    )
    item_weighted = per_unit_best_weighted - shared_weighted

    per_unit_choice = {
        str(u): max(predictions, key=lambda n: accuracy(n, idx)) for u, idx in kept.items()
    }
    return {
        "units": len(kept),
        "min_unit_size": min_unit_size,
        "unit_sizes": {str(u): len(idx) for u, idx in kept.items()},
        "dropped_units": dropped,
        "policies": sorted(predictions),
        "best_shared_policy": best_shared_name,
        "best_shared_score": round(shared[best_shared_name], 4),
        "virtual_best_score": round(virtual_best, 4),
        "headroom_bps": round(gap * 10000),
        "headroom_item_weighted_bps": round(item_weighted * 10000),
        "permutation_null_mean_bps": round(null_mean * 10000),
        "headroom_above_null_mean_bps": round((gap - null_mean) * 10000),
        "averaging": (
            "Units are weighted equally, not by size, so a small unit counts as much as a large "
            "one. The item-weighted figure is reported beside it because the two differ by several "
            "times where unit sizes are uneven."
        ),
        "selection_bias": 'The per-unit maximum is selected and scored on the same items and is optimistically biased. The null mean is a reference under shuffled membership, not an unbiased estimate of selection bias or of deployable adaptive benefit.',
        # With very few units the resampled gap can only take a handful of values, so the interval
        # is an artefact of the design rather than a measurement. It is omitted rather than shipped.
        "headroom_ci_bps": (
            [round(draws[int(0.025 * ROUNDS)] * 10000),
             round(draws[int(0.975 * ROUNDS)] * 10000)]
            if len(kept) >= 5 else None
        ),
        "permutation_p": round(permutation_p, 4),
        "exceeds_chance_structure": permutation_p < 0.05,
        "permutation_null_zero_fraction": round(null_zero_fraction, 4),
        "permutation_null_distinct_values": null_distinct_values,
        "permutation_null_is_degenerate": null_is_degenerate,
        "permutation_null_note": ('The sampled null is concentrated at zero. This does not by itself invalidate the nominal permutation p-value. Exchangeability, dependence, grouping, and policy-selection assumptions still require justification.' if null_is_degenerate else 'The p-value is a nominal comparison against shuffled partitions, conditional on exchangeability; it is not an effect-size estimate or proof of adaptive benefit.'),
        "items_driving_the_gap": _items_driving_the_gap(kept, predictions, items),
        "note_on_the_interval": 'Nonnegative support does not force a confidence interval to include zero. Per-unit maximization creates selection bias; the nominal permutation p-value addresses association under exchangeability, not held-out adaptive benefit.',
        "per_unit_best_policy": per_unit_choice,
        "units_whose_best_differs_from_shared": sorted(
            u for u, n in per_unit_choice.items() if n != best_shared_name
        ),
    }


# ------------------------------------------------------------------ policies


def constant(value: int) -> Callable[[list[dict], list[int]], list[int]]:
    return lambda items, labels: [value] * len(items)


def global_majority(items: list[dict], labels: list[int]) -> list[int]:
    return [_majority(labels)] * len(items)


def lookup_on(field: str) -> Callable[[list[dict], list[int]], list[int]]:
    """Cross-validated exact-match lookup on one categorical field."""

    def run(items: list[dict], labels: list[int]) -> list[int]:
        n = len(items)
        idx = list(range(n))
        random.Random(SEED).shuffle(idx)
        out = [0] * n
        for fold in range(5):
            test = set(idx[fold::5])
            train = [i for i in idx if i not in test]
            table: dict[Any, list[int]] = defaultdict(list)
            for i in train:
                table[items[i].get(field)].append(labels[i])
            fallback = _majority([labels[i] for i in train])
            for i in test:
                seen = table.get(items[i].get(field))
                out[i] = _majority(seen) if seen else fallback
        return out

    return run


def nearest_neighbour_on(field: str, cap: int) -> Callable[[list[dict], list[int]], list[int]]:
    """Cross-validated token-overlap 1-NN over one text field."""

    def run(items: list[dict], labels: list[int]) -> list[int]:
        n = len(items)
        bags = [tokens(items[i].get(field, "")) for i in range(n)]
        idx = list(range(n))
        random.Random(SEED).shuffle(idx)
        out = [0] * n
        for fold in range(5):
            test = [i for i in idx[fold::5]]
            train = [i for i in idx if i not in set(test)][:cap]
            protos = [(bags[i], labels[i]) for i in train]
            for i in test:
                q = bags[i]
                out[i] = max(protos, key=lambda p: len(q & p[0]) / max(1, len(q | p[0])))[1]
        return out

    return run


# ------------------------------------------------------------------ loaders


def load_rjudge(root: Path) -> tuple[list[dict], str] | None:
    directory = root / "survey" / "rjudge"
    if not directory.exists():
        return None
    items = []
    for path in sorted(directory.glob("*.json")):
        try:
            block = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(block, list):
            for r in block:
                if "label" in r:
                    items.append(
                        {
                            "_y": int(r["label"]),
                            "scenario": r.get("scenario"),
                            "attack_type": r.get("attack_type"),
                        }
                    )
    return (items, "scenario") if items else None


def load_beaver(root: Path, cap: int) -> tuple[list[dict], str] | None:
    path = root / "survey" / "beaver.json"
    if not path.exists():
        return None
    items = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        category = r["category"]
        if isinstance(category, str):
            try:
                category = ast.literal_eval(category)
            except Exception:
                category = {}
        flagged = sorted(k for k, v in (category or {}).items() if v)
        items.append(
            {
                "_y": 0 if str(r["is_safe"]) == "True" else 1,
                "prompt": r["prompt"],
                "response": r["response"],
                "harm": flagged[0] if flagged else "none",
            }
        )
    random.Random(SEED).shuffle(items)
    return items[:cap], "harm"


def load_toxicchat(root: Path, cap: int) -> tuple[list[dict], str] | None:
    path = root / "survey" / "toxicchat.json"
    if not path.exists():
        return None
    items = [
        {
            "_y": int(r["toxicity"]),
            "user_input": r["user_input"],
            "jailbreak": int(r["jailbreaking"]),
        }
        for r in json.loads(path.read_text(encoding="utf-8"))
    ]
    random.Random(SEED).shuffle(items)
    return items[:cap], "jailbreak"


def load_mind2web(root: Path) -> tuple[list[dict], str] | None:
    path = root / "m2w" / "seeact" / "sample_labeled_all.json"
    if not path.exists():
        return None
    items = []
    for r in json.loads(path.read_text(encoding="utf-8")):
        u = r["user_info"]
        items.append(
            {
                "_y": int(r["label"]),
                "domain": r["domain"],
                "profile": (u["domestic"], u["dr_license"], u["vaccine"], u["membership"]),
                "adult": u["age"] >= 18,
            }
        )
    return items, "domain"


def load_eicu(root: Path) -> tuple[list[dict], str] | None:
    path = root / "eicu" / "ehragent" / "eicu_ac.json"
    if not path.exists():
        return None
    items = [
        {"_y": int(r["label"]), "role": r["identity"], "template": r["q_tag"]}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if r.get("split") == "valid"
    ]
    return items, "role"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cap", type=int, default=1500)
    args = ap.parse_args()

    root = Path(args.root)
    report: dict[str, Any] = {"seed": SEED, "rounds": ROUNDS, "benchmarks": {}, "not_run": []}

    specs = [
        (
            "R-Judge",
            load_rjudge(root),
            {
                "always allow": constant(0),
                "always block": constant(1),
                "global majority": global_majority,
                "attack type lookup": lookup_on("attack_type"),
            },
        ),
        (
            "Mind2Web-SC",
            load_mind2web(root),
            {
                "always allow": constant(0),
                "always block": constant(1),
                "global majority": global_majority,
                "profile lookup": lookup_on("profile"),
                "adult only": lookup_on("adult"),
            },
        ),
        (
            "EICU-AC",
            load_eicu(root),
            {
                "always allow": constant(0),
                "always block": constant(1),
                "global majority": global_majority,
                "template lookup": lookup_on("template"),
            },
        ),
        (
            "BeaverTails",
            load_beaver(root, args.cap),
            {
                "always safe": constant(0),
                "always unsafe": constant(1),
                "global majority": global_majority,
                "response 1-NN": nearest_neighbour_on("response", args.cap),
                "prompt 1-NN": nearest_neighbour_on("prompt", args.cap),
            },
        ),
        (
            "ToxicChat",
            load_toxicchat(root, args.cap),
            {
                "always benign": constant(0),
                "always toxic": constant(1),
                "global majority": global_majority,
                "input 1-NN": nearest_neighbour_on("user_input", args.cap),
            },
        ),
    ]

    for name, loaded, policies in specs:
        if loaded is None:
            report["not_run"].append({"benchmark": name, "reason": "dataset not fetched"})
            continue
        items, unit = loaded
        dependence = label_dependence(items, unit)
        predictions = policy_scores(items, policies)
        result = headroom(items, predictions, unit)
        result["items"] = len(items)
        result["unit_field"] = unit
        result["label_dependence"] = dependence
        if result.get("label_derived_units"):
            report["not_run"].append({
                "benchmark": name,
                "unit_field": unit,
                "label_derived_units": True,
                "reason": result["reason"],
                "single_label_units": result["single_label_units"],
                "label_dependence": dependence,
            })
            continue
        # One verdict here turned on the threshold, so the sensitivity ships with the result.
        sensitivity = {}
        for size in (10, 20):
            at_size = headroom(items, predictions, unit, min_unit_size=size)
            if at_size.get("insufficient"):
                continue
            sensitivity[str(size)] = {
                "units": at_size["units"],
                "headroom_bps": at_size["headroom_bps"],
                "permutation_p": at_size["permutation_p"],
            }
        result["sensitivity_to_min_unit_size"] = sensitivity
        report["benchmarks"][name] = result

    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"{'benchmark':13s} {'units':>5s} {'items':>6s} {'headroom':>9s} {'perm p':>8s}  best shared"
    )
    for name, r in report["benchmarks"].items():
        if r.get("insufficient"):
            print(f"{name:13s}  too few units to compute")
            continue
        mark = "*" if r["exceeds_chance_structure"] else " "
        print(
            f"{name:13s} {r['units']:5d} {r['items']:6d} {r['headroom_bps']:8d} "
            f"{r['permutation_p']:8.3f}{mark} {r['best_shared_policy']}"
        )
    for entry in report["not_run"]:
        print(f"{entry['benchmark']:13s}  {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
