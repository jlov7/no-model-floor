"""No-model floors across published safety benchmarks, so the claim stops resting on two papers.

Section 3.10 computes floors for the two benchmarks used by GuardAgent and AGrail, and concludes
that a floor high enough to matter goes unreported there. A review made the obvious objection: a
sample of two cannot support a statement about a field, and the whole contribution rests on that
statement. This widens the sample.

The floor of a benchmark is the best score obtainable by a policy that never runs the model being
evaluated. Which policies are available depends on the benchmark, so they are named per benchmark
rather than assumed, and grouped by how much they are allowed to know:

  majority        predict the commonest label. Uses test labels, so it is the weakest possible
                  policy and the conventional trivial baseline at once.
  metadata        read fields shipped alongside the task but not the task itself: a user profile,
                  a role, an attack type.
  partial input   read part of the input and delete the rest, the standard partial-input ablation
                  from natural language inference and visual question answering.

None calls a model. The metadata and partial-input policies are nearest-neighbour lookups fitted to
training labels, which makes them stronger than a constant policy and weaker than anything learned;
they are reported separately from the majority baseline for that reason.

Datasets are not redistributed. Users must provide authorized local copies; this script writes only
aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

FOLDS = 5
SEED = 20260819
TOKEN = re.compile(r"[a-z']+")


def tokens(text: Any) -> frozenset[str]:
    return frozenset(TOKEN.findall(str(text).lower()))


def majority_rate(labels: list[int]) -> float:
    return max(Counter(labels).values()) / len(labels)


def lookup_cv(records: list[dict], key: Callable[[dict], Any], labels: list[int]) -> float:
    """Cross-validated exact-match lookup on a categorical key."""
    idx = list(range(len(records)))
    random.Random(SEED).shuffle(idx)
    correct = 0
    for fold in range(FOLDS):
        test = [i for j, i in enumerate(idx) if j % FOLDS == fold]
        train = [i for j, i in enumerate(idx) if j % FOLDS != fold]
        table: dict[Any, list[int]] = {}
        for i in train:
            table.setdefault(key(records[i]), []).append(labels[i])
        fallback = Counter(labels[i] for i in train).most_common(1)[0][0]
        for i in test:
            seen = table.get(key(records[i]))
            guess = Counter(seen).most_common(1)[0][0] if seen else fallback
            correct += guess == labels[i]
    return correct / len(records)


def nearest_neighbour_cv(records: list[dict], field: str, labels: list[int], cap: int) -> float:
    """Cross-validated token-overlap 1-NN over one text field, capped for tractability.

    The cap has to be applied to a shuffled index over the whole dataset, then truncated -- not the
    other way around. Shuffling only `range(n)` after `n` was already capped reorders just the first
    `cap` records among themselves and silently drops the rest, which is the same head-truncation
    bug `external_headroom.py`'s errata records as fixed there.
    """
    order = list(range(len(records)))
    random.Random(SEED).shuffle(order)
    idx = order[:cap]
    n = len(idx)
    bags = [tokens(records[i][field]) for i in idx]
    lbls = [labels[i] for i in idx]
    correct = 0
    for fold in range(FOLDS):
        test = [j for j in range(n) if j % FOLDS == fold]
        train = [j for j in range(n) if j % FOLDS != fold]
        protos = [(bags[j], lbls[j]) for j in train]
        for j in test:
            q = bags[j]
            best = max(protos, key=lambda p: len(q & p[0]) / max(1, len(q | p[0])))
            correct += best[1] == lbls[j]
    return correct / n


def summarise(
    name: str,
    note: str,
    labels: list[int],
    policies: dict[str, float],
    published: dict[str, float] | None = None,
) -> dict[str, Any]:
    majority = majority_rate(labels)
    everything = {"majority class": majority, **policies}
    floor_name = max(everything, key=lambda k: everything[k])
    floor = everything[floor_name]

    # Accuracy is the wrong scale for a benchmark labelled 1 for the class of interest (unsafe,
    # toxic) whose base rate is far from 50%: a constant policy can post a high accuracy while
    # catching nothing, and R-Judge's own paper reports F1 rather than accuracy for this reason.
    # "always predict positive" gets precision equal to the positive rate and recall of 1, so its
    # F1 is 2*pi / (1 + pi) in closed form; recorded alongside the positive rate rather than
    # folded into `no_model_floor`, which stays an accuracy-only statistic.
    positive_rate = sum(1 for y in labels if y == 1) / len(labels)
    always_positive_f1 = 2 * positive_rate / (1 + positive_rate) if positive_rate > 0 else 0.0

    out: dict[str, Any] = {
        "benchmark": name,
        "note": note,
        "items": len(labels),
        "label_counts": dict(Counter(labels)),
        "policies": {k: round(v, 4) for k, v in everything.items()},
        "no_model_floor": round(floor, 4),
        "floor_policy": floor_name,
        "majority_class": round(majority, 4),
        "floor_above_majority_points": round((floor - majority) * 100, 1),
        "positive_rate": round(positive_rate, 4),
        "always_positive_f1": round(always_positive_f1, 4),
    }
    if published:
        out["published"] = published
        out["published_below_floor"] = sorted(k for k, v in published.items() if v < floor)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="directory holding the fetched datasets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nn-cap", type=int, default=2500)
    args = parser.parse_args()

    root = Path(args.root)
    surveyed: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    # ---- R-Judge: agent trajectories labelled safe/unsafe, with scenario and attack type.
    rjudge = sorted((root / "rjudge").glob("*.json")) if (root / "rjudge").exists() else []
    records = []
    for path in rjudge:
        try:
            block = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(block, list):
            records += [r for r in block if "label" in r]
    if records:
        labels = [int(r["label"]) for r in records]
        surveyed.append(
            summarise(
                "R-Judge",
                "agent trajectory judged safe or unsafe; metadata is scenario and attack type",
                labels,
                {
                    "scenario only": lookup_cv(records, lambda r: r.get("scenario"), labels),
                    "attack type only": lookup_cv(records, lambda r: r.get("attack_type"), labels),
                    "scenario x attack type": lookup_cv(
                        records, lambda r: (r.get("scenario"), r.get("attack_type")), labels
                    ),
                },
            )
        )
    else:
        skipped.append({"benchmark": "R-Judge", "reason": "not fetched"})

    # ---- BeaverTails: prompt/response pairs labelled safe or unsafe.
    path = root / "beaver.json"
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
        labels = [0 if str(r["is_safe"]) == "True" else 1 for r in records]
        surveyed.append(
            summarise(
                "BeaverTails",
                "prompt and response labelled safe or unsafe; partial-input ablations",
                labels,
                {
                    "response only, prompt deleted": nearest_neighbour_cv(
                        records, "response", labels, args.nn_cap
                    ),
                    "prompt only, response deleted": nearest_neighbour_cv(
                        records, "prompt", labels, args.nn_cap
                    ),
                },
            )
        )
    else:
        skipped.append({"benchmark": "BeaverTails", "reason": "not fetched"})

    # ---- ToxicChat: user turns labelled toxic or not, heavily imbalanced.
    path = root / "toxicchat.json"
    if path.exists():
        records = json.loads(path.read_text(encoding="utf-8"))
        labels = [int(r["toxicity"]) for r in records]
        surveyed.append(
            summarise(
                "ToxicChat",
                "user turn labelled toxic; accuracy is the wrong metric here and the floor "
                "shows why",
                labels,
                {
                    "user input only": nearest_neighbour_cv(
                        records, "user_input", labels, args.nn_cap
                    )
                },
            )
        )
    else:
        skipped.append({"benchmark": "ToxicChat", "reason": "not fetched"})

    report = {
        "note": "Aggregate statistics only. No dataset is redistributed.",
        "folds": FOLDS,
        "seed": SEED,
        "nearest_neighbour_cap": args.nn_cap,
        "benchmarks": surveyed,
        "not_surveyed": skipped,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"{'benchmark':14s} {'n':>6s} {'majority':>9s} {'floor':>7s} {'gap':>6s}  floor policy")
    for entry in surveyed:
        print(
            f"{entry['benchmark']:14s} {entry['items']:6d} {entry['majority_class']:9.1%} "
            f"{entry['no_model_floor']:7.1%} {entry['floor_above_majority_points']:+6.1f}  "
            f"{entry['floor_policy']}"
        )
    for entry in skipped:
        print(f"{entry['benchmark']:14s}  skipped: {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
