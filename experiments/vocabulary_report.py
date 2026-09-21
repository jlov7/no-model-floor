"""What the seven-vocabulary sweep says about the one-pair result.

Section 3.11 measured a single contrast, finance against maintenance, and could not tell whether it
reflected something general about rewording or was a property of those two surfaces. This scores all
seven vocabularies under identical conditions and asks three things:

  1. Where does the original pair sit among all 21 pairs? If its gap is typical, the one-pair
     result generalises. If it is extreme, it was a lucky draw and section 3.11 overstates.
  2. Is finance the top? The familiarity explanation predicts every other vocabulary scores below
     it. One vocabulary scoring higher falsifies that explanation directly.
  3. What does the nonsense vocabulary do? It carries no meaning, so it is the control that
     separates "models are responding to how familiar the words are" from "models are responding to
     what the words suggest".

Intervals are cell-clustered on the eighteen cells, resampled jointly across vocabularies so each
contrast stays paired, matching the rest of this repository.

Exploratory. Not preregistered.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from runs import completed

SEED = 20260819
ROUNDS = 10000
ORDER = ["finance", "maintenance", "clinical", "logistics", "legal", "aviation", "nonsense"]


CELLS = 18


def _every_cell_has_draws(run) -> str | None:
    """Presence of a file is not evidence that it holds anything."""
    for vocabulary in ORDER:
        records = run.read(f"raw-{vocabulary}.json")
        cells = {int(r["cell_id"].rsplit("-", 1)[1]) - 1 for r in records if "decision" in r}
        if len(cells) < CELLS:
            return f"{vocabulary} produced draws for {len(cells)} of {CELLS} cells"
    return None


def load(root: Path) -> tuple[dict[str, dict[str, dict[int, list[dict]]]], list[dict[str, str]]]:
    """model -> vocabulary -> cell index -> draws, plus what was rejected and why."""
    accepted, rejected = completed(
        root,
        "vocabulary-sweep.json",
        artefacts=[f"raw-{v}.json" for v in ORDER],
        require=_every_cell_has_draws,
    )
    out: dict[str, dict[str, dict[int, list[dict]]]] = {}
    for run in accepted:
        per_vocab: dict[str, dict[int, list[dict]]] = {}
        for vocabulary in ORDER:
            cells: dict[int, list[dict]] = defaultdict(list)
            for record in run.read(f"raw-{vocabulary}.json"):
                if "decision" in record:
                    cells[int(record["cell_id"].rsplit("-", 1)[1]) - 1].append(record)
            per_vocab[vocabulary] = dict(cells)
        out[run.manifest["model"]] = per_vocab
    return out, rejected


def statistic(draws: list[dict], kind: str) -> float:
    if not draws:
        return 0.0
    if kind == "accuracy":
        return sum(d["decision"] == d["expected"] for d in draws) / len(draws)
    return sum(d["decision"] == "ASK" for d in draws) / len(draws)


def score(cells: dict[int, list[dict]], picks: list[int], kind: str) -> float:
    return sum(statistic(cells.get(i, []), kind) for i in picks) / len(picks)


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * len(ordered))], ordered[
        min(len(ordered) - 1, int(0.975 * len(ordered)))
    ]


def analyse(pooled: dict[str, dict[int, list[dict]]], kind: str) -> dict[str, Any]:
    cells = 18
    observed = {v: score(pooled[v], list(range(cells)), kind) for v in ORDER}

    rng = random.Random(SEED)
    pair_boots: dict[tuple[str, str], list[float]] = {
        p: [] for p in itertools.combinations(ORDER, 2)
    }
    for _ in range(ROUNDS):
        picks = [rng.randrange(cells) for _ in range(cells)]
        s = {v: score(pooled[v], picks, kind) for v in ORDER}
        for a, b in pair_boots:
            pair_boots[(a, b)].append(s[b] - s[a])

    # Simultaneous band over all 21 contrasts, from the same replicates. Twenty-one intervals at
    # a nominal 95% each are not 95% jointly, and reporting "excludes zero" across all of them
    # without adjustment overstates how many differences are real. Taking, in each replicate, the
    # largest absolute deviation of any contrast from its observed value gives a simultaneous band
    # whose coverage is joint rather than per-pair. It is an unstudentised max-|deviation| band,
    # not a max-t: deviations are not rescaled by per-contrast standard errors before the maximum
    # is taken, so contrasts with different sampling noise are pooled on one scale.
    keys = list(pair_boots)
    centred_max = [
        max(abs(pair_boots[k][i] - (observed[k[1]] - observed[k[0]])) for k in keys)
        for i in range(ROUNDS)
    ]
    band = sorted(centred_max)[int(0.95 * ROUNDS)]

    pairs = {}
    for (a, b), values in pair_boots.items():
        lo, hi = interval(values)
        difference = observed[b] - observed[a]
        pairs[f"{a}->{b}"] = {
            "difference_bps": round(difference * 10000),
            "ci_bps": [round(lo * 10000), round(hi * 10000)],
            "excludes_zero": lo > 0 or hi < 0,
            "simultaneous_ci_bps": [
                round((difference - band) * 10000), round((difference + band) * 10000)
            ],
            "excludes_zero_simultaneously": abs(difference) > band,
        }

    gaps = {k: abs(v["difference_bps"]) for k, v in pairs.items()}
    original = abs(pairs["finance->maintenance"]["difference_bps"])
    ranked = sorted(gaps.values(), reverse=True)
    ordered_by_score = sorted(ORDER, key=lambda v: observed[v], reverse=True)

    # The rank of the original pair among the 21 is itself an estimate, and quoting it as a bare
    # integer implies a precision the design does not have.
    ranks = []
    for i in range(ROUNDS):
        replicate = {k: abs(pair_boots[k][i]) for k in keys}
        target = replicate[("finance", "maintenance")]
        ranks.append(sorted(replicate.values(), reverse=True).index(target) + 1)
    rank_lo, rank_hi = interval([float(r) for r in ranks])

    return {
        "per_vocabulary_bps": {v: round(observed[v] * 10000) for v in ORDER},
        "ranking_best_first": ordered_by_score,
        "finance_rank": ordered_by_score.index("finance") + 1,
        "nonsense_rank": ordered_by_score.index("nonsense") + 1,
        "spread_bps": round((max(observed.values()) - min(observed.values())) * 10000),
        "original_pair_gap_bps": original,
        "original_pair_rank_among_21": ranked.index(original) + 1,
        "original_pair_rank_95ci": [int(rank_lo), int(rank_hi)],
        "pairs_whose_simultaneous_interval_excludes_zero": sum(
            1 for p in pairs.values() if p["excludes_zero_simultaneously"]
        ),
        "simultaneous_band_bps": round(band * 10000),
        "median_pair_gap_bps": ranked[len(ranked) // 2],
        "pairs_whose_interval_excludes_zero": sum(1 for p in pairs.values() if p["excludes_zero"]),
        "pairs": pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data, excluded = load(Path(args.root))
    if not data:
        print("no completed sweeps found")
        return 1

    pooled: dict[str, dict[int, list[dict]]] = {v: defaultdict(list) for v in ORDER}
    for per_vocab in data.values():
        for vocabulary, cells in per_vocab.items():
            for cell, draws in cells.items():
                pooled[vocabulary][cell].extend(draws)
    pooled = {v: dict(c) for v, c in pooled.items()}

    report = {
        "models": sorted(data),
        "excluded_runs": excluded,
        "rounds": ROUNDS,
        "seed": SEED,
        "preregistered": False,
        "pooled": {kind: analyse(pooled, kind) for kind in ("accuracy", "ask_rate")},
        "per_model_accuracy_bps": {
            model: {
                v: round(score(per_vocab[v], list(range(18)), "accuracy") * 10000) for v in ORDER
            }
            for model, per_vocab in sorted(data.items())
        },
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for record in excluded:
        print(f"excluded {record['run']}: {record['reason']}")

    for kind in ("accuracy", "ask_rate"):
        r = report["pooled"][kind]
        print(f"\n{kind}, pooled over {len(data)} models")
        for v in r["ranking_best_first"]:
            mark = (
                "  <- control"
                if v == "nonsense"
                else ("  <- original pair" if v in ("finance", "maintenance") else "")
            )
            print(f"    {v:12s} {r['per_vocabulary_bps'][v]:5d} bps{mark}")
        print(
            f"    spread {r['spread_bps']} bps; finance ranks {r['finance_rank']} of 7; "
            f"nonsense ranks {r['nonsense_rank']} of 7"
        )
        print(
            f"    the original pair's gap is {r['original_pair_gap_bps']} bps, "
            f"rank {r['original_pair_rank_among_21']} of 21 pairs "
            f"(median pair gap {r['median_pair_gap_bps']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
