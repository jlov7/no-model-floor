"""Main effects and interaction from the 2x2 surface factorial.

Reads the per-model runs produced by `factorial.py` and reports, for accuracy and for ask rate:

  vocabulary effect    maintenance minus finance, averaged over the distractor levels
  distractor effect    distractors present minus absent, averaged over the vocabulary levels
  interaction          whether the distractor effect depends on the vocabulary

Intervals are cell-clustered, matching the rest of this repository: the eighteen scenario cells are
the unit of analysis, not the individual draws, because draws within a cell are correlated. Cells
are resampled jointly across all four arms so the comparison stays paired.

Exploratory. Not preregistered, built after seeing the section 3.4 result, reported as such.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from runs import completed

SEED = 20260819
ROUNDS = 10000
ARMS = ("finance-clean", "finance-distractors", "maintenance-clean", "maintenance-distractors")


CELLS = 18


def _every_arm_has_draws(run) -> str | None:
    """Every arm must have draws for every cell, not merely one usable draw somewhere.

    `statistic()` returns 0.0 for a cell with no draws rather than raising, so a partially failed
    arm would be scored as a run of zeros and bias the effects instead of being rejected. The
    vocabulary loader already required completeness per cell; this one did not, and the asymmetry
    was an accident rather than a decision.
    """
    for arm in ARMS:
        records = run.read(f"raw-{arm}.json")
        cells = {int(r["cell_id"].rsplit("-", 1)[1]) - 1 for r in records if "decision" in r}
        if len(cells) < CELLS:
            return f"{arm} produced draws for {len(cells)} of {CELLS} cells"
    return None


def load(root: Path) -> tuple[dict[str, dict[str, dict[int, list[dict]]]], list[dict[str, str]]]:
    """model -> arm -> cell index -> draws, plus what was rejected and why."""
    accepted, rejected = completed(
        root,
        "factorial.json",
        artefacts=[f"raw-{a}.json" for a in ARMS],
        require=_every_arm_has_draws,
    )
    out: dict[str, dict[str, dict[int, list[dict]]]] = {}
    for run in accepted:
        per_arm: dict[str, dict[int, list[dict]]] = {}
        for arm in ARMS:
            cells: dict[int, list[dict]] = defaultdict(list)
            for record in run.read(f"raw-{arm}.json"):
                if "decision" in record:
                    cells[int(record["cell_id"].rsplit("-", 1)[1]) - 1].append(record)
            per_arm[arm] = dict(cells)
        out[run.manifest["model"]] = per_arm
    return out, rejected


def statistic(draws: list[dict], kind: str) -> float:
    if not draws:
        return 0.0
    if kind == "accuracy":
        return sum(d["decision"] == d["expected"] for d in draws) / len(draws)
    return sum(d["decision"] == "ASK" for d in draws) / len(draws)


def arm_score(cells: dict[int, list[dict]], picks: list[int], kind: str) -> float:
    vals = [statistic(cells.get(i, []), kind) for i in picks]
    return sum(vals) / len(vals)


def effects(
    per_arm: dict[str, dict[int, list[dict]]], picks: list[int], kind: str
) -> dict[str, float]:
    s = {arm: arm_score(per_arm[arm], picks, kind) for arm in ARMS}
    vocabulary = (
        (s["maintenance-clean"] + s["maintenance-distractors"])
        - (s["finance-clean"] + s["finance-distractors"])
    ) / 2
    distractors = (
        (s["finance-distractors"] + s["maintenance-distractors"])
        - (s["finance-clean"] + s["maintenance-clean"])
    ) / 2
    interaction = (s["maintenance-distractors"] - s["maintenance-clean"]) - (
        s["finance-distractors"] - s["finance-clean"]
    )
    return {
        "vocabulary": vocabulary,
        "distractors": distractors,
        "interaction": interaction,
        **{f"arm_{a}": s[a] for a in ARMS},
    }


def interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    lo = ordered[int(0.025 * len(ordered))]
    hi = ordered[min(len(ordered) - 1, int(0.975 * len(ordered)))]
    return lo, hi


def analyse(per_arm: dict[str, dict[int, list[dict]]], kind: str) -> dict[str, Any]:
    n_cells = 18
    observed = effects(per_arm, list(range(n_cells)), kind)
    rng = random.Random(SEED)
    boots: dict[str, list[float]] = {"vocabulary": [], "distractors": [], "interaction": []}
    for _ in range(ROUNDS):
        picks = [rng.randrange(n_cells) for _ in range(n_cells)]
        e = effects(per_arm, picks, kind)
        for key in boots:
            boots[key].append(e[key])
    out: dict[str, Any] = {f"arm_{a}_bps": round(observed[f"arm_{a}"] * 10000) for a in ARMS}
    for key in boots:
        lo, hi = interval(boots[key])
        out[key] = {
            "effect_bps": round(observed[key] * 10000),
            "ci_bps": [round(lo * 10000), round(hi * 10000)],
            "excludes_zero": lo > 0 or hi < 0,
        }
    # Returned for the joint band over all 36 intervals. Every analyse() call reseeds with SEED,
    # so replicate i is the same cell draw everywhere and the lists below are aligned row by row;
    # the band reads them without touching the random stream, so no committed interval moves.
    out["_boots"] = boots
    out["_observed"] = {k: observed[k] for k in boots}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data, rejected = load(Path(args.root))
    report: dict[str, Any] = {
        "models": {},
        "rounds": ROUNDS,
        "seed": SEED,
        "preregistered": False,
        "unit_of_analysis": "scenario cell",
        "excluded_runs": rejected,
    }
    for model, per_arm in data.items():
        report["models"][model] = {
            kind: analyse(per_arm, kind) for kind in ("accuracy", "ask_rate")
        }

    # Pool by treating every model's cells as one collection, which keeps the pairing.
    pooled: dict[str, dict[int, list[dict]]] = {arm: defaultdict(list) for arm in ARMS}
    for _, per_arm in sorted(data.items()):
        for arm in ARMS:
            for cell, draws in per_arm[arm].items():
                pooled[arm][cell].extend(draws)
    report["pooled"] = {
        kind: analyse({a: dict(pooled[a]) for a in ARMS}, kind) for kind in ("accuracy", "ask_rate")
    }

    # One simultaneous band over all 36 intervals this section reports: two outcomes x three
    # effects x (five models + pooled). An earlier version of PAPER.md quoted a "studentised
    # max-t" band with a critical value of 3.07 that no committed script computed; this is the
    # reproducible replacement, an unstudentised band built from the same replicates as the
    # intervals themselves. In each replicate the largest absolute deviation of any of the 36
    # estimates from its observed value is taken, and the 95th percentile of that maximum is the
    # band. Because every analyse() call reseeds identically, replicate i is the same cell draw
    # everywhere and the rows are aligned.
    members = []
    for model, kinds in report["models"].items():
        for kind, a in kinds.items():
            for effect in ("vocabulary", "distractors", "interaction"):
                members.append((f"{model}/{kind}/{effect}", effect, a))
    for kind, a in report["pooled"].items():
        for effect in ("vocabulary", "distractors", "interaction"):
            members.append((f"pooled/{kind}/{effect}", effect, a))
    centred_max = [
        max(abs(a["_boots"][effect][i] - a["_observed"][effect]) for _, effect, a in members)
        for i in range(ROUNDS)
    ]
    ordered = sorted(centred_max)
    band = ordered[int(0.95 * ROUNDS)]
    survivors = []
    for name, effect, a in members:
        if abs(a["_observed"][effect]) > band:
            survivors.append(name)
    report["joint_band"] = {
        "method": (
            "unstudentised simultaneous band over all 36 intervals: per replicate, the largest "
            "absolute deviation of any interval's estimate from its observed value; 95th "
            "percentile of that maximum across the same cell-clustered replicates the intervals "
            "use. Not studentised; each deviation is in its own statistic's units."
        ),
        "band_bps": round(band * 10000),
        "intervals": len(members),
        "surviving_count": len(survivors),
        "surviving": survivors,
    }

    def strip(a: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in a.items() if not k.startswith("_")}

    for kinds in report["models"].values():
        for kind in kinds:
            kinds[kind] = strip(kinds[kind])
    report["pooled"] = {kind: strip(a) for kind, a in report["pooled"].items()}

    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for kind in ("accuracy", "ask_rate"):
        print(f"\n{kind}")
        print(f"  {'model':24s} {'vocabulary':>22s} {'distractors':>22s} {'interaction':>22s}")
        for model in sorted(report["models"]):
            r = report["models"][model][kind]

            def fmt(k, r=r):
                e = r[k]
                star = "*" if e["excludes_zero"] else " "
                return f"{e['effect_bps']:+6d} [{e['ci_bps'][0]:+5d},{e['ci_bps'][1]:+5d}]{star}"

            print(
                f"  {model:24s} {fmt('vocabulary'):>22s} {fmt('distractors'):>22s} "
                f"{fmt('interaction'):>22s}"
            )
        r = report["pooled"][kind]

        def fmt2(k, r=r):
            e = r[k]
            star = "*" if e["excludes_zero"] else " "
            return f"{e['effect_bps']:+6d} [{e['ci_bps'][0]:+5d},{e['ci_bps'][1]:+5d}]{star}"

        print(
            f"  {'POOLED':24s} {fmt2('vocabulary'):>22s} {fmt2('distractors'):>22s} "
            f"{fmt2('interaction'):>22s}"
        )
    print("\n* marks a 95% cell-clustered interval excluding zero. Effects in basis points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
