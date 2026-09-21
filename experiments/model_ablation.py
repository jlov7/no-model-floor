"""How much of this benchmark's score is the model, and how much is the scaffold?

The dominance proof says one configuration is optimal for every possible model. A hostile reader
should immediately ask why that is interesting, since the guard set turns out to implement the
ground-truth rule everywhere the rule is not ACT. It is a fair objection: on 8 of the 9 contexts the
guarded system returns the correct answer no matter what the model said.

That objection points at the diagnostic that should have existed first, and unlike the dominance
argument it needs no ground truth at all.

  Replace the model with something that carries no information, and re-score.
  Whatever score survives was never the model's to begin with.

This is checkable by anyone on any benchmark. It requires the ability to substitute the model's
output, and nothing else: no error taxonomy, no knowledge of whether the controls are complete, no
oracle. If a guarded system scores near its ceiling while driven by noise, the benchmark cannot
support a claim about the model, and it certainly cannot support a claim about adapting to the
model.

Substitutes, weakest assumption first:

  adversarial   for each cell, the decision and confidence that minimise the score. A floor on
                what the benchmark reports no matter how badly the model behaves.
  constant      always ACT, always ASK, always ABSTAIN. What a model with one stuck output scores.
  uniform       averaged over every decision and every confidence class, equally weighted.

The number that matters is the span between the adversarial floor and the ceiling: the only part of
the range a model can move. Everything below the floor is the scaffold talking.

Offline: re-scores committed draws and calls no model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exhaustive import applied, configurations, expected, label
from runs import evidence_runs
from theorem import DECISIONS, OFF_CONTRACT, confidence_classes

# Substitutes stand in for a model, so they are held to the output contract: the three contract
# decisions and confidences in 0..10000. The theorem's wider space, which includes an off-contract
# decision and negative confidence, exists to prove completeness and would misattribute a contract
# violation to model behaviour if it were used here.
CONTRACT_DECISIONS = tuple(d for d in DECISIONS if d != OFF_CONTRACT)


def contract_confidences() -> list[int]:
    return [c for c in confidence_classes() if 0 <= c <= 10000]

STAR = "invalid_authority+denied_approval+unknown_approval@0"


def star_config() -> dict[str, Any]:
    return next(c for c in configurations() if label(c) == STAR)


def correct(approval: str, authority: str, decision: str, confidence: int,
            config: dict[str, Any]) -> bool:
    return applied(decision, confidence, approval, authority, config) == expected(approval, authority)


def substitute_scores(cells: list[tuple[str, str]], config: dict[str, Any]) -> dict[str, float]:
    """Score of the guarded system under model substitutes that carry no information."""
    confs = contract_confidences()

    adversarial = sum(
        min(correct(a, u, d, c, config) for d in CONTRACT_DECISIONS for c in confs) for a, u in cells
    ) / len(cells)

    uniform = sum(
        sum(correct(a, u, d, c, config) for d in CONTRACT_DECISIONS for c in confs) / (len(CONTRACT_DECISIONS) * len(confs))
        for a, u in cells
    ) / len(cells)

    out = {"adversarial": adversarial, "uniform_random": uniform}
    for stuck in CONTRACT_DECISIONS:
        out[f"always_{stuck.lower()}"] = sum(
            sum(correct(a, u, stuck, c, config) for c in confs) / len(confs) for a, u in cells
        ) / len(cells)
    return out


def observed_score(draws: list[dict], config: dict[str, Any]) -> float:
    return sum(
        correct(d["approval"], d["authority"], d["decision"], d["confidence_bps"], config)
        for d in draws
    ) / len(draws)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exploratory", required=True)
    parser.add_argument("--confirmatory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from surface_v2 import cells as v2_cells
    permission = {"GRANTED": "APPROVED", "REFUSED": "DENIED", "PENDING": "UNKNOWN"}
    mandate = {"CURRENT": "VALID", "LAPSED": "EXPIRED", "ABSENT": "MISSING"}
    v2 = {c["cell_id"]: (permission[c["permission"]], mandate[c["mandate"]]) for c in v2_cells()}

    config = star_config()
    report: dict[str, Any] = {"configuration": STAR}

    for surface, root in (("exploratory", args.exploratory), ("confirmatory", args.confirmatory)):
        per_model, cells = {}, []
        for run in evidence_runs(root):
            meta, raw = run.manifest, run.read("raw-samples.json")
            draws, seen = [], []
            for r in raw:
                if "scenario_id" in r:
                    a, u, _ = r["scenario_id"].split("-")
                    a, u, cid = a.upper(), u.upper(), r["scenario_id"]
                else:
                    cid = r["cell_id"]
                    a, u = v2[cid]
                draws.append({"approval": a, "authority": u, "decision": r["decision"],
                              "confidence_bps": r["confidence_bps"]})
                seen.append((cid, a, u))
            per_model[meta["model"]] = observed_score(draws, config)
            if not cells:
                cells = [(a, u) for _, a, u in dict.fromkeys(seen)]

        subs = substitute_scores(cells, config)
        floor = subs["adversarial"]
        best_model = max(per_model.values())
        report[surface] = {
            "cells": len(cells),
            "observed_bps": {m: round(v * 10000) for m, v in per_model.items()},
            "substitute_bps": {k: round(v * 10000) for k, v in subs.items()},
            "adversarial_floor_bps": round(floor * 10000),
            "ceiling_bps": 10000,
            "model_movable_range_bps": round((1.0 - floor) * 10000),
            "fraction_of_score_that_is_model_independent": round(floor, 4),
            "best_model_share_of_movable_range": (
                round((best_model - floor) / (1.0 - floor), 4) if floor < 1.0 else None
            ),
        }

    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for surface in ("exploratory", "confirmatory"):
        r = report[surface]
        print(f"\n{surface}  ({r['cells']} cells, configuration {STAR})")
        print(f"  a deliberately adversarial model still scores {r['adversarial_floor_bps']} bps")
        for k, v in sorted(r["substitute_bps"].items()):
            print(f"    {k:16s} {v:5d} bps")
        print(f"  observed models: {min(r['observed_bps'].values())}"
              f" to {max(r['observed_bps'].values())} bps")
        print(f"  the model can move only {r['model_movable_range_bps']} bps of the 10000: "
              f"{r['fraction_of_score_that_is_model_independent'] * 100:.1f}% of the score is the "
              f"scaffold, not the model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
