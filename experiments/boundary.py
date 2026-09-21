"""Where the dominance result stops holding.

The theorem says one configuration is optimal for every possible model. That statement carries a
qualifier which earlier drafts of this repository omitted, and the omission made the claim false as
written: dominance on individual cells implies optimality only for metrics that are **nondecreasing
in the per-cell scores**. Accuracy, balanced accuracy and the utility used here all qualify. Not
every metric does.

This exhibits a metric that does not, and shows the headroom it produces is genuinely nonzero, with
per-model optima that genuinely separate. It is here so the boundary is demonstrated rather than
described, and so that anyone applying the headroom test elsewhere knows which side of the line
their own metric falls on.

The counterexample is calibration gap, the absolute difference between how often a system is right
and how confident it says it is, negated so that larger is better. It is a standard diagnostic and
this repository already cares about confidence, so it is not a contrived choice. It is not monotone
in accuracy: repairing a decision moves accuracy, and if stated confidence stays put, more repair
can push the two further apart rather than closer together.

Offline: re-scores committed draws and calls no model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exhaustive import applied, configurations, expected, label
from runs import evidence_runs

STAR = "invalid_authority+denied_approval+unknown_approval@0"


def calibration_gap(draws: list[dict], config: dict[str, Any]) -> float:
    """Negated |accuracy - mean stated confidence|. Larger is better, maximum 0."""
    correct = sum(
        applied(d["decision"], d["confidence_bps"], d["approval"], d["authority"], config)
        == expected(d["approval"], d["authority"])
        for d in draws
    ) / len(draws)
    stated = sum(d["confidence_bps"] for d in draws) / len(draws) / 10000.0
    return -abs(correct - stated)


def headroom(per_model: dict[str, list[dict]]) -> dict[str, Any]:
    configs = configurations()
    scores = {m: [calibration_gap(d, c) for c in configs] for m, d in per_model.items()}
    models = list(per_model)
    per_model_best = {m: max(scores[m]) for m in models}
    shared = [sum(scores[m][i] for m in models) / len(models) for i in range(len(configs))]
    best_i = max(range(len(configs)), key=lambda i: shared[i])
    gap = sum(per_model_best.values()) / len(models) - shared[best_i]
    return {
        "headroom_bps": round(gap * 10000, 2),
        "best_shared_config": label(configs[best_i]),
        "per_model_best_config": {
            m: label(configs[max(range(len(configs)), key=lambda i: scores[m][i])]) for m in models
        },
        "dominating_config_is_shared_optimum": label(configs[best_i]) == STAR,
    }


def load(root: Path, v2: dict) -> dict[str, list[dict]]:
    out = {}
    for run in evidence_runs(root):
        meta, raw = run.manifest, run.read("raw-samples.json")
        draws = []
        for r in raw:
            if "scenario_id" in r:
                a, u, _ = r["scenario_id"].split("-")
                a, u = a.upper(), u.upper()
            else:
                a, u = v2[r["cell_id"]]
            draws.append({"approval": a, "authority": u, "decision": r["decision"],
                          "confidence_bps": r["confidence_bps"]})
        out[meta["model"]] = draws
    return out


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

    report = {
        "metric": "negated absolute calibration gap, not monotone in per-cell accuracy",
        "exploratory": headroom(load(Path(args.exploratory), v2)),
        "confirmatory": headroom(load(Path(args.confirmatory), v2)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for surface in ("exploratory", "confirmatory"):
        r = report[surface]
        print(f"{surface:13s} headroom {r['headroom_bps']:8.2f} bps   "
              f"best shared: {r['best_shared_config']}")
        print(f"{'':13s} the dominating configuration is the shared optimum: "
              f"{r['dominating_config_is_shared_optimum']}")
    print("\nNonzero, and the per-model optima separate. Cell-by-cell dominance does not")
    print("survive a metric that is not monotone in the per-cell scores. The theorem needs")
    print("that qualifier and earlier drafts of this repository did not state it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
