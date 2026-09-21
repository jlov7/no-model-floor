"""Uncertainty at the right unit, and a paired test that says what the claim actually is.

The earlier reporting put Wilson intervals over sampled draws while calling the scenario cell the
unit of analysis. Those are different estimands. A Wilson interval over 180 draws treats each draw
as independent, but ten draws come from the same cell and are correlated, so that interval is too
narrow for any statement about cells.

Two corrections here:

1. A cell-clustered bootstrap. Cells are resampled with replacement, carrying all their draws, which
   respects the correlation and widens the interval to something honest for "this scenario set".
2. A paired comparison between the derived policy and the best fixed policy, computed per draw on
   identical model behaviour. This is the claim that actually matters, and it is an exact equality
   rather than two intervals that happen to overlap.

Both are offline re-analyses of committed draws. No model is called.

`ablation.py` still emits a Wilson interval per policy. That file is frozen at tag
the frozen checksum manifest and cannot be annotated without breaking the freeze, so the note goes here: its
Wilson figures are retained in the raw artefacts for traceability and are not the reported
uncertainty. The cell-clustered intervals below are.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from exhaustive import GUARDS, applied, expected, score
from runs import evidence_runs

BOOTSTRAP = 10_000
SEED = 20260819


def cell_key(s: dict) -> tuple[str, str]:
    return (s["approval"], s["authority"])


def accuracy(samples: list[dict], config: dict[str, Any]) -> float:
    return score(samples, config)["accuracy"]


def clustered_bootstrap(samples: list[dict], config: dict[str, Any],
                        rounds: int = BOOTSTRAP) -> tuple[int, int]:
    """Resample whole cells, not individual draws, because draws within a cell are correlated."""
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for s in samples:
        by_cell.setdefault(cell_key(s), []).append(s)
    cells = list(by_cell.values())
    rng = random.Random(SEED)
    stats = []
    for _ in range(rounds):
        drawn = [rng.choice(cells) for _ in range(len(cells))]
        flat = [s for cell in drawn for s in cell]
        stats.append(accuracy(flat, config))
    stats.sort()
    lo = stats[int(0.025 * rounds)]
    hi = stats[int(0.975 * rounds) - 1]
    return round(lo * 10000), round(hi * 10000)


def derived_config(root: Path, model: str) -> dict[str, Any]:
    """The guards this model's run actually derived, read from its own manifest.

    The call site previously passed the all-guards config as both arguments, comparing a policy
    with itself, so the reported count of disagreements was zero for any data whatsoever. The
    derived policy genuinely differs per model: one derived two of the three guards and another a
    different two. The comparison is only worth reporting if it could have come out otherwise.
    """
    for run in evidence_runs(root):
        if run.manifest["model"] != model:
            continue
        guards = run.manifest["results"]["derived"]["guards"]
        config = {name: (name in guards) for name in GUARDS}
        config["floor"] = 0
        return config
    raise KeyError(f"no run for {model} under {root}")


def paired_disagreement(samples: list[dict], a: dict[str, Any], b: dict[str, Any]) -> dict[str, int]:
    """How often two policies differ on the same draw, and who wins when they do."""
    a_only = b_only = both = neither = differ = 0
    for s in samples:
        want = expected(s["approval"], s["authority"])
        ga = applied(s["decision"], s["confidence_bps"], s["approval"], s["authority"], a)
        gb = applied(s["decision"], s["confidence_bps"], s["approval"], s["authority"], b)
        if ga != gb:
            differ += 1
        oa, ob = ga == want, gb == want
        both += oa and ob
        neither += (not oa) and (not ob)
        a_only += oa and not ob
        b_only += ob and not oa
    return {"draws": len(samples), "decisions_differ": differ,
            "both_correct": both, "neither_correct": neither,
            "derived_only_correct": a_only, "best_fixed_only_correct": b_only}


def main() -> int:
    ap = argparse.ArgumentParser(description="Cell-clustered intervals and paired comparison")
    ap.add_argument("--exhaustive", required=True, help="output of experiments/exhaustive.py")
    ap.add_argument("--exploratory", required=True)
    ap.add_argument("--confirmatory", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import surface_v2
    from confirm import MANDATE, PERMISSION
    from exhaustive import load
    by_cell = {c["cell_id"]: c for c in surface_v2.cells()}

    def explore_cell(r):
        approval, authority, _risk = r["scenario_id"].split("-")
        return {"decision": r["decision"], "confidence_bps": r["confidence_bps"],
                "approval": approval.upper(), "authority": authority.upper()}

    def confirm_cell(r):
        c = by_cell[r["cell_id"]]
        return {"decision": r["decision"], "confidence_bps": r["confidence_bps"],
                "approval": PERMISSION[c["permission"]], "authority": MANDATE[c["mandate"]]}

    all_guards = dict.fromkeys(GUARDS, True) | {"floor": 0}
    out = {
        "protocol_version": "uncertainty-1.0",
        "bootstrap_rounds": BOOTSTRAP,
        "estimand": (
            "Accuracy of a fixed policy on this 18-cell scenario set, under the model's own "
            "sampling distribution at temperature 0.7. Cells are the resampling unit, so the "
            "interval covers uncertainty from which cells happen to be in the set and from model "
            "stochasticity. It does not license generalisation to other tasks, scenario "
            "distributions or models."
        ),
        "surfaces": {},
    }

    for name, root, cellmap in (("exploratory", Path(args.exploratory), explore_cell),
                                ("confirmatory", Path(args.confirmatory), confirm_cell)):
        per_model = load(root, cellmap)
        block = {}
        for model, samples in per_model.items():
            lo, hi = clustered_bootstrap(samples, all_guards)
            block[model] = {
                "accuracy_bps": round(accuracy(samples, all_guards) * 10000),
                "cell_clustered_95ci_bps": [lo, hi],
                "paired_vs_best_fixed": paired_disagreement(
                    samples, derived_config(root, model), all_guards
                ),
            }
        out["surfaces"][name] = block

    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for surface, block in out["surfaces"].items():
        print(f"{surface}:")
        for model, r in block.items():
            lo, hi = r["cell_clustered_95ci_bps"]
            p = r["paired_vs_best_fixed"]
            print(f"  {model:<24} {r['accuracy_bps']:>5} bps   cell-clustered 95% CI "
                  f"{lo}-{hi}   derived vs best fixed: {p['decisions_differ']} of "
                  f"{p['draws']} draws differ")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
