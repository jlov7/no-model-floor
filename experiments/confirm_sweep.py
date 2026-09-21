"""Headroom on the confirmatory surface, using the frozen sweep scoring unchanged.

`sweep.py` reads the exploratory run's cell identifiers, which encode the abstract state in their
names. The confirmatory surface uses its own identifiers, so this rebuilds the same sample records
from `surface_v2` and then calls the frozen `score` function. No scoring logic is redefined here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import surface_v2
from confirm import MANDATE, PERMISSION
from runs import evidence_runs
from sweep import GRID, score


def main() -> int:
    ap = argparse.ArgumentParser(description="Headroom for the confirmatory run")
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    by_cell = {c["cell_id"]: c for c in surface_v2.cells()}
    out = {"protocol_version": "confirm-sweep-1.0", "grid": GRID, "models": {}}

    for run in evidence_runs(args.root):
        meta, raw = run.manifest, run.read("raw-samples.json")
        samples = [
            {
                "decision": r["decision"],
                "confidence_bps": r["confidence_bps"],
                "approval": PERMISSION[by_cell[r["cell_id"]]["permission"]],
                "authority": MANDATE[by_cell[r["cell_id"]]["mandate"]],
            }
            for r in raw
        ]
        curves = {}
        for label, guards_on in (("guards_off", False), ("guards_on", True)):
            points = [{"floor": f, **score(samples, guards_on, f)} for f in GRID]
            best = max(points, key=lambda p: (p["bps"], -p["unsafe"]))
            curves[label] = {
                "best_bps": best["bps"],
                "optimal_floors": [p["floor"] for p in points if p["bps"] == best["bps"]],
                "points": points,
            }
        out["models"][meta["model"]] = curves

    for label in ("guards_off", "guards_on"):
        per_floor = {
            f: sum(
                next(p for p in m[label]["points"] if p["floor"] == f)["bps"]
                for m in out["models"].values()
            ) / len(out["models"])
            for f in GRID
        }
        best_shared = max(per_floor, key=lambda f: per_floor[f])
        mean_best = sum(m[label]["best_bps"] for m in out["models"].values()) / len(out["models"])
        out.setdefault("shared_floor", {})[label] = {
            "best_single_floor": best_shared,
            "mean_bps_at_best_single_floor": round(per_floor[best_shared]),
            "mean_bps_if_each_model_uses_its_own_best": round(mean_best),
            "adaptation_headroom_bps": round(mean_best - per_floor[best_shared]),
        }

    Path(args.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for label in ("guards_on", "guards_off"):
        s = out["shared_floor"][label]
        print(f"{label}: shared best floor {s['best_single_floor']} -> "
              f"{s['mean_bps_at_best_single_floor']} mean bps;  per-model best -> "
              f"{s['mean_bps_if_each_model_uses_its_own_best']};  "
              f"headroom {s['adaptation_headroom_bps']} bps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
