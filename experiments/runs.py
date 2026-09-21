"""The one way to enumerate evidence runs, so that a partial run cannot be counted as a whole one.

Twice in this project an analysis enumerated runs by globbing for files and silently included one
that had aborted. The first time it shifted a published figure by half a point in the factorial
permission table. The second time, three hours later and in freshly written code, an aborted sweep
contributed six vocabularies of nothing and moved another figure by 167 basis points. Both were
caught, the second only because the first was already written down and someone went looking.

Catching a class of error by remembering to look for it is not a property that survives. This is the
structural version: every analysis enumerates runs through `completed`, which requires a manifest
plus every artefact the analysis intends to read, and returns what it rejected alongside what it
accepted so the rejection can be reported rather than inferred from an absence.

The rule it enforces is deliberately narrow: a run counts if, and only if, the files it promised are
all present. Whether their contents are usable is the caller's business, and callers that need more
than presence say so through `require`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


class Run:
    """One evidence directory that passed every check."""

    def __init__(self, directory: Path, manifest: dict[str, Any]) -> None:
        self.directory = directory
        self.manifest = manifest

    @property
    def name(self) -> str:
        return self.directory.name

    def read(self, filename: str) -> Any:
        return json.loads((self.directory / filename).read_text(encoding="utf-8"))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Run({self.name})"


def completed(
    root: Path | str,
    manifest_name: str | Iterable[str],
    *,
    artefacts: Iterable[str] = (),
    require: Callable[[Run], str | None] | None = None,
) -> tuple[list[Run], list[dict[str, str]]]:
    """Return (accepted runs, rejections) under `root`.

    A directory is accepted when it holds `manifest_name` and every file in `artefacts`, and when
    `require` returns None for it. `require` exists for checks that presence cannot express, such as
    "every cell has draws"; it returns a reason string to reject, or None to accept.

    Rejections are returned rather than logged, so a caller has to decide what to do with them. An
    analysis that drops a run without saying so is the thing this module exists to prevent.
    """
    root = Path(root)
    names = [manifest_name] if isinstance(manifest_name, str) else list(manifest_name)
    accepted: list[Run] = []
    rejected: list[dict[str, str]] = []

    for directory in sorted(p for p in root.glob("*") if p.is_dir()):
        manifest_path = next((directory / n for n in names if (directory / n).exists()), None)
        if manifest_path is None:
            rejected.append(
                {
                    "run": directory.name,
                    "reason": f"no {' or '.join(names)}: the run did not finish",
                }
            )
            continue

        missing = [f for f in artefacts if not (directory / f).exists()]
        if missing:
            rejected.append(
                {
                    "run": directory.name,
                    "reason": f"missing artefacts: {', '.join(sorted(missing))}",
                }
            )
            continue

        run = Run(directory, json.loads(manifest_path.read_text(encoding="utf-8")))
        if require is not None:
            complaint = require(run)
            if complaint:
                rejected.append({"run": directory.name, "reason": complaint})
                continue

        accepted.append(run)

    return accepted, rejected


def _draws_match_the_manifest(run: Run) -> str | None:
    """The recorded draws must be exactly what the run says it produced.

    Presence-checking alone does not catch a truncated evidence file: an earlier version of this
    module verified only that `raw-samples.json` existed, and half of one model's draws could be
    deleted without any test or CI step noticing. A run is therefore required to account for every
    draw it attempted. One model here legitimately holds 179 rather than 180 because a single
    response came back off-contract; that is permitted precisely because the manifest declares it,
    and silent loss is not.
    """
    draws = run.read("raw-samples.json")
    declared = run.manifest.get("draws_scored")
    if declared is None:
        return "manifest does not declare draws_scored"
    if len(draws) != declared:
        return f"holds {len(draws)} draws but declares {declared}"

    cells = run.manifest.get("distinct_cells")
    off_contract = run.manifest.get("off_contract_responses", 0)
    key = "scenario_id" if "scenario_id" in draws[0] else "cell_id"
    seen = {d[key] for d in draws}
    if cells is not None and len(seen) != cells:
        return f"holds {len(seen)} distinct cells but declares {cells}"

    attempted = declared + off_contract
    if cells and attempted % cells:
        return f"{attempted} attempted draws is not a whole number per cell across {cells} cells"
    return None


def evidence_runs(root: Path | str) -> list[Run]:
    """The single-model evidence directories under an ablation or confirmatory root.

    Every such run writes one manifest plus `raw-samples.json`. Analyses that read those draws use
    this rather than globbing, so a directory holding a manifest but no draws, or draws but no
    manifest, is skipped by construction instead of by each caller remembering to filter. The draws
    are also checked against what the manifest declares, because a file that exists is not evidence
    that it is intact.
    """
    accepted, rejected = completed(
        root,
        ("ablation.json", "confirm.json"),
        artefacts=["raw-samples.json"],
        require=_draws_match_the_manifest,
    )
    if rejected:
        names = ", ".join(f"{r['run']} ({r['reason']})" for r in rejected)
        raise ValueError(
            f"incomplete evidence run(s) under {root}: {names}. "
            "Analyses must not silently average over a partial run; remove it or complete it."
        )
    return accepted
