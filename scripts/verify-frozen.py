#!/usr/bin/env python3
"""Verify the exact files used by the preregistered confirmatory analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen" / "PREREGISTERED_V1.sha256"
FIXED_FILES = {
    Path("experiments/ablation.py"),
    Path("experiments/probe.py"),
    Path("experiments/sweep.py"),
}


def expected_paths() -> set[Path]:
    return {path.relative_to(ROOT) for path in (ROOT / "src").rglob("*.py")} | FIXED_FILES


def read_manifest() -> dict[Path, str]:
    entries: dict[Path, str] = {}
    for line_number, raw in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, name = raw.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"{MANIFEST.name}:{line_number}: malformed entry")
        path = Path(name)
        if path in entries:
            raise ValueError(f"{MANIFEST.name}:{line_number}: duplicate path {path}")
        entries[path] = digest
    return entries


def main() -> None:
    recorded = read_manifest()
    current = expected_paths()
    if set(recorded) != current:
        missing = sorted(current - set(recorded))
        extra = sorted(set(recorded) - current)
        raise SystemExit(f"frozen manifest path set changed; missing={missing}, extra={extra}")

    failures: list[str] = []
    for relative, expected in sorted(recorded.items(), key=lambda item: str(item[0])):
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"{relative}: expected {expected}, got {actual}")
    if failures:
        raise SystemExit("frozen confirmatory logic changed:\n" + "\n".join(failures))
    print(f"verified {len(recorded)} frozen files against {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
