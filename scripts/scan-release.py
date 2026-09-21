#!/usr/bin/env python3
"""Fail on credential-like, identifying, binary, or unsafe release files.

Findings print paths and rule names only. File contents are never echoed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

IGNORED_PARTS = {
    ".git",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "external",
    "out",
}
IGNORED_SUFFIXES = {".egg-info"}
RULES = {
    "absolute-home-path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9._-]+/"),
    "private-key-header": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "openai-token": re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "github-token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws-access-key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}


def ignored(path: Path) -> bool:
    return any(part in IGNORED_PARTS or any(part.endswith(s) for s in IGNORED_SUFFIXES) for part in path.parts)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    findings: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ignored(relative) or not path.is_file():
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            findings.append((str(relative), "credential-like-filename"))
            continue
        data = path.read_bytes()
        if b"\x00" in data:
            findings.append((str(relative), "unexpected-binary"))
            continue
        if any(byte < 9 or 13 < byte < 32 for byte in data):
            findings.append((str(relative), "control-character"))
        for name, pattern in RULES.items():
            if pattern.search(data):
                findings.append((str(relative), name))
    if findings:
        for path, rule in findings:
            print(f"{path}: {rule}", file=sys.stderr)
        return 1
    print(f"release scan passed for {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
