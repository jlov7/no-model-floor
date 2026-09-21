#!/usr/bin/env bash
# Reproduce analyses that require third-party corpora. Missing corpora are a failed gate.
set -euo pipefail

for candidate in "${PYTHON:-}" python3.13 python3 python; do
  [ -n "$candidate" ] || continue
  if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)' \
     > /dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "${PYTHON:-}" ]; then
  echo "No Python 3.13 or newer found. Set PYTHON to one." >&2
  exit 1
fi

if [ ! -d external/survey ] || [ ! -d external/m2w ] || [ ! -d external/eicu ]; then
  echo "Third-party corpora are missing; external reproduction did not run." >&2
  echo "Provide authorized local copies under external/ as described in THIRD_PARTY_NOTICES.md." >&2
  exit 1
fi

export PYTHONPATH="src:experiments"
VERIFY_TMP="$(mktemp -d "${TMPDIR:-/tmp}/no-model-floor-external.XXXXXXXX")"
trap 'rm -rf "$VERIFY_TMP"' EXIT

"$PYTHON" experiments/external_floor.py \
  --mind2web external/m2w/seeact/sample_labeled_all.json \
  --eicu external/eicu/ehragent/eicu_ac.json \
  --output "$VERIFY_TMP/external-floor.json" > /dev/null
"$PYTHON" experiments/floor_survey.py --root external/survey \
  --output "$VERIFY_TMP/floor-survey.json" > /dev/null
"$PYTHON" experiments/external_headroom.py --root external \
  --output "$VERIFY_TMP/external-headroom.json" > /dev/null
"$PYTHON" experiments/rule_space_headroom.py \
  --mind2web external/m2w/seeact/sample_labeled_all.json \
  --output "$VERIFY_TMP/rule-space-headroom.json" > /dev/null

"$PYTHON" - "$VERIFY_TMP" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for committed, fresh in (
    ("evidence/external-floor.json", root / "external-floor.json"),
    ("evidence/floor-survey.json", root / "floor-survey.json"),
    ("evidence/external-headroom.json", root / "external-headroom.json"),
    ("evidence/rule-space-headroom.json", root / "rule-space-headroom.json"),
):
    with open(committed, encoding="utf-8") as expected, fresh.open(encoding="utf-8") as actual:
        if json.load(expected) != json.load(actual):
            raise SystemExit(f"{committed} drifted from its source corpus")
print("external analyses reproduce from the fetched corpora")
PY
