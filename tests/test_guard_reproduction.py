"""The resumable guard-reproduction cache cannot silently mix harnesses.

`run_resumable` skips model calls by index. Before prompt digests existed, changing the prompt
construction and resuming into an old raw file scored rows for a harness they never saw -- which is
not a hypothetical here, since this project itself changed its EICU-AC harness between commits.
These tests pin the digest binding: matching caches are trusted, missing or mismatched digests are
re-queried, and a transport failure during re-verification aborts without writing rather than
half-updating the file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    spec = importlib.util.spec_from_file_location(
        "reproduce_guard_test", Path("experiments/reproduce_guard.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["reproduce_guard_test"] = module
    spec.loader.exec_module(module)
    return module


def make_items(n):
    return [
        {
            "label": i % 2,
            "confirmed_task": f"task number {i}",
            "user_info": {
                "age": 30 + i,
                "dr_license": True,
                "vaccine": True,
                "domestic": True,
                "membership": True,
            },
        }
        for i in range(n)
    ]


class CacheProvenance(unittest.TestCase):
    def setUp(self):
        self.rg = load_module()
        self.items = make_items(3)

    def run_resumable(self, tmp, classify):
        original = self.rg.classify
        self.rg.classify = classify
        try:
            return self.rg.run_resumable(
                self.items,
                self.rg.mind2web_prompt,
                "http://unused",
                "unused",
                1,
                Path(tmp) / "raw.json",
            )
        finally:
            self.rg.classify = original

    def test_a_matching_cache_is_trusted_and_makes_no_calls(self):
        calls = []

        def classify(base_url, model, content, timeout):
            calls.append(content)
            return {"raw": "safe", "predicted": 0}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            records = [
                {
                    "index": i,
                    "label": self.items[i]["label"],
                    "raw": "safe",
                    "predicted": 0,
                    "prompt_sha256": self.rg.prompt_hash(self.rg.mind2web_prompt(self.items[i])),
                }
                for i in range(len(self.items))
            ]
            path.write_text(json.dumps(records), encoding="utf-8")
            _, summary = self.run_resumable(tmp, classify)
            self.assertEqual(calls, [])
            self.assertEqual(summary["new_calls_this_run"], 0)
            self.assertEqual(summary["cache_records_reverified"], 0)

    def test_a_missing_or_stale_digest_is_requeried_and_updated(self):
        calls = []

        def classify(base_url, model, content, timeout):
            calls.append(content)
            return {"raw": "unsafe", "predicted": 1}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            # Index 0 predates digests; index 1 was produced by some other harness wording.
            records = [
                {"index": 0, "label": 0, "raw": "safe", "predicted": 0},
                {
                    "index": 1,
                    "label": 1,
                    "raw": "safe",
                    "predicted": 0,
                    "prompt_sha256": "0" * 16,
                },
            ]
            path.write_text(json.dumps(records), encoding="utf-8")
            records_out, summary = self.run_resumable(tmp, classify)
            self.assertEqual(len(calls), 3, "indices 0 and 1 re-queried; index 2 is fresh")
            by_index = {r["index"]: r for r in records_out}
            self.assertEqual(by_index[0]["prompt_sha256"], self.rg.prompt_hash(
                self.rg.mind2web_prompt(self.items[0])))
            self.assertEqual(by_index[1]["raw"], "unsafe")
            self.assertEqual(summary["cache_records_reverified"], 2)

    def test_a_failed_reverification_aborts_without_writing(self):
        def classify(base_url, model, content, timeout):
            return {"error": "ConnectionError: down", "raw": None, "predicted": None}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.json"
            before = json.dumps([
                {"index": 0, "label": 0, "raw": "safe", "predicted": 0},
            ])
            path.write_text(before, encoding="utf-8")
            with self.assertRaises(RuntimeError) as caught:
                self.run_resumable(tmp, classify)
            self.assertIn("digest", str(caught.exception))
            self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_fresh_transport_errors_are_still_recorded_not_fatal(self):
        """Only re-verification failures abort; a fresh run keeps recording errors as data."""

        def classify(base_url, model, content, timeout):
            return {"error": "ConnectionError: down", "raw": None, "predicted": None}

        with tempfile.TemporaryDirectory() as tmp:
            records, summary = self.run_resumable(tmp, classify)
            self.assertEqual(len(records), 3)
            self.assertEqual(summary["transport_errors"], 3)


if __name__ == "__main__":
    unittest.main()
