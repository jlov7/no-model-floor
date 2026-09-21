from __future__ import annotations

import unittest

from vaa.canonical import CanonicalError, canonical_bytes, digest, envelope, verify_envelope


class CanonicalTests(unittest.TestCase):
    def test_canonical_bytes_are_order_independent(self) -> None:
        self.assertEqual(canonical_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}')

    def test_floats_are_rejected(self) -> None:
        with self.assertRaises(CanonicalError):
            canonical_bytes({"value": 1.5})

    def test_envelope_digest_is_verified(self) -> None:
        value = envelope("Example", {"answer": 42})
        self.assertTrue(verify_envelope(value))
        value["payload"]["answer"] = 41
        self.assertFalse(verify_envelope(value))
        self.assertNotEqual(value["sha256"], digest({k: v for k, v in value.items() if k != "sha256"}))


if __name__ == "__main__":
    unittest.main()
