import unittest

from vaa.scaffold import ScaffoldError, ScaffoldSpec


class ScaffoldTests(unittest.TestCase):
    def test_candidate_hash_binds_executable_spec(self):
        baseline = ScaffoldSpec.baseline()
        candidate = baseline.with_patch({"ask_on_unknown_approval": True})
        self.assertNotEqual(candidate.sha256, baseline.sha256)
        self.assertTrue(candidate.to_dict()["ask_on_unknown_approval"])

    def test_only_declared_mutation_surface_can_change(self):
        with self.assertRaises(ScaffoldError):
            ScaffoldSpec.baseline().with_patch({"scaffold_id": "forged"})

if __name__ == "__main__":
    unittest.main()
