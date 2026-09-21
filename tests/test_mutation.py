from __future__ import annotations

import unittest

from vaa.agent import AgentRunner
from vaa.environment import Scenario
from vaa.mutation import propose_mutation
from vaa.providers import ScriptedProvider
from vaa.scaffold import ScaffoldSpec


class MutationTests(unittest.TestCase):
    def test_failed_unknown_approval_trajectory_produces_executable_single_surface_patch(self) -> None:
        scenario = Scenario("unknown", "redwood", 800, 820, "UNKNOWN", "VALID", "HIGH")
        provider = ScriptedProvider({
            "unknown": {
                "decision": "ACT",
                "action_id": "apply-late-adjustment",
                "confidence_bps": 9000,
                "question": None,
                "assumptions": ["approval probably exists"],
            }
        })
        baseline = ScaffoldSpec.baseline()
        trajectory = AgentRunner().run(scenario, baseline, provider)
        proposal = propose_mutation(baseline, [trajectory])
        self.assertIsNotNone(proposal)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("ask_on_unknown_approval",))
        self.assertTrue(proposal.candidate.ask_on_unknown_approval)
        self.assertEqual(proposal.candidate_sha256, proposal.candidate.sha256)
        self.assertEqual(proposal.evidence_trajectory_sha256s, (trajectory.sha256,))


if __name__ == "__main__":
    unittest.main()
