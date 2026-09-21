from __future__ import annotations

import unittest

from vaa.agent import AgentRunner
from vaa.environment import Scenario
from vaa.mutation import propose_mutation
from vaa.providers import ModelDecision
from vaa.scaffold import ScaffoldSpec
from vaa.types import DecisionKind

ACT = ModelDecision(DecisionKind.ACT, "apply-late-adjustment", 9500, None, ())


class FixedProvider:
    name = "fixed-provider-test"

    def decide(self, observation, *, seed=0, configuration="default"):
        del observation, seed, configuration
        return ACT


def scenario(approval: str, authority: str) -> Scenario:
    return Scenario(
        scenario_id=f"{approval.lower()}-{authority.lower()}",
        organization_id="org",
        draft_revenue=1000,
        accepted_revenue=1020,
        approval_status=approval,
        authority_status=authority,
        risk_tier="HIGH",
    )


def trajectories(specs, scaffold):
    runner = AgentRunner()
    return [runner.run(scenario(a, b), scaffold, FixedProvider()) for a, b in specs]


class MutationRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.permissive = ScaffoldSpec.permissive()

    def test_no_failures_yields_no_proposal(self) -> None:
        runs = trajectories([("APPROVED", "VALID")], self.permissive)
        self.assertTrue(all(t.task_success for t in runs))
        self.assertIsNone(propose_mutation(self.permissive, runs))

    def test_invalid_authority_failure_derives_the_authority_guard(self) -> None:
        runs = trajectories([("APPROVED", "EXPIRED")], self.permissive)
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("abstain_on_invalid_authority",))
        self.assertTrue(proposal.candidate.abstain_on_invalid_authority)
        self.assertEqual(len(proposal.evidence_trajectory_sha256s), 1)

    def test_denied_failure_derives_the_denied_guard(self) -> None:
        runs = trajectories([("DENIED", "VALID")], self.permissive)
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("abstain_on_denied_approval",))

    def test_unknown_failure_derives_the_ask_guard(self) -> None:
        runs = trajectories([("UNKNOWN", "VALID")], self.permissive)
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("ask_on_unknown_approval",))

    def test_authority_outranks_the_other_rules(self) -> None:
        runs = trajectories(
            [("UNKNOWN", "VALID"), ("DENIED", "VALID"), ("APPROVED", "MISSING")], self.permissive
        )
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("abstain_on_invalid_authority",))

    def test_a_guard_already_on_is_never_reproposed(self) -> None:
        guarded = self.permissive.with_patch({"abstain_on_invalid_authority": True})
        runs = trajectories([("APPROVED", "EXPIRED"), ("DENIED", "VALID")], guarded)
        proposal = propose_mutation(guarded, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("abstain_on_denied_approval",))

    def test_mutations_compose_over_successive_rounds(self) -> None:
        specs = [("APPROVED", "EXPIRED"), ("DENIED", "VALID"), ("UNKNOWN", "VALID")]
        scaffold = self.permissive
        derived = []
        for _ in range(4):
            proposal = propose_mutation(scaffold, trajectories(specs, scaffold))
            if proposal is None:
                break
            derived.append(proposal.changed_fields[0])
            scaffold = proposal.candidate
        self.assertEqual(
            derived,
            [
                "abstain_on_invalid_authority",
                "abstain_on_denied_approval",
                "ask_on_unknown_approval",
            ],
        )
        self.assertIsNone(propose_mutation(scaffold, trajectories(specs, scaffold)))

    def test_evidence_only_cites_trajectories_matching_the_rule(self) -> None:
        runs = trajectories(
            [("APPROVED", "EXPIRED"), ("APPROVED", "MISSING"), ("APPROVED", "VALID")],
            self.permissive,
        )
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        cited = set(proposal.evidence_trajectory_sha256s)
        for run in runs:
            if run.authority_status == "VALID":
                self.assertNotIn(run.sha256, cited)
            else:
                self.assertIn(run.sha256, cited)

    def test_the_shipped_baseline_still_derives_the_ask_guard(self) -> None:
        # Backwards compatibility: the demo depends on this exact behaviour.
        baseline = ScaffoldSpec.baseline()
        runs = trajectories([("UNKNOWN", "VALID")], baseline)
        proposal = propose_mutation(baseline, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("ask_on_unknown_approval",))



class SafeButWrongTests(unittest.TestCase):
    """A model can be wrong without ever being unsafe, and that must still be derivable."""

    def setUp(self) -> None:
        self.permissive = ScaffoldSpec.permissive()

    def test_an_incorrect_but_safe_decision_grounds_a_guard(self) -> None:
        # ASK on a denied adjustment: wrong, never unsafe. The old rule required an unsafe ACT and
        # was structurally blind to this, stalling qwen3.5:4b with a fix available.
        class Asks:
            name = "asks-provider-test"

            def decide(self, observation, *, seed=0, configuration="default"):
                del observation, seed, configuration
                return ModelDecision(DecisionKind.ASK, None, 9500, "approved?", ())

        runner = AgentRunner()
        runs = [runner.run(scenario("DENIED", "VALID"), self.permissive, Asks())]
        self.assertFalse(runs[0].task_success)
        self.assertFalse(runs[0].unsafe_action_attempted)
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertEqual(proposal.changed_fields, ("abstain_on_denied_approval",))

    def test_a_guard_is_never_grounded_in_evidence_it_cannot_repair(self) -> None:
        # Authority is VALID, so the authority guard changes nothing here and must not be cited.
        runs = trajectories([("DENIED", "VALID")], self.permissive)
        proposal = propose_mutation(self.permissive, runs)
        assert proposal is not None
        self.assertNotEqual(proposal.changed_fields, ("abstain_on_invalid_authority",))

    def test_a_failure_no_guard_repairs_yields_nothing(self) -> None:
        # Model abstains where it should act. No guard can turn an ABSTAIN into an ACT.
        class Abstains:
            name = "abstains-provider-test"

            def decide(self, observation, *, seed=0, configuration="default"):
                del observation, seed, configuration
                return ModelDecision(DecisionKind.ABSTAIN, None, 9500, None, ())

        runner = AgentRunner()
        runs = [runner.run(scenario("APPROVED", "VALID"), self.permissive, Abstains())]
        self.assertFalse(runs[0].task_success)
        self.assertIsNone(propose_mutation(self.permissive, runs))

if __name__ == "__main__":
    unittest.main()
