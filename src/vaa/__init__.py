"""A small executable scaffold that sits between a language model and its actions.

The model proposes a decision; the scaffold decides what is allowed to execute. Guards only ever
make the agent more conservative, and the deriver selects among them from observed failures.
"""

from .agent import AgentRunner, TrajectoryRecord, apply_scaffold_rules
from .environment import Scenario, ScenarioSet, expected_decision_for
from .mutation import MutationProposal, propose_mutation
from .providers import ModelDecision, OpenAICompatibleProvider, ProviderError
from .scaffold import ScaffoldSpec

__all__ = [
    "AgentRunner",
    "MutationProposal",
    "ModelDecision",
    "OpenAICompatibleProvider",
    "ProviderError",
    "ScaffoldSpec",
    "Scenario",
    "ScenarioSet",
    "TrajectoryRecord",
    "apply_scaffold_rules",
    "expected_decision_for",
    "propose_mutation",
]
