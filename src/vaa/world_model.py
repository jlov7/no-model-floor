from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import digest
from .environment import Scenario


@dataclass(frozen=True)
class WorldModelPrediction:
    scenario_id: str
    organization_id: str
    action_id: str
    prestate_sha256: str
    predicted_delta: int
    assumptions: tuple[str, ...]
    unsupported_consequences: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "assumptions": list(self.assumptions),
            "organization_id": self.organization_id,
            "predicted_delta": self.predicted_delta,
            "prestate_sha256": self.prestate_sha256,
            "scenario_id": self.scenario_id,
            "unsupported_consequences": list(self.unsupported_consequences),
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


class SymbolicWorldModel:
    """Deterministic reference world model; a learned model can replace this interface later."""

    def predict(self, scenario: Scenario, action_id: str) -> WorldModelPrediction:
        if action_id != "apply-late-adjustment":
            raise ValueError("unsupported candidate action")
        assumptions = (
            f"approval_status={scenario.approval_status}",
            f"authority_status={scenario.authority_status}",
        )
        unsupported = (
            "human approver response",
            "effects outside the synthetic reporting state",
        )
        return WorldModelPrediction(
            scenario_id=scenario.scenario_id,
            organization_id=scenario.organization_id,
            action_id=action_id,
            prestate_sha256=scenario.initial_state_sha256,
            predicted_delta=scenario.expected_delta,
            assumptions=assumptions,
            unsupported_consequences=unsupported,
        )
