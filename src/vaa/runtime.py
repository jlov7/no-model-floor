from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .canonical import digest
from .environment import Scenario
from .world_model import WorldModelPrediction


@dataclass(frozen=True)
class ActionContract:
    action_id: str
    target_id: str
    prestate_sha256: str
    prediction_sha256: str
    expected_delta: int
    prohibited_effects: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "expected_delta": self.expected_delta,
            "prediction_sha256": self.prediction_sha256,
            "prestate_sha256": self.prestate_sha256,
            "prohibited_effects": list(self.prohibited_effects),
            "target_id": self.target_id,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class AuthorityEnvelope:
    action_sha256: str
    target_id: str
    prestate_sha256: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_sha256": self.action_sha256,
            "prestate_sha256": self.prestate_sha256,
            "status": self.status,
            "target_id": self.target_id,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True)
class OutcomeReceipt:
    status: str
    reason: str
    initial_revenue: int
    final_revenue: int
    external_effect_count: int
    synthetic_effect_count: int
    action_sha256: str
    authority_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_sha256": self.action_sha256,
            "authority_sha256": self.authority_sha256,
            "external_effect_count": self.external_effect_count,
            "final_revenue": self.final_revenue,
            "initial_revenue": self.initial_revenue,
            "reason": self.reason,
            "status": self.status,
            "synthetic_effect_count": self.synthetic_effect_count,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


class NoEffectRuntime:
    """Executes only against an in-memory synthetic state and always emits zero external effects."""

    def build_contracts(
        self, scenario: Scenario, prediction: WorldModelPrediction
    ) -> tuple[ActionContract, AuthorityEnvelope]:
        action = ActionContract(
            action_id=prediction.action_id,
            target_id=scenario.organization_id,
            prestate_sha256=scenario.initial_state_sha256,
            prediction_sha256=prediction.sha256,
            expected_delta=prediction.predicted_delta,
            prohibited_effects=("external-system-write", "non-revenue-metric-change"),
        )
        authority = AuthorityEnvelope(
            action_sha256=action.sha256,
            target_id=scenario.organization_id,
            prestate_sha256=scenario.initial_state_sha256,
            status=scenario.authority_status,
        )
        return action, authority

    def execute(
        self,
        scenario: Scenario,
        action: ActionContract,
        authority: AuthorityEnvelope,
    ) -> OutcomeReceipt:
        reason = "APPLIED"
        status = "APPLIED"
        final_revenue = scenario.draft_revenue
        synthetic_effect_count = 0
        if action.action_id != "apply-late-adjustment":
            status, reason = "BLOCKED", "ACTION_NOT_PERMITTED"
        elif action.target_id != scenario.organization_id or authority.target_id != action.target_id:
            status, reason = "BLOCKED", "TARGET_BINDING_MISMATCH"
        elif action.prestate_sha256 != scenario.initial_state_sha256:
            status, reason = "BLOCKED", "PRESTATE_BINDING_MISMATCH"
        elif authority.action_sha256 != action.sha256:
            status, reason = "BLOCKED", "ACTION_AUTHORITY_BINDING_MISMATCH"
        elif authority.status != "VALID":
            status, reason = "BLOCKED", "AUTHORITY_NOT_VALID"
        elif scenario.approval_status != "APPROVED":
            status, reason = "BLOCKED", "APPROVAL_NOT_ESTABLISHED"
        elif action.expected_delta != scenario.expected_delta:
            status, reason = "BLOCKED", "PREDICTED_DELTA_MISMATCH"
        else:
            final_revenue = scenario.draft_revenue + action.expected_delta
            synthetic_effect_count = 1
        return OutcomeReceipt(
            status=status,
            reason=reason,
            initial_revenue=scenario.draft_revenue,
            final_revenue=final_revenue,
            external_effect_count=0,
            synthetic_effect_count=synthetic_effect_count,
            action_sha256=action.sha256,
            authority_sha256=authority.sha256,
        )
