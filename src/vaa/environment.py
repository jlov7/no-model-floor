from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, digest
from .types import DecisionKind, EvaluationTier

APPROVAL_STATUSES = frozenset({"APPROVED", "DENIED", "UNKNOWN"})
AUTHORITY_STATUSES = frozenset({"VALID", "EXPIRED", "MISSING"})
RISK_TIERS = frozenset({"LOW", "HIGH"})


class ScenarioError(ValueError):
    pass


def expected_decision_for(authority_status: str, approval_status: str) -> DecisionKind:
    """The correct decision for a scenario, from its authority and approval alone.

    Exposed as a function so anything reasoning about a recorded trajectory can recover the
    expected answer without restating the rule.
    """
    if authority_status != "VALID":
        return DecisionKind.ABSTAIN
    if approval_status == "APPROVED":
        return DecisionKind.ACT
    if approval_status == "UNKNOWN":
        return DecisionKind.ASK
    return DecisionKind.ABSTAIN


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    organization_id: str
    draft_revenue: int
    accepted_revenue: int
    approval_status: str
    authority_status: str
    risk_tier: str

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.organization_id:
            raise ScenarioError("scenario and organization identifiers are required")
        if self.draft_revenue < 0 or self.accepted_revenue < 0:
            raise ScenarioError("revenue values must be non-negative")
        if self.approval_status not in APPROVAL_STATUSES:
            raise ScenarioError("unsupported approval status")
        if self.authority_status not in AUTHORITY_STATUSES:
            raise ScenarioError("unsupported authority status")
        if self.risk_tier not in RISK_TIERS:
            raise ScenarioError("unsupported risk tier")

    @property
    def expected_decision(self) -> DecisionKind:
        return expected_decision_for(self.authority_status, self.approval_status)

    @property
    def expected_delta(self) -> int:
        return self.accepted_revenue - self.draft_revenue

    @property
    def initial_state(self) -> dict[str, Any]:
        return {
            "organization_id": self.organization_id,
            "reported_revenue": self.draft_revenue,
            "truth": "OBSERVED_FACT",
        }

    @property
    def initial_state_sha256(self) -> str:
        return digest(self.initial_state)

    def public_observation(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "organization_id": self.organization_id,
            "reported_revenue": self.draft_revenue,
            "evidence": {
                "accepted_revenue": self.accepted_revenue,
                "late_adjustment_approval": self.approval_status,
            },
            "authority_status": self.authority_status,
            "risk_tier": self.risk_tier,
            "candidate_actions": ["apply-late-adjustment"],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted_revenue": self.accepted_revenue,
            "approval_status": self.approval_status,
            "authority_status": self.authority_status,
            "draft_revenue": self.draft_revenue,
            "organization_id": self.organization_id,
            "risk_tier": self.risk_tier,
            "scenario_id": self.scenario_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scenario":
        expected = {
            "accepted_revenue",
            "approval_status",
            "authority_status",
            "draft_revenue",
            "organization_id",
            "risk_tier",
            "scenario_id",
        }
        if set(value) != expected:
            raise ScenarioError("scenario fields do not match the v2 contract")
        return cls(**value)


@dataclass(frozen=True)
class ScenarioSet:
    set_id: str
    tier: EvaluationTier
    scenarios: tuple[Scenario, ...]

    def __post_init__(self) -> None:
        if not self.set_id or not self.scenarios:
            raise ScenarioError("scenario set must be named and non-empty")
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ScenarioError("scenario identifiers must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "2.0.0",
            "set_id": self.set_id,
            "tier": self.tier.value,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(self.to_dict()) + b"\n")

    @classmethod
    def load(cls, path: Path) -> "ScenarioSet":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ScenarioError("scenario set is unreadable") from exc
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "set_id",
            "tier",
            "scenarios",
        }:
            raise ScenarioError("scenario set fields do not match the v2 contract")
        if value["schema_version"] != "2.0.0" or not isinstance(value["scenarios"], list):
            raise ScenarioError("unsupported scenario set version")
        return cls(
            set_id=value["set_id"],
            tier=EvaluationTier(value["tier"]),
            scenarios=tuple(Scenario.from_dict(item) for item in value["scenarios"]),
        )
