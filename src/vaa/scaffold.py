from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .canonical import digest


class ScaffoldError(ValueError):
    pass


@dataclass(frozen=True)
class ScaffoldSpec:
    scaffold_id: str
    version: int
    ask_on_unknown_approval: bool
    abstain_on_denied_approval: bool
    abstain_on_invalid_authority: bool
    require_world_model_prediction: bool
    verify_outcome: bool
    min_confidence_bps: int
    max_model_calls: int
    max_tool_calls: int

    MUTABLE_FIELDS = frozenset(
        {
            "ask_on_unknown_approval",
            "abstain_on_denied_approval",
            "abstain_on_invalid_authority",
            "require_world_model_prediction",
            "verify_outcome",
            "min_confidence_bps",
        }
    )

    def __post_init__(self) -> None:
        if not self.scaffold_id or self.version < 1:
            raise ScaffoldError("invalid scaffold identity")
        if not 0 <= self.min_confidence_bps <= 10000:
            raise ScaffoldError("confidence threshold must be in basis points")
        if self.max_model_calls != 1:
            raise ScaffoldError("v2 foundation permits exactly one model call")
        if self.max_tool_calls < 0 or self.max_tool_calls > 8:
            raise ScaffoldError("tool-call budget is outside the v2 foundation range")

    @classmethod
    def baseline(cls) -> "ScaffoldSpec":
        return cls(
            scaffold_id="static-model-scaffold-v1",
            version=1,
            ask_on_unknown_approval=False,
            abstain_on_denied_approval=True,
            abstain_on_invalid_authority=True,
            require_world_model_prediction=True,
            verify_outcome=True,
            min_confidence_bps=7000,
            max_model_calls=1,
            max_tool_calls=3,
        )

    @classmethod
    def permissive(cls) -> "ScaffoldSpec":
        """A scaffold with every guard off, for measuring what the model does unaided.

        `min_confidence_bps` is zero here, not the shipped 7000. A non-zero floor is itself a
        guard: `apply_scaffold_rules` routes any decision below it to ASK-if-UNKNOWN-else-ABSTAIN,
        which is the correct answer on most cells of the development matrix. Leaving it at 7000
        meant the "unprotected" baseline was never unprotected, and every improvement measured
        against it was understated at the start and overstated as a delta.

        `baseline()` pre-solves the authority and denied-approval preconditions, which hides how
        often a model gets them wrong. This variant removes that help so the failures are visible
        and a mutation can be grounded in them.
        """
        return cls(
            scaffold_id="permissive-model-scaffold-v1",
            version=1,
            ask_on_unknown_approval=False,
            abstain_on_denied_approval=False,
            abstain_on_invalid_authority=False,
            require_world_model_prediction=True,
            verify_outcome=True,
            min_confidence_bps=0,
            max_model_calls=1,
            max_tool_calls=3,
        )

    def with_patch(self, patch: dict[str, Any]) -> "ScaffoldSpec":
        if not patch or not set(patch).issubset(self.MUTABLE_FIELDS):
            forbidden = sorted(set(patch) - self.MUTABLE_FIELDS)
            raise ScaffoldError(f"patch is empty or changes a forbidden field: {forbidden}")
        candidate = replace(self, **patch)
        if candidate == self:
            raise ScaffoldError("patch does not change executable behavior")
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "abstain_on_denied_approval": self.abstain_on_denied_approval,
            "abstain_on_invalid_authority": self.abstain_on_invalid_authority,
            "ask_on_unknown_approval": self.ask_on_unknown_approval,
            "max_model_calls": self.max_model_calls,
            "max_tool_calls": self.max_tool_calls,
            "min_confidence_bps": self.min_confidence_bps,
            "require_world_model_prediction": self.require_world_model_prediction,
            "scaffold_id": self.scaffold_id,
            "verify_outcome": self.verify_outcome,
            "version": self.version,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScaffoldSpec":
        return cls(**value)
