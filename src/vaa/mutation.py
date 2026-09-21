from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .agent import TrajectoryRecord, apply_scaffold_rules
from .canonical import digest
from .environment import expected_decision_for
from .scaffold import ScaffoldSpec
from .types import DecisionKind


@dataclass(frozen=True)
class MutationProposal:
    baseline_sha256: str
    candidate: ScaffoldSpec
    changed_fields: tuple[str, ...]
    evidence_trajectory_sha256s: tuple[str, ...]
    rationale: str

    @property
    def candidate_sha256(self) -> str:
        return self.candidate.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_sha256": self.baseline_sha256,
            "candidate": self.candidate.to_dict(),
            "candidate_sha256": self.candidate_sha256,
            "changed_fields": list(self.changed_fields),
            "evidence_trajectory_sha256s": list(self.evidence_trajectory_sha256s),
            "rationale": self.rationale,
        }

    @property
    def sha256(self) -> str:
        return digest(self.to_dict())


def _would_fix(trajectory: TrajectoryRecord, candidate: ScaffoldSpec) -> bool:
    """Replay this trajectory's model decision under the candidate and check it comes out right.

    A rule is only allowed to cite evidence it demonstrably repairs. Without this a guard could be
    proposed from any co-occurring failure and appear justified by trajectories it does nothing for.
    """
    expected = expected_decision_for(trajectory.authority_status, trajectory.approval_status)
    applied = apply_scaffold_rules(
        candidate,
        authority_status=trajectory.authority_status,
        approval_status=trajectory.approval_status,
        decision=trajectory.model_decision,
        confidence_bps=trajectory.confidence_bps,
    )
    return applied is expected


# Ordered by severity of the missing precondition. Acting with no valid mandate at all is worse
# than acting against a refusal, which is worse than acting on an unresolved question. The first
# rule with repairable evidence whose guard is still off is the one proposed, so a single bounded
# mutation is returned per round and the next round can propose the next guard.
#
# Evidence is any failing trajectory the guard repairs, not only an unsafe one. Restricting it to
# unsafe actions left the deriver blind to a model that is wrong in a safe direction - `qwen3.5:4b`
# answers ASK on a denied adjustment, which is incorrect but never unsafe, so no guard could ever
# be grounded in it and the model stalled 1111 bps short with a fix available.
_RULES: tuple[tuple[str, Callable[[TrajectoryRecord], bool], str], ...] = (
    (
        "abstain_on_invalid_authority",
        lambda t: t.authority_status != "VALID",
        "Observed an unsuccessful trajectory under invalid authority; refuse to act whenever the "
        "authority envelope is not VALID.",
    ),
    (
        "abstain_on_denied_approval",
        lambda t: t.approval_status == "DENIED",
        "Observed an unsuccessful trajectory where approval was explicitly denied; refuse to act "
        "on a denied approval.",
    ),
    (
        "ask_on_unknown_approval",
        lambda t: t.approval_status == "UNKNOWN",
        "Observed an unsuccessful high-consequence trajectory without established approval; "
        "require a clarification instead.",
    ),
)


def propose_mutation(
    baseline: ScaffoldSpec, trajectories: list[TrajectoryRecord]
) -> MutationProposal | None:
    """Derive one bounded scaffold mutation from observed failures, or None if none is supported.

    Every proposal is grounded in trajectories this scaffold actually produced, and cites only
    those the proposed guard would have repaired. Nothing is proposed from a rule the baseline
    already satisfies, and nothing is proposed without evidence.
    """
    failures = [t for t in trajectories if not t.task_success]
    if not failures:
        return None
    for field, predicate, rationale in _RULES:
        if getattr(baseline, field):
            continue
        candidate = baseline.with_patch({field: True})
        evidence = [t for t in failures if predicate(t) and _would_fix(t, candidate)]
        if not evidence:
            continue
        return MutationProposal(
            baseline_sha256=baseline.sha256,
            candidate=candidate,
            changed_fields=(field,),
            evidence_trajectory_sha256s=tuple(item.sha256 for item in evidence),
            rationale=rationale,
        )
    return None
