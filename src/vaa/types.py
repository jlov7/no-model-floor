from __future__ import annotations

from enum import StrEnum


class DecisionKind(StrEnum):
    ACT = "ACT"
    ASK = "ASK"
    ABSTAIN = "ABSTAIN"


class EvaluationTier(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    SEALED = "SEALED"


class PromotionStatus(StrEnum):
    ADVANCE_TO_SEALED = "ADVANCE_TO_SEALED"
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"
