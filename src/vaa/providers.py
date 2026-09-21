from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .types import DecisionKind


class ProviderError(RuntimeError):
    pass


# Constrains generation at the endpoint so the closed contract is requested rather than merely
# described in prose. It never repairs a response: `ModelDecision.from_mapping` still rejects
# anything that does not satisfy the contract, including output from an endpoint that ignores this.
DECISION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "vaa_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "decision",
                "action_id",
                "confidence_bps",
                "question",
                "assumptions",
            ],
            "properties": {
                "decision": {"type": "string", "enum": ["ACT", "ASK", "ABSTAIN"]},
                "action_id": {"type": ["string", "null"]},
                "confidence_bps": {"type": "integer", "minimum": 0, "maximum": 10000},
                "question": {"type": ["string", "null"]},
                "assumptions": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


@dataclass(frozen=True)
class ModelDecision:
    decision: DecisionKind
    action_id: str | None
    confidence_bps: int
    question: str | None
    assumptions: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> ModelDecision:
        if not isinstance(value, dict) or set(value) != {
            "decision",
            "action_id",
            "confidence_bps",
            "question",
            "assumptions",
        }:
            raise ProviderError("model decision does not match the closed JSON contract")
        try:
            decision = DecisionKind(value["decision"])
        except (TypeError, ValueError) as exc:
            raise ProviderError("model decision is unsupported") from exc
        confidence = value["confidence_bps"]
        if (
            not isinstance(confidence, int)
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 10000
        ):
            raise ProviderError("confidence_bps must be an integer from 0 to 10000")
        action_id = value["action_id"]
        question = value["question"]
        assumptions = value["assumptions"]
        if action_id is not None and not isinstance(action_id, str):
            raise ProviderError("action_id must be a string or null")
        if question is not None and not isinstance(question, str):
            raise ProviderError("question must be a string or null")
        if not isinstance(assumptions, list) or any(
            not isinstance(item, str) for item in assumptions
        ):
            raise ProviderError("assumptions must be a list of strings")
        if decision is DecisionKind.ACT and action_id != "apply-late-adjustment":
            raise ProviderError("ACT requires the permitted action identifier")
        if decision is not DecisionKind.ACT and action_id is not None:
            raise ProviderError("non-ACT decisions must not carry an action identifier")
        return cls(decision, action_id, confidence, question, tuple(assumptions))

    @classmethod
    def from_json(cls, raw: str) -> ModelDecision:
        try:
            return cls.from_mapping(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ProviderError("model did not return valid JSON") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "assumptions": list(self.assumptions),
            "confidence_bps": self.confidence_bps,
            "decision": self.decision.value,
            "question": self.question,
        }


class ModelProvider(Protocol):
    name: str

    def decide(
        self,
        observation: dict[str, Any],
        *,
        seed: int,
        configuration: str,
    ) -> ModelDecision: ...


class ScriptedProvider:
    """Deterministic provider used for CI and mechanism-isolation experiments."""

    name = "scripted-provider-v1"

    def __init__(self, decisions: dict[str, dict[str, Any]]) -> None:
        self._decisions = decisions
        self.calls = 0

    def decide(
        self,
        observation: dict[str, Any],
        *,
        seed: int = 0,
        configuration: str = "default",
    ) -> ModelDecision:
        del seed, configuration
        self.calls += 1
        scenario_id = observation.get("scenario_id")
        try:
            value = self._decisions[str(scenario_id)]
        except KeyError as exc:
            raise ProviderError(f"no scripted decision for scenario {scenario_id!r}") from exc
        return ModelDecision.from_mapping(value)


class OpenAICompatibleProvider:
    """Minimal `/v1/chat/completions` client for local or hosted compatible endpoints.

    Credentials are read at call time and are never returned in evidence artifacts.
    """

    name = "openai-compatible-chat-completions"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 90,
        response_format: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if response_format not in (None, "json_schema"):
            raise ProviderError("unsupported response format")
        if max_output_tokens is not None and max_output_tokens < 1:
            raise ProviderError("max_output_tokens must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort

    @classmethod
    def from_environment(cls) -> OpenAICompatibleProvider:
        base_url = os.environ.get("VAA_MODEL_BASE_URL", "http://127.0.0.1:1234/v1")
        model = os.environ.get("VAA_MODEL", "local-model")
        api_key = os.environ.get("VAA_MODEL_API_KEY")
        timeout = os.environ.get("VAA_MODEL_TIMEOUT_SECONDS")
        response_format = os.environ.get("VAA_MODEL_RESPONSE_FORMAT") or None
        reasoning_effort = os.environ.get("VAA_MODEL_REASONING_EFFORT") or None
        raw_max_tokens = os.environ.get("VAA_MODEL_MAX_TOKENS")
        try:
            timeout_seconds = 90 if timeout is None else int(timeout)
        except ValueError as exc:
            raise ProviderError("VAA_MODEL_TIMEOUT_SECONDS must be an integer") from exc
        if timeout_seconds < 1:
            raise ProviderError("VAA_MODEL_TIMEOUT_SECONDS must be positive")
        try:
            max_output_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
        except ValueError as exc:
            raise ProviderError("VAA_MODEL_MAX_TOKENS must be an integer") from exc
        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            response_format=response_format,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )

    def decide(
        self,
        observation: dict[str, Any],
        *,
        seed: int = 0,
        configuration: str = "default",
    ) -> ModelDecision:
        system = (
            "You are a bounded enterprise reporting agent. Return exactly one JSON object with "
            "keys decision, action_id, confidence_bps, question, assumptions. decision is ACT, "
            "ASK, or ABSTAIN. ACT is permitted only for action_id apply-late-adjustment. Use null "
            "for action_id on ASK/ABSTAIN and for question when no question is needed. "
            "confidence_bps is an integer in basis points from 0 to 10000, where 10000 means "
            "fully certain and 9500 means 95 percent confident; never express it as a "
            "percentage. Do not add markdown or commentary."
        )
        user = json.dumps(
            {"configuration": configuration, "observation": observation},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "seed": seed,
        }
        if self.response_format == "json_schema":
            body["response_format"] = DECISION_RESPONSE_FORMAT
        if self.max_output_tokens is not None:
            # A hard ceiling on generation. A model that does not terminate its reasoning is
            # otherwise bounded only by the request timeout.
            body["max_tokens"] = self.max_output_tokens
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ProviderError(f"model endpoint failed: {type(exc).__name__}") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("model endpoint response lacks message content") from exc
        if not isinstance(content, str):
            raise ProviderError("model endpoint content is not text")
        return ModelDecision.from_json(content)
