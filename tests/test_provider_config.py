from __future__ import annotations

import json
import unittest
import urllib.request
from contextlib import contextmanager
from typing import Any
from unittest import mock

from vaa.providers import (
    DECISION_RESPONSE_FORMAT,
    OpenAICompatibleProvider,
    ProviderError,
)

OBSERVATION = {
    "scenario_id": "provider-check",
    "organization_id": "provider-check",
    "reported_revenue": 100,
    "evidence": {"accepted_revenue": 110, "late_adjustment_approval": "UNKNOWN"},
    "authority_status": "VALID",
    "risk_tier": "HIGH",
    "candidate_actions": ["apply-late-adjustment"],
}

VALID_CONTENT = json.dumps(
    {
        "decision": "ASK",
        "action_id": None,
        "confidence_bps": 9000,
        "question": "Is the late adjustment approved?",
        "assumptions": [],
    }
)


@contextmanager
def captured_request(content: str = VALID_CONTENT):
    seen: dict[str, Any] = {}

    class _Response:
        status = 200

        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *_: object) -> bool:
            return False

    def fake_urlopen(request: Any, *_: Any, **__: Any) -> _Response:
        seen["body"] = json.loads(request.data.decode("utf-8"))
        seen["url"] = request.full_url
        return _Response()

    with mock.patch.object(urllib.request, "urlopen", fake_urlopen):
        yield seen


class ProviderConfigurationTests(unittest.TestCase):
    def test_environment_defaults_are_unchanged(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            provider = OpenAICompatibleProvider.from_environment()
        self.assertEqual(provider.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(provider.model, "local-model")
        self.assertEqual(provider.timeout_seconds, 90)
        self.assertIsNone(provider.response_format)

    def test_timeout_override_is_honoured(self) -> None:
        with mock.patch.dict("os.environ", {"VAA_MODEL_TIMEOUT_SECONDS": "600"}, clear=True):
            self.assertEqual(OpenAICompatibleProvider.from_environment().timeout_seconds, 600)

    def test_non_integer_timeout_is_rejected(self) -> None:
        with mock.patch.dict("os.environ", {"VAA_MODEL_TIMEOUT_SECONDS": "soon"}, clear=True):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider.from_environment()

    def test_non_positive_timeout_is_rejected(self) -> None:
        with mock.patch.dict("os.environ", {"VAA_MODEL_TIMEOUT_SECONDS": "0"}, clear=True):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider.from_environment()

    def test_unsupported_response_format_is_rejected(self) -> None:
        with self.assertRaises(ProviderError):
            OpenAICompatibleProvider(base_url="http://x/v1", model="m", response_format="yaml")

    def test_default_request_body_omits_response_format(self) -> None:
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        self.assertNotIn("response_format", seen["body"])
        self.assertEqual(seen["body"]["temperature"], 0)
        self.assertEqual(seen["body"]["seed"], 3)

    def test_json_schema_request_body_carries_the_closed_contract(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://x/v1", model="m", response_format="json_schema"
        )
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        self.assertEqual(seen["body"]["response_format"], DECISION_RESPONSE_FORMAT)
        schema = DECISION_RESPONSE_FORMAT["json_schema"]["schema"]
        self.assertEqual(
            set(schema["required"]),
            {"decision", "action_id", "confidence_bps", "question", "assumptions"},
        )
        self.assertFalse(schema["additionalProperties"])

    def test_schema_request_does_not_repair_a_non_conforming_response(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://x/v1", model="m", response_format="json_schema"
        )
        fenced = f"```json\n{VALID_CONTENT}\n```"
        with captured_request(content=fenced), self.assertRaises(ProviderError):
            provider.decide(OBSERVATION, seed=3, configuration="cfg")

    def test_generation_bounds_are_absent_by_default(self) -> None:
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        self.assertNotIn("max_tokens", seen["body"])
        self.assertNotIn("reasoning_effort", seen["body"])

    def test_max_output_tokens_is_sent_when_set(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://x/v1", model="m", max_output_tokens=1200
        )
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        self.assertEqual(seen["body"]["max_tokens"], 1200)

    def test_reasoning_effort_is_sent_when_set(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://x/v1", model="m", reasoning_effort="none"
        )
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        self.assertEqual(seen["body"]["reasoning_effort"], "none")

    def test_non_positive_max_output_tokens_is_rejected(self) -> None:
        with self.assertRaises(ProviderError):
            OpenAICompatibleProvider(base_url="http://x/v1", model="m", max_output_tokens=0)

    def test_generation_bounds_are_read_from_the_environment(self) -> None:
        env = {"VAA_MODEL_MAX_TOKENS": "800", "VAA_MODEL_REASONING_EFFORT": "none"}
        with mock.patch.dict("os.environ", env, clear=True):
            provider = OpenAICompatibleProvider.from_environment()
        self.assertEqual(provider.max_output_tokens, 800)
        self.assertEqual(provider.reasoning_effort, "none")

    def test_non_integer_max_tokens_is_rejected(self) -> None:
        with mock.patch.dict("os.environ", {"VAA_MODEL_MAX_TOKENS": "lots"}, clear=True):
            with self.assertRaises(ProviderError):
                OpenAICompatibleProvider.from_environment()

    def test_a_truncated_response_still_fails_rather_than_being_repaired(self) -> None:
        provider = OpenAICompatibleProvider(
            base_url="http://x/v1", model="m", max_output_tokens=10
        )
        with captured_request(content=""), self.assertRaises(ProviderError):
            provider.decide(OBSERVATION, seed=3, configuration="cfg")

    def test_system_prompt_states_the_confidence_unit(self) -> None:
        # Local models otherwise return confidence as a percentage. The contract accepts it
        # (0..10000) and the scaffold then silently downgrades the decision, so a unit error is
        # indistinguishable from genuine low confidence.
        provider = OpenAICompatibleProvider(base_url="http://x/v1", model="m")
        with captured_request() as seen:
            provider.decide(OBSERVATION, seed=3, configuration="cfg")
        system = seen["body"]["messages"][0]["content"]
        self.assertIn("basis points", system)
        self.assertIn("10000", system)
        self.assertIn("never express it as a percentage", system)


if __name__ == "__main__":
    unittest.main()
