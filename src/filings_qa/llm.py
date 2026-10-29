"""LLM backends. The agent only needs one capability: return a JSON object."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Protocol


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model reply (tolerates ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in model reply: {text[:200]!r}")
    return json.loads(candidate[start : end + 1])


class AnthropicLLM:  # pragma: no cover - needs network and an API key
    def __init__(self, model: str, max_tokens: int = 1500):
        import anthropic

        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self._model = model
        self._max_tokens = max_tokens

    def complete_json(self, system: str, user: str) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_json_object(text)


class FakeLLM:
    """Scripted LLM for tests: `responder(system, user) -> dict`."""

    def __init__(self, responder: Callable[[str, str], dict]):
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append((system, user))
        return self._responder(system, user)


def make_llm(provider: str, model: str) -> LLM:
    if provider == "anthropic":
        return AnthropicLLM(model)
    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}")
