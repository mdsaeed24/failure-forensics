"""FakeLLM: stands in for the OpenAI SDK client so tests never touch the network."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx2
import openai

_REQUEST = httpx2.Request("POST", "https://fake.invalid/chat/completions")


@dataclass
class FakeReply:
    content: str | dict[str, Any]
    finish_reason: str = "stop"
    prompt_tokens: int = 100
    completion_tokens: int = 50
    reasoning_tokens: int | None = None
    reasoning: str | None = None
    delay: float = 0.0


class FakeLLM:
    """Replies come from a script, in order. Each item is:

    - a dict (sent as JSON), or a str (sent as-is, e.g. broken JSON)
    - a FakeReply, for control over finish_reason, token counts and delay
    - an Exception, which is raised as if the API had failed
    - a callable, given the request kwargs, returning any of the above (for concurrent
      callers whose order isn't fixed)
    """

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self.replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        item = self.replies.pop(0)
        if callable(item):
            item = item(kwargs)
        if isinstance(item, BaseException):
            raise item
        reply = item if isinstance(item, FakeReply) else FakeReply(item)

        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(reply.delay)
        finally:
            self.in_flight -= 1

        content = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(
                finish_reason=reply.finish_reason,
                message=SimpleNamespace(content=content, reasoning_content=reply.reasoning),
            )],
            usage=SimpleNamespace(
                prompt_tokens=reply.prompt_tokens,
                completion_tokens=reply.completion_tokens,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=reply.reasoning_tokens),
            ),
        )


def api_status_error(cls: type[openai.APIStatusError], status: int) -> openai.APIStatusError:
    return cls("fake error", response=httpx2.Response(status, request=_REQUEST), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=_REQUEST)


# Valid step outputs, matching data/samples/sample_invoice.md.
EXTRACTION = {
    "parties": [
        {"name": "Harbor Lane Printing Ltd.", "role": "seller"},
        {"name": "Meridian Health Clinics", "role": "buyer"},
        {"name": "Priya Raman", "role": "contact"},
    ],
    "dates": [
        {"date": "2026-03-03", "label": "invoice date", "quote": "3 March 2026"},
        {"date": "2026-04-02", "label": "due date", "quote": "2 April 2026"},
    ],
    "amounts": [
        {"amount": 1028.40, "currency": "GBP", "label": "total due", "quote": "£1,028.40"},
    ],
    "confidence": 5,
}
CLASSIFICATION = {"doc_type": "invoice", "reason": "It bills for printed goods.", "confidence": 5}
SUMMARY = {
    "summary": "Harbor Lane Printing Ltd. invoiced Meridian Health Clinics £1,028.40, due 2 April 2026.",
    "confidence": 5,
}


def pipeline_reply(request: dict[str, Any]) -> dict[str, Any]:
    """The right valid reply for whichever pipeline step is asking."""
    system = request["messages"][0]["content"]
    if system.startswith("You extract"):
        return EXTRACTION
    if system.startswith("You classify"):
        return CLASSIFICATION
    return SUMMARY
