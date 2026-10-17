"""DeepSeek chat client (through the OpenAI SDK) that returns validated Pydantic objects.

What every call guarantees:
- JSON mode, and the reply is validated against the Pydantic model you pass in.
- Invalid JSON or a schema mismatch gets exactly one repair retry, which shows the
  model its own reply and the validation error. A second failure raises LLMOutputError.
- Rate limits, timeouts, connection errors and 5xx get up to `max_retries` retries
  with exponential backoff. Other API errors (bad key, bad request) fail immediately.
- At most `concurrency` requests are in flight at once, across all callers.
- Output is capped at `max_output_tokens`. finish_reason is logged, and "length"
  sets `truncated=True` on the call record.
- Tokens and cost are summed over every attempt, failed ones included.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from forensics.config import Settings
from forensics.llm.cost import cost_usd
from forensics.tracing import spans as attr
from forensics.tracing.spans import tracer

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_ERROR_CHARS_IN_REPAIR = 1500


class LLMCall(BaseModel):
    """Everything worth knowing about one logical call (all its attempts together)."""

    model: str
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0  # includes reasoning_tokens
    reasoning_tokens: int = 0  # hidden "thinking"; counts against max_tokens and is billed
    cost_usd: float | None = None
    latency_ms: float = 0.0
    finish_reason: str | None = None
    truncated: bool = False  # the final reply stopped at max_tokens
    repaired: bool = False  # the first reply was invalid and a repair retry was needed
    raw_output: str = ""  # the final reply, exactly as received
    reasoning: str = ""  # the final reply's reasoning text, if the model returns one


@dataclass
class LLMResult(Generic[T]):
    parsed: T
    call: LLMCall


class LLMError(Exception):
    def __init__(self, message: str, call: LLMCall) -> None:
        super().__init__(message)
        self.call = call


class LLMAPIError(LLMError):
    """The API request itself failed (after retries, if the error was retryable)."""


class LLMOutputError(LLMError):
    """The model replied, but the reply was still invalid after the repair retry."""


_recorder: ContextVar[list[LLMCall] | None] = ContextVar("llm_call_recorder", default=None)


@contextmanager
def record_calls() -> Iterator[list[LLMCall]]:
    """Collects every LLMCall made inside the block, including failed ones.

    Uses a ContextVar, so concurrent pipeline runs (separate asyncio tasks) each
    see only their own calls.
    """
    calls: list[LLMCall] = []
    token = _recorder.set(calls)
    try:
        yield calls
    finally:
        _recorder.reset(token)


def _is_retryable(exc: Exception) -> bool:
    # APITimeoutError is a subclass of APIConnectionError.
    if isinstance(exc, (APIConnectionError, RateLimitError)):
        return True
    return isinstance(exc, APIStatusError) and exc.status_code >= 500


class LLMClient:
    def __init__(self, api: Any, settings: Settings) -> None:
        """`api` is an AsyncOpenAI client, or anything with the same
        `chat.completions.create` coroutine (tests pass a FakeLLM)."""
        self._api = api
        self._settings = settings
        self._llm = settings.llm
        self._semaphore = asyncio.Semaphore(settings.llm.concurrency)

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        api = AsyncOpenAI(
            api_key=settings.require_api_key(),
            base_url=settings.llm.base_url,
            timeout=settings.llm.timeout_seconds,
            max_retries=0,  # retries are handled here, so they are counted and logged
        )
        return cls(api, settings)

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking: bool = True,
    ) -> LLMResult[T]:
        """Unset options fall back to config.yaml's llm section. `thinking=False` turns off
        DeepSeek's hidden reasoning for this call (faster, cheaper, no reasoning tokens)."""
        call = LLMCall(model=model or self._llm.pipeline_model)
        options: dict[str, Any] = {
            "max_tokens": max_tokens or self._llm.max_output_tokens,
            "temperature": self._llm.temperature if temperature is None else temperature,
        }
        if not thinking:
            options["extra_body"] = {"thinking": {"type": "disabled"}}
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        start = time.perf_counter()
        with tracer.start_as_current_span(f"chat {call.model}") as span:
            try:
                text = await self._create(messages, call, options)
                try:
                    parsed = _parse(text, schema)
                except (ValueError, ValidationError) as first_error:
                    log.warning("%s returned invalid output, retrying once: %s", call.model, first_error)
                    call.repaired = True
                    messages += [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": _repair_message(first_error)},
                    ]
                    text = await self._create(messages, call, options)
                    try:
                        parsed = _parse(text, schema)
                    except (ValueError, ValidationError) as second_error:
                        raise LLMOutputError(
                            f"{call.model} output still invalid after repair retry: {second_error}", call
                        ) from second_error
            finally:
                call.latency_ms = (time.perf_counter() - start) * 1000
                call.cost_usd = cost_usd(
                    call.model, call.prompt_tokens, call.completion_tokens, self._settings.prices
                )
                recorder = _recorder.get()
                if recorder is not None:
                    recorder.append(call)
                _record_on_span(span, call, messages, options, schema, thinking)
        return LLMResult(parsed=parsed, call=call)

    async def _create(
        self, messages: list[dict[str, str]], call: LLMCall, options: dict[str, Any]
    ) -> str:
        """One API request, retried on transient errors. Returns the reply text."""
        for attempt in range(self._llm.max_retries + 1):
            call.attempts += 1
            try:
                async with self._semaphore:
                    response = await self._api.chat.completions.create(
                        model=call.model,
                        messages=messages,
                        response_format={"type": "json_object"},
                        **options,
                    )
            except Exception as exc:
                if not _is_retryable(exc):
                    raise LLMAPIError(f"{type(exc).__name__}: {exc}", call) from exc
                if attempt == self._llm.max_retries:
                    raise LLMAPIError(
                        f"gave up after {attempt + 1} attempts: {type(exc).__name__}: {exc}", call
                    ) from exc
                # Sleep outside the semaphore so a backing-off call doesn't hold a slot.
                delay = self._llm.backoff_base_seconds * 2**attempt
                delay += random.uniform(0, delay * 0.25)
                log.warning(
                    "%s: %s, retry %d/%d in %.1fs",
                    call.model, type(exc).__name__, attempt + 1, self._llm.max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue
            return self._read(response, call)
        raise AssertionError("unreachable")

    def _read(self, response: Any, call: LLMCall) -> str:
        choice = response.choices[0]
        if response.usage is not None:
            call.prompt_tokens += response.usage.prompt_tokens
            call.completion_tokens += response.usage.completion_tokens
            details = getattr(response.usage, "completion_tokens_details", None)
            call.reasoning_tokens += getattr(details, "reasoning_tokens", None) or 0
        call.finish_reason = choice.finish_reason
        call.truncated = choice.finish_reason == "length"
        call.raw_output = choice.message.content or ""
        call.reasoning = getattr(choice.message, "reasoning_content", None) or ""
        log.info(
            "llm model=%s finish_reason=%s prompt_tokens=%s completion_tokens=%s (total so far, "
            "reasoning=%s)",
            call.model, choice.finish_reason, call.prompt_tokens, call.completion_tokens,
            call.reasoning_tokens,
        )
        if call.truncated:
            log.warning("%s hit its max_tokens cap; output is truncated", call.model)
        return call.raw_output


def _record_on_span(
    span: Any, call: LLMCall, messages: list[dict[str, str]], options: dict[str, Any],
    schema: type[BaseModel], thinking: bool,
) -> None:
    """Everything about the call goes on its span, failed calls included: the raw output
    and reasoning of a bad reply are the best evidence of why a step failed."""
    values = {
        attr.GENAI_OPERATION: "chat",
        attr.GENAI_PROVIDER: "deepseek",
        attr.GENAI_REQUEST_MODEL: call.model,
        attr.GENAI_MAX_TOKENS: options["max_tokens"],
        attr.GENAI_TEMPERATURE: options["temperature"],
        attr.GENAI_INPUT_TOKENS: call.prompt_tokens,
        attr.GENAI_OUTPUT_TOKENS: call.completion_tokens,
        attr.GENAI_FINISH_REASONS: [call.finish_reason] if call.finish_reason else [],
        attr.LLM_SCHEMA: schema.__name__,
        attr.LLM_THINKING: thinking,
        attr.LLM_ATTEMPTS: call.attempts,
        attr.LLM_REASONING_TOKENS: call.reasoning_tokens,
        attr.LLM_TRUNCATED: call.truncated,
        attr.LLM_REPAIRED: call.repaired,
        attr.LLM_MESSAGES: json.dumps(messages, ensure_ascii=False),
        attr.LLM_RAW_OUTPUT: call.raw_output,
        attr.LLM_REASONING: call.reasoning,
    }
    if call.cost_usd is not None:  # span attributes can't be None; absent means unknown
        values[attr.LLM_COST] = call.cost_usd
    span.set_attributes(values)


def _parse(text: str, schema: type[T]) -> T:
    if not text.strip():
        raise ValueError("the reply was empty")
    return schema.model_validate_json(text)


def _repair_message(error: Exception) -> str:
    detail = str(error)[:_MAX_ERROR_CHARS_IN_REPAIR]
    return (
        "Your previous reply could not be used because it was not valid JSON matching the "
        f"required format. The error was:\n{detail}\n\n"
        "Reply again with only the corrected JSON object."
    )
