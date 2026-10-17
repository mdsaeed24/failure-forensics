"""Span helpers and attribute names.

Attribute names follow the OpenTelemetry GenAI conventions where one exists
(gen_ai.*), so standard tools understand our LLM spans; everything else is forensics.*.
Without a configured TracerProvider all of this is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel

tracer = trace.get_tracer("forensics")

# Root span
DOC_ID = "forensics.doc_id"
BATCH_ID = "forensics.batch_id"
STATUS = "forensics.status"
FAILED_STEP = "forensics.failed_step"
ERROR = "forensics.error"

# Step spans
STEP_NAME = "forensics.step.name"
STEP_INDEX = "forensics.step.index"
STEP_INPUT = "forensics.step.input"
STEP_OUTPUT = "forensics.step.output"
SIGNAL_PREFIX = "forensics.signal."

# LLM call spans
GENAI_OPERATION = "gen_ai.operation.name"
GENAI_PROVIDER = "gen_ai.provider.name"
GENAI_REQUEST_MODEL = "gen_ai.request.model"
GENAI_MAX_TOKENS = "gen_ai.request.max_tokens"
GENAI_TEMPERATURE = "gen_ai.request.temperature"
GENAI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GENAI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GENAI_FINISH_REASONS = "gen_ai.response.finish_reasons"
LLM_SCHEMA = "forensics.llm.schema"
LLM_THINKING = "forensics.llm.thinking"
LLM_ATTEMPTS = "forensics.llm.attempts"
LLM_REASONING_TOKENS = "forensics.llm.reasoning_tokens"
LLM_COST = "forensics.llm.cost_usd"
LLM_TRUNCATED = "forensics.llm.truncated"
LLM_REPAIRED = "forensics.llm.repaired"
LLM_MESSAGES = "forensics.llm.messages"
LLM_RAW_OUTPUT = "forensics.llm.raw_output"
LLM_REASONING = "forensics.llm.reasoning"

# Attributes holding JSON text; the exporter decodes them so trace files are readable.
JSON_ATTRIBUTES = frozenset({STEP_INPUT, STEP_OUTPUT, LLM_MESSAGES})


def output_signals(output: BaseModel) -> dict[str, Any]:
    """Cheap facts about a step's output that the analyzer can compare across runs."""
    signals: dict[str, Any] = {}
    for field, value in output:
        if isinstance(value, list):
            signals[f"{field}_count"] = len(value)
        elif field in ("confidence", "doc_type", "char_count", "word_count"):
            signals[field] = value
        elif field == "summary":
            signals["summary_words"] = len(value.split())
    return signals


class StepSpan:
    def __init__(self, span: Span) -> None:
        self.span = span

    def set_output(self, output: BaseModel) -> None:
        self.span.set_attribute(STEP_OUTPUT, output.model_dump_json())
        for key, value in output_signals(output).items():
            self.span.set_attribute(SIGNAL_PREFIX + key, value)
        self.span.set_status(Status(StatusCode.OK))


@contextmanager
def traced_step(name: str, index: int, step_input: BaseModel) -> Iterator[StepSpan]:
    """One span per step. An exception is recorded on the span and re-raised.

    A context manager rather than a decorator on the step functions: the runner's
    hooks run inside it, so the output recorded is the one the next step receives.
    """
    attributes = {STEP_NAME: name, STEP_INDEX: index, STEP_INPUT: step_input.model_dump_json()}
    with tracer.start_as_current_span(f"step {name}", attributes=attributes) as span:
        yield StepSpan(span)


def trace_id_of(span: Span) -> str | None:
    context = span.get_span_context()
    return trace.format_trace_id(context.trace_id) if context.is_valid else None
