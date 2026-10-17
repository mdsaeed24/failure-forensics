"""The only module that knows the order of the steps.

    intake → extraction → classification → summarization

Steps are plain functions: typed input in, typed output out. The runner wires
them together and offers two hook points that fault injection (Phase 5) uses
without touching any step code:

- `prompt_variants` picks a named prompt variant per step.
- `hooks` run after each step and may replace its output. What a hook returns
  is what the result records for that step *and* what the next step receives,
  so a planted fault looks exactly like that step failing.

A step that fails in an expected way (bad document, API error, invalid model
output) ends the run with status "error" and keeps every output produced so
far. Anything else is a bug and is raised.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from forensics.config import Settings
from forensics.llm.client import LLMCall, LLMClient, LLMError, record_calls
from forensics.pipeline import classification, extraction, intake, summarization
from forensics.pipeline.intake import IntakeError
from forensics.pipeline.models import (
    ClassificationOutput,
    ExtractionOutput,
    IntakeOutput,
    RawDocument,
    SummarizationInput,
    SummaryOutput,
)
from forensics.tracing import spans as attr
from forensics.tracing.spans import trace_id_of, traced_step, tracer

STEP_ORDER = ("intake", "extraction", "classification", "summarization")
LLM_STEPS = STEP_ORDER[1:]

Hook = Callable[[str, BaseModel], BaseModel]
"""(step_name, output) -> the output the rest of the pipeline should see."""

_EXPECTED_FAILURES = (IntakeError, LLMError)


@dataclass
class PipelineOptions:
    prompt_variants: Mapping[str, str] = field(default_factory=dict)
    hooks: Sequence[Hook] = ()
    batch_id: str | None = None  # groups the traces of one script run

    def __post_init__(self) -> None:
        unknown = set(self.prompt_variants) - set(LLM_STEPS)
        if unknown:
            raise ValueError(f"prompt_variants has unknown steps {sorted(unknown)}; use {LLM_STEPS}")

    def variant(self, step: str) -> str:
        return self.prompt_variants.get(step, "default")


class StepRecord(BaseModel):
    step: str
    status: Literal["ok", "error"]
    duration_ms: float
    llm_calls: list[LLMCall] = []
    error: str | None = None


class PipelineResult(BaseModel):
    doc_id: str
    trace_id: str | None = None  # None when tracing isn't configured
    status: Literal["ok", "error"] = "ok"
    failed_step: str | None = None
    error: str | None = None
    intake: IntakeOutput | None = None
    extraction: ExtractionOutput | None = None
    classification: ClassificationOutput | None = None
    summary: SummaryOutput | None = None
    steps: list[StepRecord] = []

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for s in self.steps for c in s.llm_calls)

    @property
    def total_cost_usd(self) -> float | None:
        costs = [c.cost_usd for s in self.steps for c in s.llm_calls]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]


class _StepFailed(Exception):
    pass


async def run_pipeline(
    doc: RawDocument,
    llm: LLMClient,
    settings: Settings,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    options = options or PipelineOptions()
    result = PipelineResult(doc_id=doc.doc_id)

    async def step(name: str, step_input: BaseModel, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        with record_calls() as calls:
            try:
                with traced_step(name, STEP_ORDER.index(name) + 1, step_input) as span:
                    output = fn()
                    if inspect.isawaitable(output):
                        output = await output
                    for hook in options.hooks:
                        output = hook(name, output)
                    span.set_output(output)
            except _EXPECTED_FAILURES as exc:
                message = f"{type(exc).__name__}: {exc}"
                result.steps.append(
                    StepRecord(step=name, status="error", duration_ms=_ms(start),
                               llm_calls=calls, error=message)
                )
                result.status, result.failed_step, result.error = "error", name, message
                raise _StepFailed from exc
        result.steps.append(
            StepRecord(step=name, status="ok", duration_ms=_ms(start), llm_calls=calls)
        )
        return output

    # Deliberately not recorded on the trace: prompt variants and hooks. They are how
    # faults get planted (Phase 5), and the analyzer must not be able to see them.
    root_attributes = {attr.DOC_ID: doc.doc_id}
    if options.batch_id:
        root_attributes[attr.BATCH_ID] = options.batch_id
    with tracer.start_as_current_span("pipeline", attributes=root_attributes) as root:
        result.trace_id = trace_id_of(root)
        try:
            result.intake = await step(
                "intake", doc, lambda: intake.run(doc, settings.pipeline)
            )
            result.extraction = await step(
                "extraction", result.intake,
                lambda: extraction.run(result.intake, llm, options.variant("extraction")),
            )
            result.classification = await step(
                "classification", result.intake,
                lambda: classification.run(result.intake, llm, options.variant("classification")),
            )
            summary_input = SummarizationInput(
                extraction=result.extraction, classification=result.classification
            )
            result.summary = await step(
                "summarization", summary_input,
                lambda: summarization.run(summary_input, llm, options.variant("summarization")),
            )
        except _StepFailed:
            pass
        root.set_attribute(attr.STATUS, result.status)
        if result.status == "error":
            root.set_attributes({attr.FAILED_STEP: result.failed_step, attr.ERROR: result.error})
            root.set_status(Status(StatusCode.ERROR, result.error))
    return result


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
