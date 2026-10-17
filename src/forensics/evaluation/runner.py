"""Run the evaluation: clean runs of every document, then planted faults.

1. Clean: each document once, no fault. Every run is analyzed. A good run that the
   analyzer blames anyway is a false alarm; a bad one is a natural pipeline failure.
2. Injected: for each fault, a seeded sample of documents whose clean run was good
   (so a bad output can be blamed on the fault) and that the fault can apply to.
   Only runs where the fault took effect, i.e. the final output really was bad, are
   analyzed and counted towards accuracy.

Every finished run is saved at once, so an interrupted evaluation can be resumed.
"""

from __future__ import annotations

import asyncio
import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forensics.analyzer.root_cause import Diagnosis, analyze
from forensics.config import Settings
from forensics.evaluation.labels import EvalStore, RunRecord
from forensics.faults.injectors import Fault
from forensics.llm.client import LLMClient, LLMError
from forensics.pipeline.models import RawDocument
from forensics.pipeline.runner import PipelineOptions, PipelineResult, run_pipeline
from forensics.scoring.final_score import FinalScore, score_trace
from forensics.tracing.store import TraceStore


@dataclass
class EvalDoc:
    doc_id: str
    text: str
    truth: dict[str, Any]


Progress = Callable[[RunRecord], None]


async def run_evaluation(
    eval_id: str,
    docs: list[EvalDoc],
    faults: list[Fault],
    per_fault: int,
    seed: int,
    llm: LLMClient,
    settings: Settings,
    traces: TraceStore,
    labels: EvalStore,
    progress: Progress,
) -> list[RunRecord]:
    done = {(r.doc_id, r.fault): r for r in labels.records(eval_id) if r.error is None}

    async def run_and_save(doc: EvalDoc, fault: Fault | None) -> RunRecord:
        key = (doc.doc_id, fault.name if fault else "")
        if key in done:
            return done[key]
        record = await _one_run(eval_id, doc, fault, seed, llm, settings, traces)
        labels.save(record)
        progress(record)
        return record

    clean = await asyncio.gather(*(run_and_save(d, None) for d in docs))
    good = {r.doc_id for r in clean if r.final_good}

    tasks = []
    for fault in faults:
        eligible = [d for d in docs if d.doc_id in good and fault.applicable(d.text, d.truth)]
        sample = random.Random(f"{seed}:{fault.name}").sample(eligible, min(per_fault, len(eligible)))
        tasks += [run_and_save(d, fault) for d in sample]
    injected = await asyncio.gather(*tasks)
    return [*clean, *injected]


async def _one_run(
    eval_id: str, doc: EvalDoc, fault: Fault | None, seed: int,
    llm: LLMClient, settings: Settings, traces: TraceStore,
) -> RunRecord:
    record = RunRecord(eval_id=eval_id, kind="injected" if fault else "clean", doc_id=doc.doc_id)
    if fault:
        record.fault = fault.name
        record.mechanism = "prompt" if fault.prompt_variant else "hook"
        record.expected_step, record.expected_type = fault.step, fault.failure_type.value
        # crc32, not hash(): str hashes change every run, which would break reproducibility
        injection = fault.inject(seed=zlib.crc32(f"{seed}:{fault.name}:{doc.doc_id}".encode()))
        options = injection.options(batch_id=eval_id)
    else:
        options = PipelineOptions(batch_id=eval_id)

    raw = RawDocument(doc_id=doc.doc_id, text=doc.text)
    try:
        result = await run_pipeline(raw, llm, settings, options)
        if fault:
            record.applied, record.detail = injection.applied, injection.detail
        _record_pipeline(record, result)
        if result.trace_id is None:
            raise RuntimeError("tracing is not configured; the evaluation needs traces")
        trace_doc = traces.load(result.trace_id)
        score = score_trace(trace_doc, doc.truth)
        _record_score(record, score)

        # Clean runs are always analyzed (that's how false alarms are measured).
        # Injected runs only if the fault applied and the output really went bad.
        if fault is None or record.took_effect:
            diagnosis = await analyze(trace_doc, llm, settings, [d.message for d in score.discrepancies])
            traces.save_diagnosis(diagnosis.model_dump(mode="json"))
            _record_diagnosis(record, diagnosis)
    except LLMError as exc:  # an API outage mid-evaluation: retried on --resume
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def _record_pipeline(record: RunRecord, result: PipelineResult) -> None:
    record.trace_id = result.trace_id
    record.pipeline_status = result.status
    record.pipeline_cost = result.total_cost_usd
    if result.extraction:
        record.extraction_confidence = result.extraction.confidence
    if result.classification:
        record.classification_confidence = result.classification.confidence


def _record_score(record: RunRecord, score: FinalScore) -> None:
    record.final_good = score.good
    record.discrepancies = [d.message for d in score.discrepancies]
    fields = {d.field for d in score.discrepancies}
    if record.extraction_confidence is not None:
        record.extraction_correct = not fields & {"party", "date", "amount"}
    if record.classification_confidence is not None:
        record.classification_correct = "doc_type" not in fields


def _record_diagnosis(record: RunRecord, d: Diagnosis) -> None:
    record.analyzed = True
    record.diag_step = d.root_cause_step
    record.diag_type = d.failure_type.value if d.failure_type else None
    record.diag_confidence = d.confidence if d.root_cause_step else None
    chosen = next((v for v in d.steps if v.step == d.root_cause_step), None)
    if chosen is not None:
        record.decided_by = ("crash" if chosen.verdict == "crashed"
                             else "deterministic" if chosen.findings else "judge")
    record.judge_calls = d.judge_calls
    record.analysis_cost = d.cost_usd
