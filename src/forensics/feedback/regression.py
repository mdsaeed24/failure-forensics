"""Re-run every eval case through the current pipeline: is the known issue gone?

For each case, the document (taken from the case's source trace, so it works for
documents with no ground truth too) goes through the pipeline as it is configured
now, with no faults. The case passes if:
- the failing step's deterministic checks find nothing,
- the judge, told what the human reported, scores that step above "suspect",
- and, when the document has ground truth, the final output matches it.

Status: open -> resolved when it passes; resolved -> regressed when it fails again.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from forensics.analyzer.judge import judge_step
from forensics.analyzer.root_cause import step_nodes
from forensics.checks import run_checks
from forensics.config import Settings
from forensics.feedback.eval_cases import EvalCase, RegressionResult
from forensics.llm.client import LLMClient, LLMError
from forensics.pipeline.models import RawDocument
from forensics.pipeline.runner import PipelineOptions, run_pipeline
from forensics.scoring.final_score import score_trace
from forensics.tracing import spans as attr
from forensics.tracing.store import TraceStore


async def check_case(
    case: EvalCase, llm: LLMClient, settings: Settings, traces: TraceStore,
    truth: dict[str, dict[str, Any]], batch_id: str,
) -> RegressionResult:
    now = dt.datetime.now(dt.UTC)
    source = traces.load(case.source_trace_id)
    raw = step_nodes(source)["intake"]["attributes"][attr.STEP_INPUT]
    doc = RawDocument.model_validate(raw)

    try:
        result = await run_pipeline(doc, llm, settings, PipelineOptions(batch_id=batch_id))
        if result.trace_id is None:
            raise RuntimeError("tracing is not configured; regression needs traces")
        trace_doc = traces.load(result.trace_id)
        node = step_nodes(trace_doc).get(case.root_cause_step)
        if node is None or node["status"] == "error":
            reason = f"{case.root_cause_step} did not complete: {node['error'] if node else 'never ran'}"
            return RegressionResult(run_at=now, trace_id=result.trace_id, passed=False, reason=reason)

        a = node["attributes"]
        findings = run_checks(case.root_cause_step, a[attr.STEP_INPUT], a[attr.STEP_OUTPUT])
        if findings:
            return RegressionResult(run_at=now, trace_id=result.trace_id, passed=False,
                                    reason=findings[0].message)
        if case.root_cause_step != "intake":
            verdict, _ = await judge_step(case.root_cause_step, a[attr.STEP_INPUT], a[attr.STEP_OUTPUT],
                                          llm, settings, known_issue=case.note)
            if verdict.score <= settings.analysis.suspect_at_or_below:
                return RegressionResult(run_at=now, trace_id=result.trace_id, passed=False,
                                        reason=f"judge {verdict.score}/5: {verdict.reason}")
        if case.doc_id in truth:
            score = score_trace(trace_doc, truth[case.doc_id])
            if not score.good:
                return RegressionResult(run_at=now, trace_id=result.trace_id, passed=False,
                                        reason="final output: " + "; ".join(d.message for d in score.discrepancies))
        return RegressionResult(run_at=now, trace_id=result.trace_id, passed=True,
                                reason="the step now passes its checks and the judge")
    except LLMError as exc:
        return RegressionResult(run_at=now, trace_id=None, passed=False,
                                reason=f"could not run: {type(exc).__name__}: {exc}")


def apply_result(case: EvalCase, result: RegressionResult) -> EvalCase:
    if result.trace_id is None:  # an API failure says nothing about the issue
        status = case.status
    elif result.passed:
        status = "resolved"
    else:
        status = "regressed" if case.status == "resolved" else "open"
    return case.model_copy(update={"status": status, "history": [*case.history, result]})
