"""Walk the steps in order and name the FIRST one that failed.

For each step, cheapest and most certain evidence first:
1. Did the step crash? (recorded on its span) -> failed.
2. Deterministic checks against the step's own input -> any finding means failed.
3. Otherwise the LLM judge scores it 1-5 -> failed at <= 2, suspect at 3.

The walk stops at the first failed step: a later step that faithfully carried a bad
input forward didn't fail, and judging it would only cost money. If nothing clearly
failed, the first suspect is named, with low confidence. If nothing is even
suspect, the analyzer says so instead of guessing.

The analyzer sees only the trace. It never sees the ground truth or how a run was set up.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel

from forensics.analyzer.evidence import build_evidence
from forensics.analyzer.judge import JudgeVerdict, judge_step
from forensics.analyzer.taxonomy import FailureType, Finding, primary_type
from forensics.checks import run_checks
from forensics.config import Settings
from forensics.llm.client import LLMClient, record_calls
from forensics.pipeline.runner import STEP_ORDER
from forensics.tracing import spans as attr

Verdict = Literal["ok", "suspect", "failed", "crashed", "not_checked"]


class StepVerdict(BaseModel):
    step: str
    verdict: Verdict
    findings: list[Finding] = []
    judge: JudgeVerdict | None = None
    note: str | None = None


class Diagnosis(BaseModel):
    trace_id: str
    doc_id: str
    root_cause_step: str | None
    failure_type: FailureType | None
    confidence: Literal["high", "medium", "low"]
    headline: str
    steps: list[StepVerdict]
    evidence: list[str]
    symptom: list[str] = []  # what was wrong with the final output, if known; display only
    judge_model: str
    judge_calls: int
    cost_usd: float | None
    analyzed_at: dt.datetime


def step_nodes(trace_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        n["attributes"][attr.STEP_NAME]: n
        for n in trace_doc["root"]["children"]
        if attr.STEP_NAME in n["attributes"]
    }


async def analyze(
    trace_doc: dict[str, Any], llm: LLMClient, settings: Settings, symptom: list[str] | None = None,
) -> Diagnosis:
    nodes = step_nodes(trace_doc)
    thresholds = settings.analysis
    verdicts: list[StepVerdict] = []
    culprit: StepVerdict | None = None
    first_suspect: StepVerdict | None = None

    with record_calls() as calls:
        for name in STEP_ORDER:
            node = nodes.get(name)
            if culprit is not None:
                verdicts.append(StepVerdict(step=name, verdict="not_checked",
                                            note="after the root cause"))
                continue
            if node is None:
                verdicts.append(StepVerdict(step=name, verdict="not_checked", note="never ran"))
                continue
            if node["status"] == "error":
                culprit = _crashed(name, node["error"] or "unknown error")
                verdicts.append(culprit)
                continue

            a = node["attributes"]
            findings = run_checks(name, a[attr.STEP_INPUT], a[attr.STEP_OUTPUT])
            if findings:
                culprit = StepVerdict(step=name, verdict="failed", findings=findings)
                verdicts.append(culprit)
                continue
            if name == "intake":  # plain code: the checks cover it fully
                verdicts.append(StepVerdict(step=name, verdict="ok"))
                continue

            judge, _ = await judge_step(name, a[attr.STEP_INPUT], a[attr.STEP_OUTPUT], llm, settings)
            if judge.score <= thresholds.fail_at_or_below:
                culprit = StepVerdict(step=name, verdict="failed", judge=judge)
                verdicts.append(culprit)
            elif judge.score <= thresholds.suspect_at_or_below:
                verdict = StepVerdict(step=name, verdict="suspect", judge=judge)
                first_suspect = first_suspect or verdict
                verdicts.append(verdict)
            else:
                verdicts.append(StepVerdict(step=name, verdict="ok", judge=judge))

    chosen = culprit or first_suspect
    failure_type, confidence = _classify(chosen)
    costs = [c.cost_usd for c in calls]
    evidence = build_evidence(nodes, verdicts, chosen, failure_type, symptom or [])
    return Diagnosis(
        trace_id=trace_doc["trace_id"],
        doc_id=trace_doc["doc_id"],
        root_cause_step=chosen.step if chosen else None,
        failure_type=failure_type,
        confidence=confidence,
        headline=_headline(chosen, failure_type, confidence),
        steps=verdicts,
        evidence=evidence,
        symptom=symptom or [],
        judge_model=settings.llm.judge_model,
        judge_calls=len(calls),
        cost_usd=None if None in costs else sum(costs),  # type: ignore[arg-type]
        analyzed_at=dt.datetime.now(dt.UTC),
    )


def _crashed(step: str, error: str) -> StepVerdict:
    if error.startswith("LLMAPIError"):
        return StepVerdict(step=step, verdict="crashed",
                           note=f"the API call failed, which is infrastructure, not model behaviour: {error}")
    return StepVerdict(step=step, verdict="crashed", findings=[Finding(
        step=step, check="step_crashed", failure_type=FailureType.FORMAT_ERROR,
        message=f"the step produced no usable output: {error}",
    )])


def _classify(chosen: StepVerdict | None) -> tuple[FailureType | None, Literal["high", "medium", "low"]]:
    if chosen is None:
        return None, "low"
    if chosen.findings:
        return primary_type(chosen.findings), "high"
    if chosen.verdict == "crashed":  # API error: a known step, but no failure type
        return None, "high"
    assert chosen.judge is not None
    judged = None if chosen.judge.failure_type == "none" else FailureType(chosen.judge.failure_type)
    if chosen.step == "classification":
        judged = FailureType.MISCLASSIFICATION  # the only way a valid label can be wrong
    elif judged == FailureType.MISCLASSIFICATION:
        # Only the classifier assigns types. A summary that describes the document as the
        # wrong kind is making an unsupported claim.
        judged = FailureType.HALLUCINATION
    return judged, "medium" if chosen.verdict == "failed" else "low"


def _headline(chosen: StepVerdict | None, failure_type: FailureType | None, confidence: str) -> str:
    if chosen is None:
        return "No step-level failure found: every step looks like a faithful transformation of its input."
    if chosen.verdict == "crashed" and failure_type is None:
        return f"{chosen.step} did not run to completion: {chosen.note}"
    reason = chosen.findings[0].message if chosen.findings else (chosen.judge.reason if chosen.judge else "")
    kind = failure_type.value if failure_type else "unclassified failure"
    return f"Root cause: {chosen.step} ({kind}, {confidence} confidence). {reason}"
