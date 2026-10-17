"""The evidence chain: a plain-English walk from the symptom to the root cause and on
to where the bad value surfaced. Every line points at something in the trace."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.textscan import (
    find_dates,
    find_money,
    format_amount,
    format_date,
    name_in_text,
    parse_iso_date,
)

if TYPE_CHECKING:
    from forensics.analyzer.root_cause import StepVerdict

_MAX_FINDINGS_SHOWN = 4
_CHECKED = {  # what the deterministic checks cover, per step (classification has none)
    "intake": "text handed on intact",
    "extraction": "every value grounded in the document, nothing missed",
    "summarization": "only given dates and amounts",
}


def build_evidence(
    nodes: dict[str, dict[str, Any]],
    verdicts: list[StepVerdict],
    chosen: StepVerdict | None,
    failure_type: FailureType | None,
    symptom: list[str],
) -> list[str]:
    lines = [f"Symptom: {'; '.join(symptom)}"] if symptom else []
    for n, v in enumerate(verdicts, start=1):
        if v.verdict == "not_checked":
            lines.append(f"Step {n} {v.step}: not checked ({v.note})")
            continue
        judged = f"judge {v.judge.score}/5" if v.judge else None
        if v.verdict == "ok":
            basis = [b for b in (_CHECKED.get(v.step), judged) if b]
            lines.append(f"Step {n} {v.step}: OK ({', '.join(basis)})")
            continue
        label = {"failed": "FAILED", "crashed": "CRASHED", "suspect": "SUSPECT"}[v.verdict]
        kind = (failure_type.value if v is chosen and failure_type else
                v.judge.failure_type if v.judge else "")
        lines.append(f"Step {n} {v.step}: {label}" + (f" ({kind})" if kind else "")
                     + (f", {judged}" if judged else ""))
        for f in v.findings[:_MAX_FINDINGS_SHOWN]:
            lines.append(f"    - {f.message}")
        if len(v.findings) > _MAX_FINDINGS_SHOWN:
            lines.append(f"    - ...and {len(v.findings) - _MAX_FINDINGS_SHOWN} more")
        if v.judge and v.verdict != "ok":
            lines.append(f"    - judge: {v.judge.reason}")
            if v.judge.evidence:
                lines.append(f'    - judge quotes: "{v.judge.evidence}"')
        if v.note:
            lines.append(f"    - {v.note}")
    if chosen is not None:
        lines += _propagation(nodes, chosen, failure_type)
    return lines


def _propagation(nodes: dict[str, dict[str, Any]], chosen: StepVerdict,
                 failure_type: FailureType | None) -> list[str]:
    """Follow the root cause's bad values into the final summary."""
    summary_node = nodes.get("summarization")
    if chosen.step == "summarization" or summary_node is None:
        return []
    output = summary_node["attributes"].get("forensics.step.output")
    if not output:
        return []
    summary = output["summary"]
    lines = []

    if failure_type == FailureType.MISCLASSIFICATION and "classification" in nodes:
        label = nodes["classification"]["attributes"]["forensics.step.output"]["doc_type"]
        line = f'Downstream: summarization was told the document type is "{label}"'
        if label.replace("_", " ") in summary.lower():
            line += f', and the summary repeats it: "{summary}"'
        lines.append(line)

    seen = set()
    for f in chosen.findings[:_MAX_FINDINGS_SHOWN]:
        if (f.value, f.failure_type) in seen:  # the same value under several labels
            continue
        seen.add((f.value, f.failure_type))
        if f.value is None or f.value_kind is None:
            continue
        appears = _appears(f, summary)
        shown = _display(f)
        if f.failure_type == FailureType.OMISSION and not appears:
            lines.append(f"Downstream: {shown} never reached the summary, so it is missing there too")
        elif f.failure_type != FailureType.OMISSION and appears:
            lines.append(f"Downstream: the summary repeats {shown}: \"{summary}\"")
        elif f.failure_type != FailureType.OMISSION:
            lines.append(f"Downstream: the summary does not repeat {shown}; the error stayed "
                         "in the extracted data")
    return lines


def _display(f: Finding) -> str:
    if f.value_kind == "amount":
        parsed = _amount(f.value or "")
        if parsed:
            return format_amount(*parsed)
    if f.value_kind == "date" and (value := parse_iso_date(f.value or "")):
        return format_date(value)
    return f'"{f.value}"'


def _amount(value: str) -> tuple[Decimal, str] | None:
    try:
        number, currency = value.split()
        return Decimal(number), currency
    except (ValueError, InvalidOperation):
        return None


def _appears(f: Finding, summary: str) -> bool:
    if f.value_kind == "name":
        return name_in_text(f.value or "", summary)
    if f.value_kind == "date":
        value = parse_iso_date(f.value or "")
        return value is not None and any(m.value == value for m in find_dates(summary))
    if f.value_kind == "amount":
        parsed = _amount(f.value or "")
        return parsed is not None and any((m.amount, m.currency) == parsed for m in find_money(summary))
    return False
