"""Compare a run's final result with the ground truth: good or bad, and what's wrong.

This decides only WHETHER a run went wrong (the trigger for analysis). It is never
given to the analyzer, which has to find WHERE from the trace alone, as it would
in production, where there's a human flag and no answer key.

Facts are compared as sets: a fact the document states three times (unit price,
line total, total due) is one fact. Party roles are free text and not scored.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from forensics.pipeline.models import ExtractionOutput, SummaryOutput
from forensics.textscan import find_dates, find_money, format_amount, parse_iso_date, same_name


class Discrepancy(BaseModel):
    field: Literal["pipeline", "doc_type", "party", "date", "amount", "summary"]
    kind: Literal["failed", "wrong", "missing", "extra"]
    message: str


class FinalScore(BaseModel):
    doc_id: str
    good: bool
    discrepancies: list[Discrepancy] = []


def load_ground_truth(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def step_outputs(trace_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """step name -> its recorded output, for the steps that produced one."""
    outputs = {}
    for node in trace_doc["root"]["children"]:
        a = node["attributes"]
        if "forensics.step.output" in a:
            outputs[a["forensics.step.name"]] = a["forensics.step.output"]
    return outputs


def score_trace(trace_doc: dict[str, Any], truth: dict[str, Any]) -> FinalScore:
    found: list[Discrepancy] = []

    def add(field: str, kind: str, message: str) -> None:
        found.append(Discrepancy(field=field, kind=kind, message=message))  # type: ignore[arg-type]

    outputs = step_outputs(trace_doc)
    if trace_doc["status"] != "ok":
        add("pipeline", "failed", f"the pipeline failed at {trace_doc.get('failed_step')}")

    if "classification" in outputs:
        doc_type = outputs["classification"]["doc_type"]
        if doc_type != truth["doc_type"]:
            add("doc_type", "wrong", f'classified as "{doc_type}", should be "{truth["doc_type"]}"')

    truth_dates = {parse_iso_date(d["date"]) for d in truth["dates"]}
    truth_money = {(Decimal(a["amount"]), a["currency"]) for a in truth["amounts"]}

    if "extraction" in outputs:
        ex = ExtractionOutput.model_validate(outputs["extraction"])
        names = [p.name for p in ex.parties]
        for party in truth["parties"]:
            if not any(same_name(party["name"], n) for n in names):
                add("party", "missing", f'party "{party["name"]}" was not extracted')
        for name in names:
            if not any(same_name(name, p["name"]) for p in truth["parties"]):
                add("party", "extra", f'"{name}" was extracted but is not a party')

        dates = {parse_iso_date(d.date) or d.date for d in ex.dates}
        for value in sorted(truth_dates - dates, key=str):
            add("date", "missing", f"date {value} was not extracted")
        for value in sorted(dates - truth_dates, key=str):
            add("date", "extra", f"date {value} was extracted but is not in the document's facts")

        money = {(a.amount, a.currency) for a in ex.amounts}
        for amount, currency in sorted(truth_money - money):
            add("amount", "missing", f"amount {format_amount(amount, currency)} was not extracted")
        for amount, currency in sorted(money - truth_money):
            add("amount", "extra", f"amount {_amount(amount, currency)} was extracted but is not "
                "in the document's facts")

    if "summarization" in outputs:
        summary = SummaryOutput.model_validate(outputs["summarization"]).summary
        for m in find_dates(summary):
            if m.value and m.value not in truth_dates:
                add("summary", "wrong", f'the summary states "{m.surface}", which is not a true date')
        for m in find_money(summary):
            if (m.amount, m.currency) not in truth_money:
                add("summary", "wrong", f'the summary states "{m.surface}", which is not a true amount')

    return FinalScore(doc_id=trace_doc["doc_id"], good=not found, discrepancies=found)


def _amount(amount: Decimal, currency: str) -> str:
    return format_amount(amount, currency) if re.fullmatch(r"[A-Z]{3}", currency) else f"{amount} {currency}"
