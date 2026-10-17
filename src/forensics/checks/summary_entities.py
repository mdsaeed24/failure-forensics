"""Does the summary mention only dates and amounts it was given?

A date or amount in the summary that isn't among the extracted facts is:
- wrong_value if it looks like a given fact gone wrong: a given date within a month of
  it, or a given amount in the same currency with the same digits or within 20%,
  that the summary doesn't otherwise mention (it was replaced, e.g. digits transposed),
- hallucination otherwise (it was added).
Invented names and claims can't be found by pattern matching; the judge looks for those.
"""

from __future__ import annotations

from decimal import Decimal

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.pipeline.models import SummarizationInput, SummaryOutput
from forensics.textscan import find_dates, find_money, parse_iso_date


def check_summary_entities(inp: SummarizationInput, output: SummaryOutput) -> list[Finding]:
    findings: list[Finding] = []
    given_dates = {v for d in inp.extraction.dates if (v := parse_iso_date(d.date))}
    given_money = {(a.amount, a.currency) for a in inp.extraction.amounts}
    said_dates = [m for m in find_dates(output.summary) if m.value]
    said_money = find_money(output.summary)
    unmentioned_dates = given_dates - {m.value for m in said_dates}
    unmentioned_money = given_money - {(m.amount, m.currency) for m in said_money}

    for m in said_dates:
        if m.value not in given_dates:
            replaced = any(abs((m.value - d).days) <= 31 for d in unmentioned_dates)
            findings.append(Finding(
                step="summarization", check="summary_date_not_given",
                failure_type=FailureType.WRONG_VALUE if replaced else FailureType.HALLUCINATION,
                message=f'the summary says "{m.surface}", a date that is not among its input facts',
                value=m.value.isoformat(), value_kind="date",
            ))
    for m in said_money:
        if (m.amount, m.currency) not in given_money:
            replaced = any(c == m.currency and _similar(m.amount, a) for a, c in unmentioned_money)
            findings.append(Finding(
                step="summarization", check="summary_amount_not_given",
                failure_type=FailureType.WRONG_VALUE if replaced else FailureType.HALLUCINATION,
                message=f'the summary says "{m.surface}", an amount that is not among its input facts',
                value=f"{m.amount} {m.currency}", value_kind="amount",
            ))
    return findings


def _similar(a: Decimal, b: Decimal) -> bool:
    digits = lambda x: sorted(f"{x:.2f}".replace(".", ""))  # noqa: E731
    return digits(a) == digits(b) or abs(a - b) <= Decimal("0.2") * max(a, b)
