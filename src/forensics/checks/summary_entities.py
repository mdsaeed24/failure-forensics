"""Does the summary mention only dates and amounts it was given?

A date or amount in the summary that isn't among the extracted facts is:
- wrong_value if a fact of the same kind was given but not mentioned (it was replaced),
- hallucination otherwise (it was added).
Invented names and claims can't be found by pattern matching; the judge looks for those.
"""

from __future__ import annotations

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
            replaced = bool(unmentioned_dates)
            findings.append(Finding(
                step="summarization", check="summary_date_not_given",
                failure_type=FailureType.WRONG_VALUE if replaced else FailureType.HALLUCINATION,
                message=f'the summary says "{m.surface}", a date that is not among its input facts',
                value=m.value.isoformat(), value_kind="date",
            ))
    for m in said_money:
        if (m.amount, m.currency) not in given_money:
            replaced = any(c == m.currency for _, c in unmentioned_money)
            findings.append(Finding(
                step="summarization", check="summary_amount_not_given",
                failure_type=FailureType.WRONG_VALUE if replaced else FailureType.HALLUCINATION,
                message=f'the summary says "{m.surface}", an amount that is not among its input facts',
                value=f"{m.amount} {m.currency}", value_kind="amount",
            ))
    return findings
