"""Is every extracted name, date and amount really in the document, with the same value?
And is every date and amount in the document extracted?

Values are grounded, not quotes: "2026-03-03" is grounded if the document states that
date in any form. The quote only helps tell a *wrong* value (the quoted text exists but
says something else) from an *invented* one (nothing in the document supports it).
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.checks.currency import ISO_CODE
from forensics.pipeline.models import ExtractionOutput
from forensics.textscan import (
    find_dates,
    find_money,
    format_amount,
    format_date,
    name_in_text,
    parse_iso_date,
)


def check_grounding(source: str, output: ExtractionOutput) -> list[Finding]:
    findings: list[Finding] = []
    source_dates = {m.value for m in find_dates(source) if m.value}
    source_money = {}
    for m in find_money(source):
        source_money.setdefault((m.amount, m.currency), m.surface)

    for party in output.parties:
        if not name_in_text(party.name, source):
            findings.append(_finding(
                "ungrounded_party", FailureType.HALLUCINATION,
                f'party "{party.name}" ({party.role}) does not appear in the document',
                party.name, "name",
            ))

    extracted_dates: set[dt.date] = set()
    accounted_dates: set[dt.date] = set()  # true values explained by a wrong_value finding
    for d in output.dates:
        value = parse_iso_date(d.date)
        if value is None:
            continue  # reported by dates.py
        extracted_dates.add(value)
        if value in source_dates:
            continue
        quoted = [m.value for m in find_dates(d.quote) if m.value in source_dates]
        if quoted and _in_source(d.quote, source):
            accounted_dates.add(quoted[0])
            findings.append(_finding(
                "date_mismatch", FailureType.WRONG_VALUE,
                f'extracted {d.date} for "{d.label}", but '
                + _says(d.quote, format_date(quoted[0])),
                d.date, "date",
            ))
        else:
            findings.append(_finding(
                "ungrounded_date", FailureType.HALLUCINATION,
                f'date {d.date} ("{d.label}") is not stated anywhere in the document',
                d.date, "date",
            ))

    extracted_money: set[tuple[Decimal, str]] = set()
    accounted_money: set[tuple[Decimal, str]] = set()
    for a in output.amounts:
        if not ISO_CODE.fullmatch(a.currency):
            continue  # reported by currency.py
        key = (a.amount, a.currency)
        extracted_money.add(key)
        if key in source_money:
            continue
        other_currency = [k for k in source_money if k[0] == a.amount]
        quoted = [(m.amount, m.currency) for m in find_money(a.quote)
                  if (m.amount, m.currency) in source_money]
        if other_currency:
            accounted_money.add(other_currency[0])
            findings.append(_finding(
                "currency_mismatch", FailureType.WRONG_VALUE,
                f"extracted {format_amount(a.amount, a.currency)} ({a.label}), but the document gives "
                f'that amount as "{source_money[other_currency[0]]}"',
                f"{a.amount} {a.currency}", "amount",
            ))
        elif quoted and _in_source(a.quote, source):
            accounted_money.add(quoted[0])
            findings.append(_finding(
                "amount_mismatch", FailureType.WRONG_VALUE,
                f'extracted {format_amount(a.amount, a.currency)} for "{a.label}", but '
                + _says(a.quote, source_money[quoted[0]]),
                f"{a.amount} {a.currency}", "amount",
            ))
        else:
            findings.append(_finding(
                "ungrounded_amount", FailureType.HALLUCINATION,
                f'amount {format_amount(a.amount, a.currency)} ("{a.label}") is not stated anywhere '
                "in the document",
                f"{a.amount} {a.currency}", "amount",
            ))

    for value in sorted(source_dates - extracted_dates - accounted_dates):
        findings.append(_finding(
            "missed_date", FailureType.OMISSION,
            f"the document states {format_date(value)}, but it was not extracted",
            value.isoformat(), "date",
        ))
    for key, surface in source_money.items():
        if key not in extracted_money and key not in accounted_money:
            findings.append(_finding(
                "missed_amount", FailureType.OMISSION,
                f'the document states "{surface}", but it was not extracted',
                f"{key[0]} {key[1]}", "amount",
            ))
    return findings


def _says(quote: str, true_value: str) -> str:
    if _flatten(quote) == _flatten(true_value):
        return f'the document says "{quote}"'
    return f'the quoted text "{quote}" says {true_value}'


def _in_source(quote: str, source: str) -> bool:
    """Lenient on layout: ignores Markdown emphasis, table pipes and spacing."""
    return _flatten(quote) in _flatten(source)


def _flatten(text: str) -> str:
    return " ".join(re.sub(r"[*_`|#]", " ", text).split())


def _finding(check: str, failure_type: FailureType, message: str, value: str, kind: str) -> Finding:
    return Finding(step="extraction", check=check, failure_type=failure_type, message=message,
                   value=value, value_kind=kind)  # type: ignore[arg-type]
