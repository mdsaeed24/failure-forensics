"""Checks that a generated document says exactly what its fact sheet says.

Two layers, cheapest first:

1. Deterministic (free, certain): every fact is present, no *other* date or
   money amount appears, and there are no "[placeholders]". Extra facts matter as much as missing ones: the
   pipeline would extract them, and a correct extraction would then be scored
   wrong against the answer key.
2. LLM audit (runs only if layer 1 passes): an independent model lists every
   name, date and amount it can find and says what type of document it reads
   as. Anything it finds that isn't in the fact sheet is an error. It catches
   what regex can't: names, the document's type, and money written without a
   symbol (a table column headed "Unit Price (AUD)") or in words.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from forensics.config import ModelCallSettings
from forensics.documents.fact_sheets import FactSheet
from forensics.llm.client import LLMClient
from forensics.pipeline.models import DOCUMENT_TYPE_DEFINITIONS, DocumentType
from forensics.textscan import (
    find_dates,
    find_money,
    format_amount,
    format_date,
    line_of,
    same_name,
)


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    check: str
    message: str


class FoundFact(BaseModel):
    kind: Literal["party", "date", "amount"]
    index: int  # position in the fact sheet's list of that kind
    surface: str  # how the document wrote it
    line: int


class AuditAmount(BaseModel):
    amount: Decimal
    currency: str


class AuditOutput(BaseModel):
    names: list[str]
    dates: list[str]
    amounts: list[AuditAmount]
    doc_type: DocumentType
    doc_type_reason: str


class Verification(BaseModel):
    passed: bool
    issues: list[Issue] = []
    found: list[FoundFact] = []
    audit: AuditOutput | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]


# "[address]", "[Date]" — but not Markdown links "[text](url)" or checkboxes "[ ]" / "[x]".
_PLACEHOLDER = re.compile(r"\[(?!\s?\]|x\])[^\]\n]{1,40}\](?!\()")


def check_deterministic(sheet: FactSheet, text: str) -> tuple[list[Issue], list[FoundFact]]:
    issues: list[Issue] = []
    found: list[FoundFact] = []

    for i, party in enumerate(sheet.parties):
        index = text.find(party.name)
        if index < 0:
            near = text.casefold().find(party.name.casefold())
            hint = (f'; it appears as "{text[near:near + len(party.name)]}". Use exactly this '
                    f'capitalization everywhere, headers included: "{party.name}"') if near >= 0 else ""
            issues.append(_error("missing_fact", f'party "{party.name}" ({party.role}) does not '
                                 f"appear exactly as written{hint}"))
        else:
            found.append(FoundFact(kind="party", index=i, surface=party.name,
                                   line=line_of(text, index)))

    for m in _PLACEHOLDER.finditer(text):
        issues.append(_error("placeholder", f'"{m.group(0)}" is a placeholder; write a real '
                             "detail or leave it out"))

    dates = find_dates(text)
    for mention in dates:
        if mention.ambiguous:
            issues.append(_error("ambiguous_date", f'"{mention.surface}" is a numeric date; '
                                 "write dates with the month name"))
        elif mention.value is None:
            issues.append(_error("invalid_date", f'"{mention.surface}" is not a real date'))
    sheet_dates = {d.date for d in sheet.dates}
    for i, fact in enumerate(sheet.dates):
        hit = next((m for m in dates if m.value == fact.date), None)
        if hit is None:
            issues.append(_error("missing_fact", f"date {format_date(fact.date)} ({fact.label}) "
                                 "is missing"))
        else:
            found.append(FoundFact(kind="date", index=i, surface=hit.surface,
                                   line=line_of(text, hit.start)))
    for mention in dates:
        if mention.value is not None and mention.value not in sheet_dates:
            issues.append(_error("extra_date", f'"{mention.surface}" is a date that is not in '
                                 "the fact sheet"))

    money = find_money(text)
    sheet_amounts = {(a.amount, a.currency) for a in sheet.amounts}
    for i, fact in enumerate(sheet.amounts):
        hit = next((m for m in money if (m.amount, m.currency) == (fact.amount, fact.currency)), None)
        if hit is None:
            issues.append(_error("missing_fact", f"amount {format_amount(fact.amount, fact.currency)} "
                                 f"({fact.label}) is missing, or written without its currency "
                                 "symbol or code"))
        else:
            found.append(FoundFact(kind="amount", index=i, surface=hit.surface,
                                   line=line_of(text, hit.start)))
    for mention in money:
        if (mention.amount, mention.currency) not in sheet_amounts:
            issues.append(_error("extra_amount", f'"{mention.surface}" is a money amount that is '
                                 "not in the fact sheet"))
    return issues, found


_AUDIT_SYSTEM = f"""\
You audit business documents for a test dataset. Read the document carefully and
reply with a single JSON object and nothing else, in exactly this shape:
{{
  "names": ["<every person and every organization named anywhere>"],
  "dates": ["<every calendar date mentioned, as YYYY-MM-DD>"],
  "amounts": [{{"amount": 1234.50, "currency": "<ISO 4217 code>"}}],
  "doc_type": "<one label from the list>",
  "doc_type_reason": "<one sentence>"
}}

"names" means people and organizations only (companies, institutions, banks,
agencies). Not places, products, job titles or departments.
"amounts" means money only, including amounts written in words.

Document type labels:
""" + "\n".join(f'- "{k}": {v}' for k, v in DOCUMENT_TYPE_DEFINITIONS.items())


async def audit(text: str, llm: LLMClient, settings: ModelCallSettings) -> AuditOutput:
    result = await llm.complete_json(
        system=_AUDIT_SYSTEM,
        user=f"<document>\n{text}\n</document>",
        schema=AuditOutput,
        model=settings.model,
        max_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
        thinking=settings.thinking,
    )
    return result.parsed


def compare_audit(sheet: FactSheet, report: AuditOutput) -> list[Issue]:
    issues: list[Issue] = []
    if report.doc_type != sheet.doc_type:
        issues.append(_error("doc_type", f'reads as "{report.doc_type}", not "{sheet.doc_type}": '
                             f"{report.doc_type_reason}"))
    for name in report.names:
        if not any(same_name(name, p.name) for p in sheet.parties):
            issues.append(_error("extra_name", f'"{name}" is named but is not a party in the '
                                 "fact sheet"))
    sheet_dates = {d.date.isoformat() for d in sheet.dates}
    for value in report.dates:
        if value not in sheet_dates:
            issues.append(_error("extra_date", f"auditor found a date not in the fact sheet: {value}"))
    sheet_amounts = {(a.amount, a.currency) for a in sheet.amounts}
    for amount in report.amounts:
        if (amount.amount, amount.currency) not in sheet_amounts:
            issues.append(_error("extra_amount", "auditor found a money amount not in the fact "
                                 f"sheet: {amount.amount} {amount.currency} (prices in tables "
                                 "count too, even without a currency symbol)"))
    return issues


async def verify(
    sheet: FactSheet, text: str, llm: LLMClient, settings: ModelCallSettings
) -> Verification:
    issues, found = check_deterministic(sheet, text)
    if issues:  # no point paying for an audit of a document that's already wrong
        return Verification(passed=False, issues=issues, found=found)
    report = await audit(text, llm, settings)
    issues = compare_audit(sheet, report)
    return Verification(
        passed=not any(i.severity == "error" for i in issues),
        issues=issues, found=found, audit=report,
    )


def _error(check: str, message: str) -> Issue:
    return Issue(severity="error", check=check, message=message)

