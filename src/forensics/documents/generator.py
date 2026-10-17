"""Fact sheet → document, verified, rewritten with feedback until it passes."""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from forensics.config import GenerationSettings, ModelCallSettings
from forensics.documents.fact_sheets import FactSheet
from forensics.documents.verifier import Verification, verify
from forensics.llm.client import LLMCall, LLMClient, LLMError, record_calls
from forensics.pipeline.models import DOCUMENT_TYPE_DEFINITIONS
from forensics.textscan import format_amount, format_date

log = logging.getLogger(__name__)


class GeneratedDocument(BaseModel):
    document: str


class GenerationOutcome(BaseModel):
    doc_id: str
    status: Literal["verified", "failed", "error"]  # error: an API call failed, nothing to review
    document: str | None = None  # the last attempt, even if it failed verification
    attempts: int = 0
    rejected_attempts: list[list[str]] = []  # the errors that sent each earlier attempt back
    verification: Verification | None = None
    error: str | None = None
    llm_calls: list[LLMCall] = []

    @property
    def cost_usd(self) -> float | None:
        costs = [c.cost_usd for c in self.llm_calls]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]


_SYSTEM = """\
You write realistic business documents in Markdown for a test dataset whose answer
key is fixed in advance. Reply with a single JSON object and nothing else:
{"document": "<the full document in Markdown>"}

Hard rules. The document is rejected if it breaks any of them:
1. Include every fact you are given, each at least once.
2. Write party names exactly as given, character for character, including capitalization,
   even in headers and logos.
3. Write every date with the month name, e.g. "3 March 2026" or "March 3, 2026".
   Never use numeric dates like 03/03/2026.
4. Write every money amount with its currency symbol or ISO code next to it, e.g.
   "£1,028.40" or "GBP 1,028.40".
5. Do NOT mention any other person, organization, date or money amount. That rules out
   line-item prices (including bare numbers in a price column of a table), subtotals or
   taxes not in the list, brand names, other company names (banks, card networks,
   couriers, software vendors), signatures by unlisted people, email addresses and
   websites, and dates such as "today" or a period like "March 2026" written as a date.
6. Things that are not facts are fine and make it realistic: reference numbers,
   quantities, product descriptions, street addresses without names, percentages,
   payment terms stated in days.
7. Never use placeholders such as [address] or [date]. Leave a detail out or write a
   realistic non-fact one instead; a signature block may leave its date line blank
   ("Date: ____________").
8. Make the document's purpose unmistakable: a reader should know at a glance which
   type it is. Label each fact clearly (e.g. "Due date:").
9. Between 120 and 350 words."""


def render_facts(sheet: FactSheet) -> str:
    lines = [f"Document type: {sheet.doc_type} ({DOCUMENT_TYPE_DEFINITIONS[sheet.doc_type]})"]
    if sheet.style:
        lines.append(f"Style: {sheet.style}")
    lines.append("\nParties:")
    lines += [f'- "{p.name}" ({p.role})' for p in sheet.parties]
    if sheet.dates:
        lines.append("\nDates:")
        lines += [f"- {d.label}: {format_date(d.date)}" for d in sheet.dates]
    if sheet.amounts:
        lines.append("\nMoney amounts:")
        lines += [f"- {a.label}: {format_amount(a.amount, a.currency)}" for a in sheet.amounts]
    return "\n".join(lines)


def _user_message(sheet: FactSheet, previous: str | None, problems: list[str]) -> str:
    message = f"Write this document.\n\n{render_facts(sheet)}"
    if previous is not None:
        message += (
            "\n\nYour previous attempt was rejected:\n<previous>\n" + previous + "\n</previous>\n"
            "Problems:\n" + "\n".join(f"- {p}" for p in problems)
            + "\n\nWrite the full corrected document."
        )
    elif problems:  # e.g. a human reviewer's rejection note, before any new attempt
        message += "\n\nA reviewer rejected an earlier version of this document:\n" + "\n".join(
            f"- {p}" for p in problems
        )
    return message


async def _write(sheet: FactSheet, llm: LLMClient, settings: ModelCallSettings,
                 previous: str | None, problems: list[str]) -> str:
    result = await llm.complete_json(
        system=_SYSTEM,
        user=_user_message(sheet, previous, problems),
        schema=GeneratedDocument,
        model=settings.model,
        max_tokens=settings.max_output_tokens,
        temperature=settings.temperature,
        thinking=settings.thinking,
    )
    return result.parsed.document.strip() + "\n"


async def generate(
    sheet: FactSheet,
    llm: LLMClient,
    settings: GenerationSettings,
    reviewer_note: str | None = None,
) -> GenerationOutcome:
    outcome = GenerationOutcome(doc_id=sheet.doc_id, status="failed")
    previous: str | None = None
    problems = [reviewer_note] if reviewer_note else []
    with record_calls() as calls:
        try:
            for attempt in range(1, settings.max_attempts + 1):
                outcome.attempts = attempt
                text = await _write(sheet, llm, settings.generator, previous, problems)
                verification = await verify(sheet, text, llm, settings.verifier)
                outcome.document, outcome.verification = text, verification
                if verification.passed:
                    outcome.status = "verified"
                    break
                log.info("%s attempt %d failed: %d error(s)", sheet.doc_id, attempt,
                         len(verification.errors))
                previous, problems = text, [i.message for i in verification.errors]
                if attempt < settings.max_attempts:  # the last attempt's errors stay in .verification
                    outcome.rejected_attempts.append(problems)
        except LLMError as exc:
            outcome.status, outcome.error = "error", f"{type(exc).__name__}: {exc}"
        outcome.llm_calls = list(calls)
    return outcome
