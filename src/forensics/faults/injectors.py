"""One class per fault, each knowing its correct answer: the step it breaks and how.

Two mechanisms, both plugging into the runner without touching step code:
- hook faults edit one step's output before the next step receives it
  (deterministic, so they always "apply", though not always "take"),
- prompt faults swap in a weakened prompt variant (realistic: the model may or
  may not actually fail on a given document).

The trace records neither. The label lives only in the evaluation database.
"""

from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar

from pydantic import BaseModel

from forensics.analyzer.taxonomy import FailureType
from forensics.pipeline.models import (
    AmountFact,
    ClassificationOutput,
    ExtractionOutput,
    IntakeOutput,
    Party,
    SummaryOutput,
)
from forensics.pipeline.runner import PipelineOptions
from forensics.textscan import CURRENCY_SYMBOLS, find_money, format_amount, parse_iso_date


class Fault:
    name: ClassVar[str]
    step: ClassVar[str]  # the step this fault breaks: the correct root cause
    failure_type: ClassVar[FailureType]
    description: ClassVar[str]
    prompt_variant: ClassVar[str | None] = None  # set for prompt faults

    def applicable(self, doc_text: str, truth: dict[str, Any]) -> bool:
        """Whether this fault can do anything to this document at all."""
        return True

    def mutate(self, output: Any, injection: Injection) -> Any | None:
        """Hook faults: the changed output, or None if there was nothing to change."""
        return None

    def inject(self, seed: int) -> Injection:
        return Injection(fault=self, rng=random.Random(seed))


@dataclass
class Injection:
    """One fault applied to one run. Records whether it changed anything, and what."""

    fault: Fault
    rng: random.Random
    applied: bool = False
    detail: str = ""  # what was changed; for the evaluation report only

    def hook(self, step: str, output: BaseModel) -> BaseModel:
        if step != self.fault.step or self.fault.prompt_variant:
            return output
        changed = self.fault.mutate(output, self)
        if changed is None:
            return output
        self.applied = True
        return changed

    def options(self, batch_id: str | None = None) -> PipelineOptions:
        if self.fault.prompt_variant:
            self.applied, self.detail = True, f"prompt variant {self.fault.prompt_variant!r}"
            return PipelineOptions(prompt_variants={self.fault.step: self.fault.prompt_variant},
                                   batch_id=batch_id)
        return PipelineOptions(hooks=[self.hook], batch_id=batch_id)


# --- intake ---------------------------------------------------------------------

class TruncateDocument(Fault):
    name = "truncate_document"
    step = "intake"
    failure_type = FailureType.OMISSION
    description = "intake hands on only the first ~45% of the document"

    def mutate(self, output: IntakeOutput, injection: Injection) -> IntakeOutput:
        cut = output.text.rfind("\n", 0, int(len(output.text) * 0.45))
        text = output.text[: cut if cut > 0 else int(len(output.text) * 0.45)].rstrip()
        injection.detail = f"kept {len(text)} of {len(output.text)} characters"
        return output.model_copy(update={"text": text, "char_count": len(text),
                                         "word_count": len(text.split())})


# --- extraction -----------------------------------------------------------------

def _has_amounts(truth: dict[str, Any]) -> bool:
    return bool(truth["amounts"])


def _has_dates(truth: dict[str, Any]) -> bool:
    return bool(truth["dates"])


class DropAmount(Fault):
    name = "drop_amount"
    step = "extraction"
    failure_type = FailureType.OMISSION
    description = "extraction loses the largest amount (usually the total)"

    def applicable(self, doc_text, truth):
        return _has_amounts(truth)

    def mutate(self, output: ExtractionOutput, injection):
        if not output.amounts:
            return None
        biggest = max(a.amount for a in output.amounts)
        injection.detail = f"dropped {biggest}"
        return output.model_copy(update={"amounts": [a for a in output.amounts if a.amount != biggest]})


class TransposeAmount(Fault):
    name = "transpose_amount"
    step = "extraction"
    failure_type = FailureType.WRONG_VALUE
    description = "two adjacent digits of an amount swapped (12,980 -> 12,890); quote untouched"

    def applicable(self, doc_text, truth):
        return any(_transpose(Decimal(a["amount"]), random.Random(0)) for a in truth["amounts"])

    def mutate(self, output: ExtractionOutput, injection):
        candidates = sorted({a.amount for a in output.amounts if _transpose(a.amount, injection.rng)})
        if not candidates:
            return None
        old = injection.rng.choice(candidates)
        new = _transpose(old, injection.rng)
        injection.detail = f"{old} -> {new}"
        return output.model_copy(update={"amounts": [
            a.model_copy(update={"amount": new}) if a.amount == old else a for a in output.amounts
        ]})


def _transpose(amount: Decimal, rng: random.Random) -> Decimal | None:
    whole, _, cents = f"{amount:.2f}".partition(".")
    spots = [i for i in range(len(whole) - 1) if whole[i] != whole[i + 1] and not (i == 0 and whole[1] == "0")]
    if not spots:
        return None
    i = rng.choice(spots)
    swapped = whole[:i] + whole[i + 1] + whole[i] + whole[i + 2:]
    return Decimal(f"{swapped}.{cents}")


class SwapCurrency(Fault):
    name = "swap_currency"
    step = "extraction"
    failure_type = FailureType.WRONG_VALUE
    description = "every amount gets the wrong currency code (GBP -> EUR, others -> GBP)"

    def applicable(self, doc_text, truth):
        return _has_amounts(truth)

    def mutate(self, output: ExtractionOutput, injection):
        if not output.amounts:
            return None
        swap = lambda c: "EUR" if c == "GBP" else "GBP"  # noqa: E731
        injection.detail = ", ".join(sorted({f"{a.currency} -> {swap(a.currency)}" for a in output.amounts}))
        return output.model_copy(update={"amounts": [
            a.model_copy(update={"currency": swap(a.currency)}) for a in output.amounts
        ]})


class ShiftDate(Fault):
    name = "shift_date"
    step = "extraction"
    failure_type = FailureType.WRONG_VALUE
    description = "one date moved by a day or two; quote untouched"

    def applicable(self, doc_text, truth):
        return _has_dates(truth)

    def mutate(self, output: ExtractionOutput, injection):
        valid = [(i, v) for i, d in enumerate(output.dates) if (v := parse_iso_date(d.date))]
        if not valid:
            return None
        taken = {v for _, v in valid}
        i, value = injection.rng.choice(valid)
        new = next(value + dt.timedelta(days=s) for s in (1, -1, 2, -2, 3)
                   if value + dt.timedelta(days=s) not in taken)
        injection.detail = f"{value} -> {new}"
        dates = list(output.dates)
        dates[i] = dates[i].model_copy(update={"date": new.isoformat()})
        return output.model_copy(update={"dates": dates})


class InventParty(Fault):
    name = "invent_party"
    step = "extraction"
    failure_type = FailureType.HALLUCINATION
    description = "a party that appears nowhere in the document is added"
    NAMES: ClassVar[tuple[str, ...]] = (
        "Northbridge Consulting Group", "Samuel Achterberg", "Vantage Point Logistics",
        "Marguerite Ellwood", "Oakfield Capital Partners",
    )

    def mutate(self, output: ExtractionOutput, injection):
        name = injection.rng.choice(self.NAMES)
        injection.detail = f"added {name}"
        return output.model_copy(update={"parties": [*output.parties, Party(name=name, role="contact")]})


class InventAmount(Fault):
    name = "invent_amount"
    step = "extraction"
    failure_type = FailureType.HALLUCINATION
    description = "an amount that appears nowhere in the document is added, with a made-up quote"

    def mutate(self, output: ExtractionOutput, injection):
        currency = output.amounts[0].currency if output.amounts else "USD"
        if currency not in CURRENCY_SYMBOLS:
            currency = "USD"
        existing = {a.amount for a in output.amounts}
        amount = next(Decimal(v) for v in ("250.00", "375.00", "180.00", "95.00") if Decimal(v) not in existing)
        if currency in ("JPY", "KRW"):
            amount *= 100
        quote = format_amount(amount, currency)
        injection.detail = f"added {quote}"
        fee = AmountFact(amount=amount, currency=currency, label="handling fee", quote=quote)
        return output.model_copy(update={"amounts": [*output.amounts, fee]})


class NumericDates(Fault):
    name = "numeric_dates"
    step = "extraction"
    failure_type = FailureType.FORMAT_ERROR
    description = "dates come out as DD/MM/YYYY instead of YYYY-MM-DD"

    def applicable(self, doc_text, truth):
        return _has_dates(truth)

    def mutate(self, output: ExtractionOutput, injection):
        dates = [d.model_copy(update={"date": f"{v:%d/%m/%Y}"}) if (v := parse_iso_date(d.date)) else d
                 for d in output.dates]
        if dates == output.dates:
            return None
        injection.detail = f"{len(dates)} date(s) reformatted"
        return output.model_copy(update={"dates": dates})


class SymbolCurrency(Fault):
    name = "symbol_currency"
    step = "extraction"
    failure_type = FailureType.FORMAT_ERROR
    description = 'currencies come out as symbols or words ("£", "dollars") instead of ISO codes'
    WORDS: ClassVar[dict[str, str]] = {"USD": "dollars", "EUR": "€", "GBP": "£", "JPY": "yen", "INR": "rupees"}

    def applicable(self, doc_text, truth):
        return _has_amounts(truth)

    def mutate(self, output: ExtractionOutput, injection):
        if not output.amounts:
            return None
        to = lambda c: self.WORDS.get(c, c.lower())  # noqa: E731
        injection.detail = ", ".join(sorted({f"{a.currency} -> {to(a.currency)}" for a in output.amounts}))
        return output.model_copy(update={"amounts": [
            a.model_copy(update={"currency": to(a.currency)}) for a in output.amounts
        ]})


class MainFactsOnlyPrompt(Fault):
    name = "prompt_main_facts_only"
    step = "extraction"
    failure_type = FailureType.OMISSION
    description = 'extraction prompt "simplified" to keep only the main facts'
    prompt_variant = "main_facts_only"

    def applicable(self, doc_text, truth):  # needs something secondary to skip
        return len(truth["dates"]) + len(truth["amounts"]) + len(truth["parties"]) >= 5


# --- classification -------------------------------------------------------------

CONFUSABLE: dict[str, str] = {
    "invoice": "quotation", "receipt": "invoice", "purchase_order": "invoice",
    "quotation": "invoice", "contract": "purchase_order", "payment_reminder": "invoice",
    "other": "invoice",
}


class ConfusableLabel(Fault):
    name = "confusable_label"
    step = "classification"
    failure_type = FailureType.MISCLASSIFICATION
    description = "the label is swapped for the most easily confused type, with a plausible reason"

    def mutate(self, output: ClassificationOutput, injection):
        label = CONFUSABLE[output.doc_type]
        injection.detail = f"{output.doc_type} -> {label}"
        reason = f"It reads as a {label.replace('_', ' ')}, given its heading and the amounts it lists."
        return output.model_copy(update={"doc_type": label, "reason": reason})


_TYPE_WORDS = {
    "invoice": r"invoice", "receipt": r"receipt", "purchase_order": r"purchase order|\bpo\b",
    "quotation": r"quot|estimate|proposal", "contract": r"agreement|contract",
    "payment_reminder": r"reminder|overdue|notice",
}


class TitleOnlyPrompt(Fault):
    name = "prompt_title_only"
    step = "classification"
    failure_type = FailureType.MISCLASSIFICATION
    description = "classification prompt changed to go by the title or first heading"
    prompt_variant = "title_only"

    def applicable(self, doc_text, truth):
        """Only where the title names a DIFFERENT type ("INVOICE" on a receipt): a title
        that names no type, or the right one, can't mislead a title-only classifier."""
        title = _title(doc_text).lower()
        return any(re.search(pattern, title) for t, pattern in _TYPE_WORDS.items()
                   if t != truth["doc_type"])


def _title(doc_text: str) -> str:
    lines = [line.strip() for line in doc_text.splitlines() if line.strip()]
    heading = next((line for line in lines if line.startswith("#")), None)
    return (heading or (lines[0] if lines else "")).lstrip("#").strip()


# --- summarization --------------------------------------------------------------

class SummaryInventsFee(Fault):
    name = "summary_invents_fee"
    step = "summarization"
    failure_type = FailureType.HALLUCINATION
    description = "the summary gains a late fee that isn't among its facts"

    def mutate(self, output: SummaryOutput, injection):
        said = find_money(output.summary)
        currency = said[0].currency if said else "USD"
        amount = Decimal("45.00") * (100 if currency in ("JPY", "KRW") else 1)
        sentence = f"A late payment fee of {format_amount(amount, currency)} also applies."
        injection.detail = sentence
        return output.model_copy(update={"summary": f"{output.summary.rstrip()} {sentence}"})


class SummaryWrongAmount(Fault):
    name = "summary_wrong_amount"
    step = "summarization"
    failure_type = FailureType.WRONG_VALUE
    description = "an amount in the summary has two digits transposed"

    def mutate(self, output: SummaryOutput, injection):
        for m in find_money(output.summary):
            new = _transpose(m.amount, injection.rng)
            if new is None:
                continue
            surface = format_amount(new, m.currency)
            injection.detail = f'"{m.surface}" -> "{surface}"'
            text = output.summary[: m.start] + surface + output.summary[m.start + len(m.surface):]
            return output.model_copy(update={"summary": text})
        return None


class SummaryDropsAmounts(Fault):
    name = "summary_drops_amounts"
    step = "summarization"
    failure_type = FailureType.OMISSION
    description = "every sentence of the summary that mentions money is removed"

    def applicable(self, doc_text, truth):
        return _has_amounts(truth)

    def mutate(self, output: SummaryOutput, injection):
        sentences = re.split(r"(?<=[.!?])\s+", output.summary.strip())
        kept = [s for s in sentences if not find_money(s)]
        if len(kept) == len(sentences) or not kept:
            return None
        injection.detail = f"removed {len(sentences) - len(kept)} of {len(sentences)} sentences"
        return output.model_copy(update={"summary": " ".join(kept)})


class EmbellishPrompt(Fault):
    name = "prompt_embellish"
    step = "summarization"
    failure_type = FailureType.HALLUCINATION
    description = 'summary prompt changed to "add helpful context such as usual terms or fees"'
    prompt_variant = "embellish"
