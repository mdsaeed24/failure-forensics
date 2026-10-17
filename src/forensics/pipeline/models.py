"""Input and output models for each pipeline step.

These schemas check *shape*, not *correctness*. A date like "2026-02-30" or a
currency like "Pounds" is allowed through on purpose: catching those is the
analyzer's job (checks/dates.py, checks/currency.py). If the schema rejected
them, a wrong answer would turn into a retry or a crash and never show up in a
trace as the failure it really is.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Confidence = Annotated[int, Field(ge=1, le=5, description="1 = guessing, 5 = certain")]


# --- Step 1: intake (no LLM) -------------------------------------------------


class RawDocument(BaseModel):
    doc_id: str
    text: str
    source_path: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> RawDocument:
        return cls(doc_id=path.stem, text=path.read_text(encoding="utf-8"), source_path=str(path))


class IntakeOutput(BaseModel):
    doc_id: str
    text: str  # normalized; every later step and every check uses this exact text
    title: str | None
    char_count: int
    word_count: int
    sha256: str


# --- Step 2: extraction -------------------------------------------------------


class Party(BaseModel):
    name: str
    role: str  # e.g. "seller", "buyer", "contact"


class DateFact(BaseModel):
    date: str  # should be YYYY-MM-DD; deliberately not validated here
    label: str  # e.g. "invoice date", "due date"
    quote: str  # the exact text in the document this came from


class AmountFact(BaseModel):
    amount: Decimal
    currency: str  # should be an ISO 4217 code; deliberately not validated here
    label: str  # e.g. "total due", "subtotal"
    quote: str


class ExtractionOutput(BaseModel):
    parties: list[Party] = []
    dates: list[DateFact] = []
    amounts: list[AmountFact] = []
    confidence: Confidence


# --- Step 3: classification ---------------------------------------------------

DocumentType = Literal[
    "invoice", "receipt", "purchase_order", "quotation", "contract", "payment_reminder", "other"
]

# Shown to the model in the prompt, so the labels and their meanings can't drift apart.
DOCUMENT_TYPE_DEFINITIONS: dict[str, str] = {
    "invoice": "a bill asking for payment for goods or services already supplied",
    "receipt": "confirms that a payment has already been made",
    "purchase_order": "a buyer's order to a seller, issued before delivery or billing",
    "quotation": "a seller's offer of prices for possible future work; nothing is owed yet",
    "contract": "an agreement setting out terms between parties, usually signed",
    "payment_reminder": "a follow-up about an invoice that is overdue or about to be",
    "other": "none of the above",
}


class ClassificationOutput(BaseModel):
    doc_type: DocumentType
    reason: str
    confidence: Confidence


# --- Step 4: summarization ----------------------------------------------------


class SummarizationInput(BaseModel):
    """The summarizer sees only upstream outputs, never the raw document, so every
    fact in the final summary can be traced back to the step that produced it."""

    extraction: ExtractionOutput
    classification: ClassificationOutput


class SummaryOutput(BaseModel):
    summary: str
    confidence: Confidence
