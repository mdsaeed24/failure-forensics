"""Fact sheets: the facts of each test document, written by a human before the document exists.

Unlike the pipeline's output models, these validate strictly: a fact sheet *is*
the answer key, so a bad date or an unknown currency here is an error, not data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from forensics.pipeline.models import DocumentType
from forensics.textscan import SUPPORTED_CURRENCIES, ZERO_DECIMAL_CURRENCIES


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo'd key is an error, not silently ignored


class SheetParty(_Strict):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)


class SheetDate(_Strict):
    date: dt.date
    label: str = Field(min_length=1)


class SheetAmount(_Strict):
    amount: Decimal = Field(gt=0)
    currency: str
    label: str = Field(min_length=1)

    @field_validator("currency")
    @classmethod
    def _known_currency(cls, value: str) -> str:
        if value not in SUPPORTED_CURRENCIES:
            raise ValueError(
                f"{value!r} is not a supported ISO 4217 code; use one of "
                f"{', '.join(sorted(SUPPORTED_CURRENCIES))}"
            )
        return value

    @field_validator("amount")
    @classmethod
    def _at_most_two_decimals(cls, value: Decimal) -> Decimal:
        if value != value.quantize(Decimal("0.01")):
            raise ValueError("amounts have at most 2 decimal places")
        return value

    @model_validator(mode="after")
    def _normalize_places(self) -> SheetAmount:
        """1028.4 and 1028.40 are the same fact; store it one way (1028.40, or 5000 for JPY)."""
        if self.currency in ZERO_DECIMAL_CURRENCIES:
            if self.amount != self.amount.to_integral():
                raise ValueError(f"{self.currency} amounts are whole numbers")
            self.amount = self.amount.quantize(Decimal("1"))
        else:
            self.amount = self.amount.quantize(Decimal("0.01"))
        return self


class FactSheet(_Strict):
    doc_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    doc_type: DocumentType
    style: str | None = None  # optional hint for the generator: tone, layout, setting
    parties: list[SheetParty] = Field(min_length=1)
    dates: list[SheetDate] = []
    amounts: list[SheetAmount] = []

    def sha256(self) -> str:
        """Changes whenever any fact or the style changes, which marks the document stale."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class FactSheetError(ValueError):
    pass


def load_fact_sheets(path: Path) -> list[FactSheet]:
    """Loads and validates every sheet, reporting all problems at once."""
    if not path.is_file():
        raise FactSheetError(f"{path} not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise FactSheetError(f"{path}: the top level must be a list of fact sheets")

    sheets, problems, seen = [], [], set()
    for index, item in enumerate(raw, start=1):
        doc_id = item.get("doc_id", "?") if isinstance(item, dict) else "?"
        try:
            sheet = FactSheet.model_validate(item)
        except ValidationError as exc:
            for error in exc.errors():
                where = ".".join(str(part) for part in error["loc"]) or "(sheet)"
                problems.append(f"sheet #{index} ({doc_id}): {where}: {error['msg']}")
            continue
        if sheet.doc_id in seen:
            problems.append(f"sheet #{index}: duplicate doc_id {sheet.doc_id!r}")
        seen.add(sheet.doc_id)
        sheets.append(sheet)
    if problems:
        raise FactSheetError(f"{path} has {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return sheets
