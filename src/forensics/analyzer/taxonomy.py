"""The five failure types, and the Finding every check reports."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class FailureType(StrEnum):
    HALLUCINATION = "hallucination"
    OMISSION = "omission"
    WRONG_VALUE = "wrong_value"
    MISCLASSIFICATION = "misclassification"
    FORMAT_ERROR = "format_error"


DEFINITIONS: dict[FailureType, str] = {
    FailureType.HALLUCINATION: (
        "the output states something its input does not support: an invented name, date, "
        "amount or claim"
    ),
    FailureType.OMISSION: "the output leaves out something its input clearly contains",
    FailureType.WRONG_VALUE: (
        "the right fact, wrongly carried over: a date converted wrongly, an amount or "
        "currency changed, a value attached to the wrong label"
    ),
    FailureType.MISCLASSIFICATION: "the document was given the wrong type",
    FailureType.FORMAT_ERROR: (
        "the output is malformed: unusable or cut-off output, a date not in YYYY-MM-DD, "
        "a currency that is not an ISO code"
    ),
}

# When one step shows several kinds of failure, the primary one is the most upstream
# cause. A cut-off reply (format) explains missing facts; a changed value explains why
# the true value is "missing".
PRIORITY: tuple[FailureType, ...] = (
    FailureType.FORMAT_ERROR,
    FailureType.HALLUCINATION,
    FailureType.WRONG_VALUE,
    FailureType.MISCLASSIFICATION,
    FailureType.OMISSION,
)


class Finding(BaseModel):
    """One concrete problem a deterministic check found, in plain English."""

    step: str
    check: str
    failure_type: FailureType
    message: str
    # The offending value, so the evidence chain can follow it downstream.
    value: str | None = None
    value_kind: Literal["date", "amount", "name"] | None = None


def primary_type(findings: list[Finding]) -> FailureType | None:
    present = {f.failure_type for f in findings}
    return next((t for t in PRIORITY if t in present), None)
