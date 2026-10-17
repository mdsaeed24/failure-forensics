"""Are the extracted currencies ISO 4217 codes?

Whether the code matches the document ("$" written, "EUR" extracted) is a grounding
question: see grounding.py.
"""

from __future__ import annotations

import re

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.pipeline.models import ExtractionOutput

ISO_CODE = re.compile(r"[A-Z]{3}")


def check_currency_codes(output: ExtractionOutput) -> list[Finding]:
    return [
        Finding(
            step="extraction", check="bad_currency_code", failure_type=FailureType.FORMAT_ERROR,
            message=f'currency "{a.currency}" for {a.amount} ({a.label}) is not a 3-letter ISO code',
            value=f"{a.amount} {a.currency}", value_kind="amount",
        )
        for a in output.amounts
        if not ISO_CODE.fullmatch(a.currency)
    ]
