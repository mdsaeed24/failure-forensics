"""Do the extracted dates parse as YYYY-MM-DD?"""

from __future__ import annotations

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.pipeline.models import ExtractionOutput
from forensics.textscan import parse_iso_date


def check_date_formats(output: ExtractionOutput) -> list[Finding]:
    return [
        Finding(
            step="extraction", check="bad_date_format", failure_type=FailureType.FORMAT_ERROR,
            message=f'"{d.date}" ({d.label}) is not a valid YYYY-MM-DD date',
            value=d.date, value_kind="date",
        )
        for d in output.dates
        if parse_iso_date(d.date) is None
    ]
