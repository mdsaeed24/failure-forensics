"""Did intake hand on the document text intact (only normalized)?"""

from __future__ import annotations

from forensics.analyzer.taxonomy import FailureType, Finding
from forensics.pipeline.intake import normalize
from forensics.pipeline.models import IntakeOutput, RawDocument


def check_intake(raw: RawDocument, output: IntakeOutput) -> list[Finding]:
    expected = normalize(raw.text)
    if output.text == expected:
        return []
    if len(output.text) < len(expected) and expected.startswith(output.text):
        dropped = len(expected) - len(output.text)
        return [Finding(
            step="intake", check="intake_dropped_text", failure_type=FailureType.OMISSION,
            message=f"intake cut off the last {dropped} of {len(expected)} characters; "
                    f'the text handed on ends at "…{output.text[-40:]}"',
        )]
    return [Finding(
        step="intake", check="intake_altered_text", failure_type=FailureType.WRONG_VALUE,
        message="intake changed the document text beyond normalizing it",
    )]
