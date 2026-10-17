"""Deterministic checks: free, repeatable, no LLM. Each compares a step's output with
that step's OWN input, never with ground truth, so they work on production traces too.

    run_checks("extraction", step_input, step_output) -> [Finding, ...]
"""

from __future__ import annotations

from typing import Any

from forensics.analyzer.taxonomy import Finding
from forensics.checks.currency import check_currency_codes
from forensics.checks.dates import check_date_formats
from forensics.checks.grounding import check_grounding
from forensics.checks.intake import check_intake
from forensics.checks.summary_entities import check_summary_entities
from forensics.pipeline.models import (
    ExtractionOutput,
    IntakeOutput,
    RawDocument,
    SummarizationInput,
    SummaryOutput,
)


def run_checks(step: str, step_input: dict[str, Any], step_output: dict[str, Any]) -> list[Finding]:
    """Inputs and outputs as recorded in the trace (JSON dicts)."""
    if step == "intake":
        findings = check_intake(RawDocument.model_validate(step_input),
                                IntakeOutput.model_validate(step_output))
    elif step == "extraction":
        source = IntakeOutput.model_validate(step_input).text
        output = ExtractionOutput.model_validate(step_output)
        findings = (check_date_formats(output) + check_currency_codes(output)
                    + check_grounding(source, output))
    elif step == "summarization":
        findings = check_summary_entities(SummarizationInput.model_validate(step_input),
                                          SummaryOutput.model_validate(step_output))
    else:  # classification: whether a label fits is a judgment call, left to the judge
        findings = []
    return [f.model_copy(update={"step": step}) for f in findings]
