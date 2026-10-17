"""LLM judge: is this step's output a reasonable transformation of ITS OWN input?

The judge sees one step at a time: what the step is supposed to do, the input it
actually received, and what it produced. It does not see the ground truth, the
prompt the step was given (a weakened prompt is exactly what it should catch), or
the step's self-reported confidence (which would anchor it).

A step that faithfully processes a bad input did its job. That is what lets the
analyzer find the FIRST failing step instead of blaming everything downstream.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from forensics.analyzer.taxonomy import DEFINITIONS
from forensics.config import AnalysisSettings, Settings
from forensics.llm.client import LLMCall, LLMClient
from forensics.pipeline.models import DOCUMENT_TYPE_DEFINITIONS, SummarizationInput
from forensics.pipeline.summarization import render_facts


class JudgeVerdict(BaseModel):
    score: int = Field(ge=1, le=5)
    failure_type: Literal[
        "none", "hallucination", "omission", "wrong_value", "misclassification", "format_error"
    ]
    reason: str
    evidence: str = ""


_DOC_TYPES = "\n".join(f'  - "{k}": {v}' for k, v in DOCUMENT_TYPE_DEFINITIONS.items())

STEP_JOBS: dict[str, str] = {
    "extraction": (
        "Extract every party (person or organization) named in the document, as a name and "
        "a role; every calendar date, converted to YYYY-MM-DD; and every money amount, as a "
        "plain number plus its ISO currency code. Each date and amount also has a label "
        "saying what it is and a quote copied from the document (parties have no quote or "
        "label). Nothing may be invented; nothing stated may be left out."
    ),
    "classification": (
        "Give the document the one type label that matches its purpose (not merely its "
        f"title), from:\n{_DOC_TYPES}"
    ),
    "summarization": (
        "Write a 2-4 sentence summary using only the facts and document type it is given: "
        "what kind of document it is, who it is between, and the key dates and amounts. It "
        "must not add any name, date, amount or claim that is not in its input. It does not "
        "see the original document, so it is not responsible for facts its input lacks."
    ),
}

_SYSTEM = f"""\
You review ONE step of a document-processing pipeline. You are shown the step's job,
the exact input the step received, and the output it produced.

Judge only whether the output correctly and completely does the step's job FOR THIS
INPUT. Do not judge the input itself: if the input is wrong but the output faithfully
follows it, the step did its job and scores 5.

Score:
  5 = correct and complete
  4 = minor imperfections that change no fact or meaning
  3 = a noticeable problem, but you are unsure it matters
  2 = a clear error
  1 = badly wrong or unusable

Failure types (use "none" for scores 4-5):
""" + "\n".join(f'  - "{t.value}": {d}' for t, d in DEFINITIONS.items()) + """

Reply with a single JSON object and nothing else:
{"score": <1-5>, "failure_type": "<type or none>", "reason": "<one or two sentences>",
 "evidence": "<the exact words from the input or output that show the problem, or empty>"}"""


def render_input(step: str, step_input: dict[str, Any]) -> str:
    if step in ("extraction", "classification"):
        return f"<document>\n{step_input['text']}\n</document>"
    if step == "summarization":
        inp = SummarizationInput.model_validate(step_input)
        return f"Document type: {inp.classification.doc_type}\n\nExtracted facts (JSON):\n{render_facts(inp)}"
    raise ValueError(f"the judge does not review step {step!r}")


def render_output(step_output: dict[str, Any]) -> str:
    shown = {k: v for k, v in step_output.items() if k != "confidence"}
    return json.dumps(shown, indent=2, ensure_ascii=False)


async def judge_step(
    step: str, step_input: dict[str, Any], step_output: dict[str, Any],
    llm: LLMClient, settings: Settings, known_issue: str | None = None,
) -> tuple[JudgeVerdict, LLMCall]:
    """`known_issue`: regression only. A problem a human confirmed on an earlier run of
    the same document, which the judge then checks for specifically. The analyzer
    never passes one: there it must find failures unprompted."""
    analysis: AnalysisSettings = settings.analysis
    user = (
        f"STEP: {step}\n\nTHE STEP'S JOB:\n{STEP_JOBS[step]}\n\n"
        f"INPUT THIS STEP RECEIVED:\n{render_input(step, step_input)}\n\n"
        f"OUTPUT THIS STEP PRODUCED (JSON):\n{render_output(step_output)}"
    )
    if known_issue:
        user += (
            "\n\nA REVIEWER REPORTED THIS PROBLEM WITH AN EARLIER OUTPUT OF THIS STEP ON THE "
            f"SAME DOCUMENT:\n{known_issue}\nCheck specifically whether it is present in this "
            "output, as well as judging it as usual. If it is present, the score is at most 2."
        )
    result = await llm.complete_json(
        system=_SYSTEM,
        user=user,
        schema=JudgeVerdict,
        model=settings.llm.judge_model,
        max_tokens=analysis.judge_max_output_tokens,
        temperature=0.0,
        thinking=analysis.judge_thinking,
    )
    return result.parsed, result.call
