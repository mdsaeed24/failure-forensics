"""Step 2: pull parties, dates and amounts out of the document text."""

from __future__ import annotations

from forensics.llm.client import LLMClient
from forensics.pipeline.models import ExtractionOutput, IntakeOutput
from forensics.pipeline.prompts import get_prompt


async def run(inp: IntakeOutput, llm: LLMClient, variant: str = "default") -> ExtractionOutput:
    prompt = get_prompt("extraction", variant)
    result = await llm.complete_json(
        system=prompt.system,
        user=prompt.render_user(document=inp.text),
        schema=ExtractionOutput,
    )
    return result.parsed
