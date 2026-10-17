"""Step 3: decide what kind of document this is."""

from __future__ import annotations

from forensics.llm.client import LLMClient
from forensics.pipeline.models import ClassificationOutput, IntakeOutput
from forensics.pipeline.prompts import get_prompt


async def run(inp: IntakeOutput, llm: LLMClient, variant: str = "default") -> ClassificationOutput:
    prompt = get_prompt("classification", variant)
    result = await llm.complete_json(
        system=prompt.system,
        user=prompt.render_user(document=inp.text),
        schema=ClassificationOutput,
    )
    return result.parsed
