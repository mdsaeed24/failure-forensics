"""Step 4: summarize the document from the extracted facts and its type."""

from __future__ import annotations

import json

from forensics.llm.client import LLMClient
from forensics.pipeline.models import SummarizationInput, SummaryOutput
from forensics.pipeline.prompts import get_prompt


def render_facts(inp: SummarizationInput) -> str:
    """How the extraction is shown to the summarizer. Kept separate because fault
    injection will vary this hand-off format."""
    facts = inp.extraction.model_dump(mode="json", exclude={"confidence"})
    return json.dumps(facts, indent=2, ensure_ascii=False)


async def run(inp: SummarizationInput, llm: LLMClient, variant: str = "default") -> SummaryOutput:
    prompt = get_prompt("summarization", variant)
    result = await llm.complete_json(
        system=prompt.system,
        user=prompt.render_user(doc_type=inp.classification.doc_type, facts=render_facts(inp)),
        schema=SummaryOutput,
    )
    return result.parsed
