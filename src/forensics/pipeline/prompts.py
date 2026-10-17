"""Prompt templates, by step and named variant.

Each step uses the "default" variant unless the runner is told otherwise. Fault
injection (Phase 5) adds deliberately weakened variants here and selects them
through PipelineOptions.prompt_variants, so step code never changes.

User templates use string.Template ($name placeholders). Values are inserted
verbatim, so braces or dollar signs inside a document are safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from string import Template

from forensics.pipeline.models import DOCUMENT_TYPE_DEFINITIONS


@dataclass(frozen=True)
class PromptTemplate:
    system: str
    user: str

    def render_user(self, **values: str) -> str:
        return Template(self.user).substitute(values)


_EXTRACTION_SYSTEM = """\
You extract facts from business documents. Reply with a single JSON object and nothing else.

The JSON object must have exactly this shape:
{
  "parties": [{"name": "<name as written>", "role": "<e.g. seller, buyer, contact>"}],
  "dates": [{"date": "YYYY-MM-DD", "label": "<what the date is>", "quote": "<exact text>"}],
  "amounts": [{"amount": 1234.50, "currency": "<ISO 4217 code>", "label": "<what the amount is>", "quote": "<exact text>"}],
  "confidence": <integer 1-5>
}

Rules:
- Include only facts stated in the document. Never guess or infer missing values.
- "parties" are the people and organizations named in the document.
- "quote" must be copied exactly, character for character, from the document.
- "date" is the date converted to YYYY-MM-DD.
- "amount" is a plain number: no currency symbols, no thousands separators.
- "currency" is a 3-letter ISO 4217 code such as USD, EUR or GBP.
- "confidence": 5 if every fact is clearly stated, lower if the document is ambiguous."""

_EXTRACTION_USER = """\
<document>
$document
</document>"""


def _doc_type_lines() -> str:
    return "\n".join(f'- "{name}": {meaning}' for name, meaning in DOCUMENT_TYPE_DEFINITIONS.items())


_CLASSIFICATION_SYSTEM = f"""\
You classify business documents. Reply with a single JSON object and nothing else.

The JSON object must have exactly this shape:
{{"doc_type": "<one label from the list>", "reason": "<one sentence>", "confidence": <integer 1-5>}}

Labels:
{_doc_type_lines()}

Choose the label that matches the document's purpose, not just its title.
"confidence": 5 if the type is unmistakable, lower if two labels could fit."""

_CLASSIFICATION_USER = _EXTRACTION_USER

_SUMMARIZATION_SYSTEM = """\
You write short summaries of business documents from facts that were already
extracted from them. Reply with a single JSON object and nothing else.

The JSON object must have exactly this shape:
{"summary": "<2-4 sentences>", "confidence": <integer 1-5>}

Rules:
- Use only the facts given. Do not add any name, date or amount that is not listed.
- Say what kind of document it is, who it is between, and the key dates and amounts.
- "confidence": 5 if the facts are enough for a clear summary, lower if they are thin."""

_SUMMARIZATION_USER = """\
Document type: $doc_type

Extracted facts (JSON):
$facts"""


# --- Weakened variants, used only by fault injection (Phase 5) -----------------
# Each is a plausible prompt regression, not a sabotage: the kind of edit someone
# might ship to "simplify" a prompt. Whether it actually breaks a given document is
# up to the model, which is what makes these faults realistic.

_EXTRACTION_MAIN_FACTS_ONLY = _EXTRACTION_SYSTEM.replace(
    "- Include only facts stated in the document. Never guess or infer missing values.",
    "- Keep it short: include only the main parties, the single most important date and\n"
    "  the single most important amount. Skip anything secondary.",
)
_CLASSIFICATION_TITLE_ONLY = _CLASSIFICATION_SYSTEM.replace(
    "Choose the label that matches the document's purpose, not just its title.",
    "To be fast and consistent, classify by the document's title or first heading.",
)
_SUMMARIZATION_EMBELLISH = _SUMMARIZATION_SYSTEM.replace(
    "- Use only the facts given. Do not add any name, date or amount that is not listed.",
    "- Make the summary genuinely useful: where helpful, add the practical context a reader\n"
    "  would expect, such as usual payment terms, typical fees or next steps.",
)
assert _EXTRACTION_MAIN_FACTS_ONLY != _EXTRACTION_SYSTEM
assert _CLASSIFICATION_TITLE_ONLY != _CLASSIFICATION_SYSTEM
assert _SUMMARIZATION_EMBELLISH != _SUMMARIZATION_SYSTEM

PROMPTS: dict[str, dict[str, PromptTemplate]] = {
    "extraction": {
        "default": PromptTemplate(_EXTRACTION_SYSTEM, _EXTRACTION_USER),
        "main_facts_only": PromptTemplate(_EXTRACTION_MAIN_FACTS_ONLY, _EXTRACTION_USER),
    },
    "classification": {
        "default": PromptTemplate(_CLASSIFICATION_SYSTEM, _CLASSIFICATION_USER),
        "title_only": PromptTemplate(_CLASSIFICATION_TITLE_ONLY, _CLASSIFICATION_USER),
    },
    "summarization": {
        "default": PromptTemplate(_SUMMARIZATION_SYSTEM, _SUMMARIZATION_USER),
        "embellish": PromptTemplate(_SUMMARIZATION_EMBELLISH, _SUMMARIZATION_USER),
    },
}


def get_prompt(step: str, variant: str = "default") -> PromptTemplate:
    try:
        return PROMPTS[step][variant]
    except KeyError:
        known = {s: sorted(v) for s, v in PROMPTS.items()}
        raise KeyError(f"No prompt variant {variant!r} for step {step!r}. Known: {known}") from None
