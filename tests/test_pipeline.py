from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from fakes import CLASSIFICATION, EXTRACTION, SUMMARY
from forensics.config import PROJECT_ROOT
from forensics.pipeline.models import (
    ClassificationOutput,
    ExtractionOutput,
    RawDocument,
    SummaryOutput,
)
from forensics.pipeline.runner import STEP_ORDER, PipelineOptions, run_pipeline

SAMPLE = PROJECT_ROOT / "data" / "samples" / "sample_invoice.md"


def run(llm, settings, options=None, doc=None):
    doc = doc or RawDocument.from_file(SAMPLE)
    return asyncio.run(run_pipeline(doc, llm, settings, options))


def test_full_run_produces_typed_outputs(make_llm, settings):
    llm, fake = make_llm(EXTRACTION, CLASSIFICATION, SUMMARY)
    result = run(llm, settings)

    assert result.status == "ok"
    assert [s.step for s in result.steps] == list(STEP_ORDER)
    assert isinstance(result.extraction, ExtractionOutput)
    assert isinstance(result.classification, ClassificationOutput)
    assert isinstance(result.summary, SummaryOutput)
    assert result.steps[0].llm_calls == []  # intake uses no LLM
    assert all(len(s.llm_calls) == 1 for s in result.steps[1:])
    assert result.total_tokens == 3 * 150
    assert result.total_cost_usd == pytest.approx(3 * (100 * 1.0 + 50 * 2.0) / 1e6)

    extraction_prompt = fake.requests[0]["messages"][1]["content"]
    assert "Harbor Lane Printing Ltd." in extraction_prompt


def test_summarizer_sees_only_upstream_outputs(make_llm, settings):
    llm, fake = make_llm(EXTRACTION, CLASSIFICATION, SUMMARY)
    run(llm, settings)
    summary_prompt = fake.requests[2]["messages"][1]["content"]
    assert "Document type: invoice" in summary_prompt
    assert "£1,028.40" in summary_prompt  # from the extraction quote
    assert "Design revisions" not in summary_prompt  # raw document text never reaches it


def test_hook_output_is_recorded_and_passed_downstream(make_llm, settings):
    def drop_amounts(step, output):
        return output.model_copy(update={"amounts": []}) if step == "extraction" else output

    llm, fake = make_llm(EXTRACTION, CLASSIFICATION, SUMMARY)
    result = run(llm, settings, PipelineOptions(hooks=[drop_amounts]))

    assert result.extraction.amounts == []
    facts = fake.requests[2]["messages"][1]["content"].split("(JSON):\n", 1)[1]
    assert json.loads(facts)["amounts"] == []


def test_failed_step_stops_the_run_and_keeps_earlier_outputs(make_llm, settings):
    llm, _ = make_llm(EXTRACTION, "not json", "still not json")
    result = run(llm, settings)

    assert result.status == "error"
    assert result.failed_step == "classification"
    assert "LLMOutputError" in result.error
    assert result.extraction is not None and result.summary is None
    failed = result.steps[-1]
    assert failed.status == "error" and failed.llm_calls[0].raw_output == "still not json"


def test_intake_failure_is_recorded_without_calling_the_llm(make_llm, settings):
    llm, fake = make_llm()
    result = run(llm, settings, doc=RawDocument(doc_id="blank", text="   "))
    assert result.failed_step == "intake"
    assert fake.requests == []


def test_prompt_variants_are_validated():
    with pytest.raises(ValueError, match="unknown steps"):
        PipelineOptions(prompt_variants={"intake": "x"})


def test_unknown_prompt_variant_is_a_bug_not_a_step_failure(make_llm, settings):
    llm, _ = make_llm()
    with pytest.raises(KeyError, match="No prompt variant"):
        run(llm, settings, PipelineOptions(prompt_variants={"extraction": "nope"}))


def test_sample_document_exists():
    assert Path(SAMPLE).is_file()
