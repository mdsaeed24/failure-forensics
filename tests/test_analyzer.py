"""Scoring and root-cause analysis on real traces made with the FakeLLM."""

from __future__ import annotations

import asyncio

import openai
import pytest

from fakes import CLASSIFICATION, EXTRACTION, SUMMARY, api_status_error
from forensics.analyzer.root_cause import analyze
from forensics.analyzer.taxonomy import FailureType
from forensics.pipeline.models import RawDocument
from forensics.pipeline.runner import PipelineOptions, run_pipeline
from forensics.scoring.final_score import score_trace
from test_checks import SOURCE

DOC = RawDocument(doc_id="inv_test", text=SOURCE)
TRUTH = {
    "doc_type": "invoice",
    "parties": [{"name": "Harbor Lane Printing Ltd.", "role": "seller"},
                {"name": "Meridian Health Clinics", "role": "buyer"},
                {"name": "Priya Raman", "role": "contact"}],
    "dates": [{"date": "2026-03-03", "label": "invoice date"}, {"date": "2026-04-02", "label": "due date"}],
    "amounts": [{"amount": "1028.40", "currency": "GBP", "label": "total due"}],
}
OK = {"score": 5, "failure_type": "none", "reason": "Correct and complete."}


def make_trace(make_llm, settings, trace_store, *replies, hooks=()):
    llm, _ = make_llm(*(replies or (EXTRACTION, CLASSIFICATION, SUMMARY)))
    result = asyncio.run(run_pipeline(DOC, llm, settings, PipelineOptions(hooks=list(hooks))))
    return trace_store.load(result.trace_id)


def diagnose(make_llm, settings, trace_doc, *judge_replies):
    llm, fake = make_llm(*judge_replies)
    diagnosis = asyncio.run(analyze(trace_doc, llm, settings, symptom=["something was wrong"]))
    return diagnosis, fake


def hook(step_name, **update):
    def change(step, output):
        return output.model_copy(update=update) if step == step_name else output
    return change


# --- final score ----------------------------------------------------------------

def test_faithful_run_scores_good(make_llm, settings, trace_store):
    assert score_trace(make_trace(make_llm, settings, trace_store), TRUTH).good


def test_score_lists_every_discrepancy(make_llm, settings, trace_store):
    from forensics.pipeline.models import AmountFact, Party
    trace_doc = make_trace(make_llm, settings, trace_store, hooks=[
        hook("classification", doc_type="quotation"),
        hook("extraction", parties=[Party(name="Harbor Lane Printing", role="seller"),
                                    Party(name="Meridian Health Clinics", role="buyer")],
             amounts=[AmountFact(amount=1028.40, currency="GBP", label="t", quote="x"),
                      AmountFact(amount=25, currency="GBP", label="fee", quote="y")]),
    ])
    score = score_trace(trace_doc, TRUTH)
    assert not score.good
    assert {(d.field, d.kind) for d in score.discrepancies} == {
        ("doc_type", "wrong"), ("party", "missing"), ("amount", "extra"),
    }  # "Harbor Lane Printing" still matches "Harbor Lane Printing Ltd."


def test_failed_run_scores_bad(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store, EXTRACTION, "bad", "bad")
    assert ("pipeline", "failed") in {(d.field, d.kind) for d in score_trace(trace_doc, TRUTH).discrepancies}


# --- root cause -------------------------------------------------------------------

def test_clean_run_finds_no_root_cause_after_judging_every_step(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    d, fake = diagnose(make_llm, settings, trace_doc, OK, OK, OK)
    assert d.root_cause_step is None and d.failure_type is None
    assert [v.verdict for v in d.steps] == ["ok"] * 4
    assert d.judge_calls == 3 and "No step-level failure" in d.headline
    request = fake.requests[0]
    assert request["model"] == "fake-pro"  # the judge model, not the pipeline model
    assert '"confidence"' not in request["messages"][1]["content"]  # self-confidence hidden


def test_deterministic_finding_stops_the_walk_without_any_judge_call(make_llm, settings, trace_store):
    from forensics.pipeline.models import AmountFact
    bad = AmountFact(amount=1082.40, currency="GBP", label="total due", quote="£1,028.40")
    trace_doc = make_trace(make_llm, settings, trace_store, hooks=[hook("extraction", amounts=[bad])])
    d, _ = diagnose(make_llm, settings, trace_doc)  # no judge replies scripted: none may be needed
    assert (d.root_cause_step, d.failure_type, d.confidence) == ("extraction", FailureType.WRONG_VALUE, "high")
    assert [v.verdict for v in d.steps] == ["ok", "failed", "not_checked", "not_checked"]
    assert d.judge_calls == 0
    assert d.evidence[0] == "Symptom: something was wrong"
    assert any("extracted £1,082.40" in line for line in d.evidence)
    assert any("summary does not repeat £1,082.40" in line for line in d.evidence)


def test_judge_finds_a_misclassification_and_the_chain_follows_it(make_llm, settings, trace_store):
    summary = {"summary": "A quotation from Harbor Lane Printing Ltd. for £1,028.40.", "confidence": 4}
    trace_doc = make_trace(make_llm, settings, trace_store, EXTRACTION, CLASSIFICATION, summary,
                           hooks=[hook("classification", doc_type="quotation")])
    wrong = {"score": 1, "failure_type": "misclassification", "reason": "It bills for supplied goods.",
             "evidence": "Total due: £1,028.40"}
    d, fake = diagnose(make_llm, settings, trace_doc, OK, wrong)
    assert (d.root_cause_step, d.failure_type, d.confidence) == (
        "classification", FailureType.MISCLASSIFICATION, "medium")
    assert d.judge_calls == 2  # summarization never judged
    assert any('told the document type is "quotation", and the summary repeats it' in l for l in d.evidence)
    assert '"doc_type": "quotation"' in fake.requests[1]["messages"][1]["content"]


def test_judge_failure_type_none_on_classification_still_means_misclassification(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    d, _ = diagnose(make_llm, settings, trace_doc, OK, {"score": 2, "failure_type": "none", "reason": "?"})
    assert d.failure_type == FailureType.MISCLASSIFICATION


def test_only_a_suspect_is_named_with_low_confidence(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    unsure = {"score": 3, "failure_type": "omission", "reason": "The contact's role may be missing."}
    d, _ = diagnose(make_llm, settings, trace_doc, unsure, OK, OK)
    assert (d.root_cause_step, d.failure_type, d.confidence) == ("extraction", FailureType.OMISSION, "low")
    assert [v.verdict for v in d.steps] == ["ok", "suspect", "ok", "ok"]


def test_a_later_clear_failure_beats_an_earlier_suspect(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    d, _ = diagnose(make_llm, settings, trace_doc,
                    {"score": 3, "failure_type": "omission", "reason": "maybe"},
                    OK, {"score": 2, "failure_type": "hallucination", "reason": "adds a name"})
    assert (d.root_cause_step, d.confidence) == ("summarization", "medium")


def test_crashed_step_is_a_format_error(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store, EXTRACTION, "bad", "still bad")
    d, _ = diagnose(make_llm, settings, trace_doc, OK)  # extraction judged first, then the crash
    assert (d.root_cause_step, d.failure_type, d.confidence) == ("classification", FailureType.FORMAT_ERROR, "high")
    assert [v.verdict for v in d.steps] == ["ok", "ok", "crashed", "not_checked"]


def test_api_outage_is_not_blamed_on_the_model(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store,
                           api_status_error(openai.AuthenticationError, 401))
    d, _ = diagnose(make_llm, settings, trace_doc)
    assert d.root_cause_step == "extraction" and d.failure_type is None
    assert "infrastructure" in d.headline


def test_diagnosis_is_stored(make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    d, _ = diagnose(make_llm, settings, trace_doc, OK, OK, OK)
    trace_store.save_diagnosis(d.model_dump(mode="json"))
    assert trace_store.load_diagnosis(d.trace_id)["headline"] == d.headline
