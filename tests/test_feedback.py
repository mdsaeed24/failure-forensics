"""Flags -> eval cases -> regression, through the Python API and over HTTP."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from fakes import CLASSIFICATION, EXTRACTION, SUMMARY, pipeline_or_judge_reply
from forensics.api.main import create_app
from forensics.feedback.eval_cases import Feedback, FeedbackError
from forensics.feedback.regression import apply_result, check_case
from test_analyzer import DOC, OK, TRUTH, diagnose, hook, make_trace


@pytest.fixture
def feedback(trace_store, tmp_path):
    return Feedback(trace_store, tmp_path / "eval_cases.json")


@pytest.fixture
def bad_trace(make_llm, settings, trace_store):
    """A run whose classification was swapped, with a stored diagnosis blaming it."""
    trace_doc = make_trace(make_llm, settings, trace_store, hooks=[hook("classification", doc_type="quotation")])
    d, _ = diagnose(make_llm, settings, trace_doc, OK,
                    {"score": 1, "failure_type": "misclassification", "reason": "it bills for goods"})
    trace_store.save_diagnosis(d.model_dump(mode="json"))
    return trace_doc


# --- feedback service -----------------------------------------------------------

def test_flag_then_confirm_the_diagnosis(feedback, bad_trace, tmp_path):
    flag = feedback.flag_trace(bad_trace["trace_id"], "  called a quotation  ", "sam")
    assert (flag.status, flag.note, flag.doc_id) == ("open", "called a quotation", "inv_test")

    case = feedback.confirm(bad_trace["trace_id"])
    assert (case.case_id, case.root_cause_step, case.failure_type.value) == ("case_001", "classification", "misclassification")
    assert case.diagnosis_agreed and case.analyzer_said is None and case.status == "open"
    assert feedback.get_flag(bad_trace["trace_id"]).status == "confirmed"
    assert (tmp_path / "eval_cases.json").is_file() and len(feedback.load_cases()) == 1


def test_override_keeps_what_the_analyzer_said(feedback, bad_trace):
    feedback.flag_trace(bad_trace["trace_id"], "the extraction missed a party")
    case = feedback.confirm(bad_trace["trace_id"], "extraction", "omission", "checked by hand")
    assert not case.diagnosis_agreed and case.analyzer_said == "classification / misclassification"
    assert case.note == "the extraction missed a party / checked by hand"


def test_confirm_without_a_diagnosis_needs_step_and_type(feedback, make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    feedback.flag_trace(trace_doc["trace_id"], "looks off")
    with pytest.raises(FeedbackError, match="analyze the trace first"):
        feedback.confirm(trace_doc["trace_id"])
    case = feedback.confirm(trace_doc["trace_id"], "summarization", "hallucination")
    assert case.case_id == "case_001" and case.analyzer_said == "not analyzed"


@pytest.mark.parametrize("call, message", [
    (lambda fb, t: fb.flag_trace(t, "   "), "say what looks wrong"),
    (lambda fb, t: fb.flag_trace("nope", "x"), "no trace"),
    (lambda fb, t: fb.confirm(t), "has not been flagged"),
    (lambda fb, t: fb.dismiss(t), "has not been flagged"),
])
def test_invalid_requests_are_refused(feedback, bad_trace, call, message):
    with pytest.raises(FeedbackError, match=message):
        call(feedback, bad_trace["trace_id"])


def test_a_closed_flag_cannot_be_closed_again_but_the_trace_can_be_reflagged(feedback, bad_trace):
    t = bad_trace["trace_id"]
    feedback.flag_trace(t, "first")
    with pytest.raises(FeedbackError, match="already flagged"):
        feedback.flag_trace(t, "again")
    feedback.dismiss(t, "fine after all")
    with pytest.raises(FeedbackError, match="already dismissed"):
        feedback.confirm(t)
    assert feedback.flag_trace(t, "actually not fine").status == "open"


# --- HTTP API -------------------------------------------------------------------

@pytest.fixture
def client(settings, trace_store, tmp_path, make_llm):
    s = settings.model_copy(update={"paths": settings.paths.model_copy(update={
        "traces_dir": trace_store.dir, "eval_cases": tmp_path / "eval_cases.json",
        "ground_truth": tmp_path / "missing.json"})})
    judge = make_llm(OK, OK, OK)[0]
    return TestClient(create_app(s, llm_factory=lambda _: judge))


def test_api_end_to_end(client, make_llm, settings, trace_store):
    trace_doc = make_trace(make_llm, settings, trace_store)
    short = trace_doc["trace_id"][:8]  # prefixes work everywhere

    assert client.get("/health").json() == {"status": "ok"}
    assert [t["trace_id"] for t in client.get("/traces").json()] == [trace_doc["trace_id"]]
    analyzed = client.post(f"/traces/{short}/analyze")
    assert analyzed.status_code == 200 and analyzed.json()["root_cause_step"] is None

    flagged = client.post(f"/traces/{short}/flag", json={"note": "summary is vague", "flagged_by": "qa"})
    assert flagged.status_code == 201 and flagged.json()["status"] == "open"
    assert client.post(f"/traces/{short}/flag", json={"note": "again"}).status_code == 409
    assert len(client.get("/flags?status=open").json()) == 1

    no_diagnosis = client.post(f"/flags/{short}/confirm", json={})
    assert no_diagnosis.status_code == 400  # the analyzer found nothing to confirm
    confirmed = client.post(f"/flags/{short}/confirm",
                            json={"root_cause_step": "summarization", "failure_type": "omission"})
    assert confirmed.status_code == 201 and confirmed.json()["case_id"] == "case_001"
    assert client.get(f"/traces/{short}").json()["flag"]["status"] == "confirmed"
    assert len(client.get("/eval-cases").json()) == 1


def test_api_unknown_trace_is_404(client):
    assert client.get("/traces/deadbeef").status_code == 404
    assert client.post("/traces/deadbeef/flag", json={"note": "x"}).status_code == 404


# --- regression -----------------------------------------------------------------

def make_case(feedback, bad_trace):
    feedback.flag_trace(bad_trace["trace_id"], "called a quotation")
    return feedback.confirm(bad_trace["trace_id"])


def regress(case, make_llm, settings, trace_store, *replies, truth=None):
    llm, fake = make_llm(*replies)
    result = asyncio.run(check_case(case, llm, settings, trace_store, truth or {"inv_test": TRUTH}, "reg"))
    return result, fake


def test_regression_resolves_a_fixed_issue(feedback, bad_trace, make_llm, settings, trace_store):
    case = make_case(feedback, bad_trace)
    result, fake = regress(case, make_llm, settings, trace_store, EXTRACTION, CLASSIFICATION, SUMMARY, OK)
    assert result.passed and result.trace_id != case.source_trace_id
    judge_prompt = fake.requests[-1]["messages"][1]["content"]
    assert "A REVIEWER REPORTED THIS PROBLEM" in judge_prompt and "called a quotation" in judge_prompt
    assert apply_result(case, result).status == "resolved"


def test_regression_keeps_a_case_open_while_the_judge_still_sees_it(feedback, bad_trace, make_llm, settings, trace_store):
    case = make_case(feedback, bad_trace)
    still = {"score": 2, "failure_type": "misclassification", "reason": "still a quotation"}
    result, _ = regress(case, make_llm, settings, trace_store, EXTRACTION, CLASSIFICATION, SUMMARY, still)
    assert not result.passed and "still a quotation" in result.reason
    assert apply_result(case, result).status == "open"
    resolved = case.model_copy(update={"status": "resolved"})
    assert apply_result(resolved, result).status == "regressed"


def test_regression_uses_ground_truth_when_there_is_some(feedback, bad_trace, make_llm, settings, trace_store):
    case = make_case(feedback, bad_trace)
    wrong_truth = {"inv_test": TRUTH | {"doc_type": "receipt"}}
    result, _ = regress(case, make_llm, settings, trace_store, EXTRACTION, CLASSIFICATION, SUMMARY, OK,
                        truth=wrong_truth)
    assert not result.passed and result.reason.startswith("final output:")


def test_api_outage_leaves_the_status_alone(feedback, bad_trace, make_llm, settings, trace_store):
    case = make_case(feedback, bad_trace)
    result, _ = regress(case, make_llm, settings, trace_store, "bad", "bad")  # extraction can't complete
    assert not result.passed


def test_regression_reads_the_document_from_the_source_trace(feedback, bad_trace, make_llm, settings, trace_store):
    case = make_case(feedback, bad_trace)
    _, fake = regress(case, make_llm, settings, trace_store, *[pipeline_or_judge_reply] * 4)
    assert DOC.text.strip()[:20] in fake.requests[0]["messages"][1]["content"]
