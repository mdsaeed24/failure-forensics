"""Verifier, generator loop and document store, all against the FakeLLM."""

from __future__ import annotations

import asyncio
import json

import pytest

from forensics.documents.fact_sheets import FactSheet
from forensics.documents.generator import generate
from forensics.documents.report import side_by_side, verification_report
from forensics.documents.store import DocumentStore
from forensics.documents.verifier import check_deterministic, compare_audit, AuditOutput

SHEET = FactSheet.model_validate({
    "doc_id": "inv_001",
    "doc_type": "invoice",
    "parties": [{"name": "Acme Ltd", "role": "seller"}, {"name": "Priya Raman", "role": "buyer"}],
    "dates": [{"date": "2026-03-03", "label": "invoice date"}],
    "amounts": [{"amount": 1028.40, "currency": "GBP", "label": "total due"}],
})
GOOD_DOC = "# Invoice\n\nFrom: Acme Ltd\nTo: Priya Raman\nInvoice date: 3 March 2026\n\nTotal due: £1,028.40\n"
AUDIT_OK = {"names": ["Acme Ltd", "Priya Raman"], "dates": ["2026-03-03"],
            "amounts": [{"amount": 1028.40, "currency": "GBP"}],
            "doc_type": "invoice", "doc_type_reason": "It bills for goods."}


# --- deterministic checks -----------------------------------------------------

def test_good_document_passes_with_every_fact_located():
    issues, found = check_deterministic(SHEET, GOOD_DOC)
    assert issues == []
    assert {(f.kind, f.surface, f.line) for f in found} == {
        ("party", "Acme Ltd", 3), ("party", "Priya Raman", 4),
        ("date", "3 March 2026", 5), ("amount", "£1,028.40", 7),
    }


@pytest.mark.parametrize("doc, check", [
    (GOOD_DOC.replace("Acme Ltd", "ACME Ltd"), "missing_fact"),          # names are exact
    (GOOD_DOC.replace("£1,028.40", "1,028.40"), "missing_fact"),         # no currency
    (GOOD_DOC.replace("£1,028.40", "€1,028.40"), "missing_fact"),        # wrong currency
    (GOOD_DOC + "Late fee: £25.00\n", "extra_amount"),
    (GOOD_DOC + "Delivered 1 March 2026\n", "extra_date"),
    (GOOD_DOC + "Ref 03/03/2026\n", "ambiguous_date"),
    (GOOD_DOC + "Shipped 30 February 2026\n", "invalid_date"),
    (GOOD_DOC + "Address: [address]\n", "placeholder"),
])
def test_deterministic_failures(doc, check):
    issues, _ = check_deterministic(SHEET, doc)
    assert check in {i.check for i in issues}
    assert all(i.severity == "error" for i in issues)


def test_links_and_checkboxes_are_not_placeholders():
    issues, _ = check_deterministic(SHEET, GOOD_DOC + "- [x] paid\n- [ ] filed\nSee [terms](#terms)\n")
    assert issues == []


def test_wrong_capitalization_gets_a_specific_hint():
    issues, _ = check_deterministic(SHEET, GOOD_DOC.replace("From: Acme Ltd", "From: ACME LTD"))
    assert 'it appears as "ACME LTD"' in issues[0].message


def test_anything_the_auditor_finds_beyond_the_sheet_is_an_error():
    report = AuditOutput.model_validate(AUDIT_OK | {
        "names": ["Acme", "Priya Raman", "Barclays Bank"],  # "Acme" matches "Acme Ltd"
        "amounts": [{"amount": 1028.40, "currency": "GBP"}, {"amount": 5, "currency": "GBP"}],
        "doc_type": "quotation",
    })
    issues = {(i.check, i.severity) for i in compare_audit(SHEET, report)}
    assert issues == {("doc_type", "error"), ("extra_name", "error"), ("extra_amount", "error")}


# --- generator loop -----------------------------------------------------------

def run_generate(make_llm, settings, *replies, note=None):
    llm, fake = make_llm(*replies)
    return asyncio.run(generate(SHEET, llm, settings.generation, reviewer_note=note)), fake


def test_passing_document_is_audited_once(make_llm, settings):
    outcome, fake = run_generate(make_llm, settings, {"document": GOOD_DOC}, AUDIT_OK)
    assert outcome.status == "verified" and outcome.attempts == 1
    assert outcome.verification.audit.doc_type == "invoice"
    writer, auditor = fake.requests
    assert writer["extra_body"] == {"thinking": {"type": "disabled"}}  # generator: thinking off
    assert "extra_body" not in auditor and auditor["max_tokens"] == 8000  # verifier: thinking on
    assert len(outcome.llm_calls) == 2


def test_failed_check_skips_the_audit_and_feeds_errors_back(make_llm, settings):
    bad = GOOD_DOC.replace("£1,028.40", "1,028.40")
    outcome, fake = run_generate(make_llm, settings, {"document": bad}, {"document": GOOD_DOC}, AUDIT_OK)
    assert outcome.status == "verified" and outcome.attempts == 2
    assert len(fake.requests) == 3  # write, rewrite, audit: no audit of the bad draft
    rewrite_prompt = fake.requests[1]["messages"][1]["content"]
    assert "Your previous attempt was rejected" in rewrite_prompt and bad in rewrite_prompt
    assert "is missing" in rewrite_prompt
    assert len(outcome.rejected_attempts) == 1


def test_gives_up_after_max_attempts(make_llm, settings):
    bad = {"document": GOOD_DOC + "Late fee: £25.00\n"}
    outcome, _ = run_generate(make_llm, settings, bad, bad, bad)
    assert outcome.status == "failed" and outcome.attempts == 3
    assert outcome.document is not None  # kept for the reviewer to look at
    assert len(outcome.rejected_attempts) == 2


def test_audit_failure_triggers_a_rewrite(make_llm, settings):
    wrong_type = AUDIT_OK | {"doc_type": "quotation", "doc_type_reason": "It offers prices."}
    outcome, fake = run_generate(make_llm, settings, {"document": GOOD_DOC}, wrong_type,
                                 {"document": GOOD_DOC}, AUDIT_OK)
    assert outcome.status == "verified" and outcome.attempts == 2
    assert 'reads as "quotation"' in fake.requests[2]["messages"][1]["content"]


def test_reviewer_note_reaches_the_generator(make_llm, settings):
    _, fake = run_generate(make_llm, settings, {"document": GOOD_DOC}, AUDIT_OK, note="too short")
    assert "A reviewer rejected an earlier version" in fake.requests[0]["messages"][1]["content"]
    assert "too short" in fake.requests[0]["messages"][1]["content"]


def test_api_error_is_reported_not_raised(make_llm, settings):
    outcome, _ = run_generate(make_llm, settings, "not json", "still not json")
    assert outcome.status == "error" and "LLMOutputError" in outcome.error


# --- store and ground truth ---------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return DocumentStore(tmp_path / "documents", tmp_path / "ground_truth.json")


def verified_outcome(make_llm, settings):
    return run_generate(make_llm, settings, {"document": GOOD_DOC}, AUDIT_OK)[0]


def test_only_approved_current_documents_enter_ground_truth(store, make_llm, settings, tmp_path):
    store.record(SHEET, verified_outcome(make_llm, settings), "fake-pro")
    assert store.is_current(SHEET)
    assert store.build_ground_truth([SHEET]) == (0, ["inv_001: review pending"])

    store.set_review("inv_001", "approved")
    assert store.build_ground_truth([SHEET])[0] == 1
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    assert truth["inv_001"]["amounts"][0] == {"amount": "1028.40", "currency": "GBP", "label": "total due"}

    reloaded = DocumentStore(tmp_path / "documents", tmp_path / "ground_truth.json")
    assert reloaded.entries["inv_001"].review.status == "approved"


def test_editing_the_sheet_or_the_document_drops_it_from_ground_truth(store, make_llm, settings):
    store.record(SHEET, verified_outcome(make_llm, settings), "fake-pro")
    store.set_review("inv_001", "approved")

    changed_sheet = SHEET.model_copy(update={"doc_type": "receipt"})
    assert store.build_ground_truth([changed_sheet])[0] == 0

    store.doc_path("inv_001").write_text(GOOD_DOC + "edited\n", encoding="utf-8")
    included, skipped = store.build_ground_truth([SHEET])
    assert included == 0 and "changed since approval" in skipped[0]


def test_regenerating_resets_review(store, make_llm, settings):
    store.record(SHEET, verified_outcome(make_llm, settings), "fake-pro")
    store.set_review("inv_001", "rejected", "too short")
    store.record(SHEET, verified_outcome(make_llm, settings), "fake-pro")
    assert store.entries["inv_001"].review.status == "pending"


def test_reports_render(store, make_llm, settings):
    store.record(SHEET, verified_outcome(make_llm, settings), "fake-pro")
    report = verification_report(store, [SHEET])
    assert "| inv_001 | invoice | verified | 1 | 0 | 0 | pending |" in report
    screen = side_by_side(SHEET, store.entries["inv_001"], GOOD_DOC, width=100, color=False)
    assert "✓ Acme Ltd (seller)" in screen and '"£1,028.40" · line 7' in screen
    assert "auditor reads it as invoice" in screen
