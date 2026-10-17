from __future__ import annotations

import copy

import pytest

from fakes import EXTRACTION, SUMMARY
from forensics.analyzer.taxonomy import FailureType
from forensics.checks import run_checks

SOURCE = """# INVOICE
From: **Harbor Lane Printing Ltd.**
Bill to: Meridian Health Clinics, attn. Priya Raman
Invoice date: 3 March 2026
Due date: 2 April 2026
Total due: £1,028.40
"""
INTAKE_OUT = {"doc_id": "d", "text": SOURCE.strip(), "title": "INVOICE", "char_count": 0,
              "word_count": 0, "sha256": ""}


def extraction_findings(change):
    output = copy.deepcopy(EXTRACTION)
    change(output)
    return run_checks("extraction", INTAKE_OUT, output)


def kinds(findings):
    return sorted((f.check, f.failure_type) for f in findings)


def test_faithful_extraction_has_no_findings():
    assert extraction_findings(lambda o: None) == []


def test_invented_party_is_a_hallucination():
    findings = extraction_findings(lambda o: o["parties"].append({"name": "Daniel Okafor", "role": "contact"}))
    assert kinds(findings) == [("ungrounded_party", FailureType.HALLUCINATION)]
    assert findings[0].step == "extraction" and findings[0].value == "Daniel Okafor"


def test_changed_date_with_true_quote_is_a_wrong_value_not_also_an_omission():
    findings = extraction_findings(lambda o: o["dates"][0].update(date="2026-03-04"))
    assert kinds(findings) == [("date_mismatch", FailureType.WRONG_VALUE)]
    assert findings[0].message.endswith('but the document says "3 March 2026"')


def test_date_with_no_support_is_a_hallucination():
    findings = extraction_findings(lambda o: o["dates"].append(
        {"date": "2026-05-01", "label": "delivery", "quote": "1 May 2026"}))
    assert kinds(findings) == [("ungrounded_date", FailureType.HALLUCINATION)]


def test_changed_amount_with_true_quote_is_a_wrong_value():
    findings = extraction_findings(lambda o: o["amounts"][0].update(amount=1082.40))
    assert kinds(findings) == [("amount_mismatch", FailureType.WRONG_VALUE)]
    assert '"£1,028.40"' in findings[0].message


def test_changed_currency_is_a_wrong_value():
    findings = extraction_findings(lambda o: o["amounts"][0].update(currency="EUR"))
    assert kinds(findings) == [("currency_mismatch", FailureType.WRONG_VALUE)]


def test_dropped_facts_are_omissions():
    def drop(o):
        o["amounts"].clear()
        o["dates"].pop()
    assert kinds(extraction_findings(drop)) == [
        ("missed_amount", FailureType.OMISSION), ("missed_date", FailureType.OMISSION),
    ]


def test_malformed_values_are_format_errors_and_not_double_reported():
    def malform(o):
        o["dates"][0]["date"] = "03/03/2026"
        o["amounts"][0]["currency"] = "£"
    findings = extraction_findings(malform)
    assert ("bad_date_format", FailureType.FORMAT_ERROR) in kinds(findings)
    assert ("bad_currency_code", FailureType.FORMAT_ERROR) in kinds(findings)
    assert {f.check for f in findings} == {"bad_date_format", "bad_currency_code",
                                           "missed_date", "missed_amount"}


def test_quotes_may_drop_markdown():
    findings = extraction_findings(lambda o: o["amounts"][0].update(quote="Total due: £1,028.40"))
    assert findings == []


@pytest.mark.parametrize("text, check, failure_type", [
    (SOURCE.strip()[:60], "intake_dropped_text", FailureType.OMISSION),
    (SOURCE.strip().replace("1,028.40", "1,082.40"), "intake_altered_text", FailureType.WRONG_VALUE),
])
def test_intake_must_hand_on_the_text_intact(text, check, failure_type):
    raw = {"doc_id": "d", "text": SOURCE}
    [finding] = run_checks("intake", raw, INTAKE_OUT | {"text": text})
    assert (finding.check, finding.failure_type) == (check, failure_type)
    assert run_checks("intake", raw, INTAKE_OUT) == []


def summary_findings(summary):
    inp = {"extraction": EXTRACTION, "classification": {"doc_type": "invoice", "reason": "", "confidence": 5}}
    return run_checks("summarization", inp, {"summary": summary, "confidence": 5})


def test_summary_using_only_given_facts_passes():
    assert summary_findings(SUMMARY["summary"]) == []


def test_summary_replacing_a_given_amount_is_a_wrong_value():
    [f] = summary_findings("Harbor Lane invoiced £1,082.40, due 2 April 2026.")
    assert f.failure_type == FailureType.WRONG_VALUE and f.value == "1082.40 GBP"


def test_summary_adding_an_amount_is_a_hallucination():
    [f] = summary_findings("Total £1,028.40 due 2 April 2026, plus a £25.00 late fee.")
    assert f.failure_type == FailureType.HALLUCINATION


def test_classification_has_no_deterministic_checks():
    assert run_checks("classification", INTAKE_OUT, {"doc_type": "receipt", "reason": "", "confidence": 1}) == []
