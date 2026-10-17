from __future__ import annotations

from decimal import Decimal

import pytest
import yaml

from forensics.config import PROJECT_ROOT
from forensics.documents.fact_sheets import FactSheet, FactSheetError, load_fact_sheets

GOOD = {
    "doc_id": "inv_001",
    "doc_type": "invoice",
    "parties": [{"name": "Acme Ltd", "role": "seller"}],
    "dates": [{"date": "2026-03-03", "label": "invoice date"}],
    "amounts": [{"amount": 19.99, "currency": "GBP", "label": "total"}],
}


def write(tmp_path, sheets):
    path = tmp_path / "sheets.yaml"
    path.write_text(yaml.safe_dump(sheets), encoding="utf-8")
    return path


def test_project_fact_sheets_are_valid():
    sheets = load_fact_sheets(PROJECT_ROOT / "data" / "fact_sheets.yaml")
    assert len(sheets) >= 3


def test_yaml_floats_become_exact_decimals(tmp_path):
    [sheet] = load_fact_sheets(write(tmp_path, [GOOD]))
    assert sheet.amounts[0].amount == Decimal("19.99")


@pytest.mark.parametrize("change, message", [
    ({"doc_type": "bill"}, "doc_type"),
    ({"dates": [{"date": "2026-02-30", "label": "x"}]}, "dates.0.date"),
    ({"amounts": [{"amount": 5, "currency": "XYZ", "label": "x"}]}, "not a supported"),
    ({"amounts": [{"amount": 1.234, "currency": "GBP", "label": "x"}]}, "2 decimal"),
    ({"amounts": [{"amount": 10.5, "currency": "JPY", "label": "x"}]}, "whole numbers"),
    ({"parties": []}, "parties"),
    ({"doc_id": "Inv 1"}, "doc_id"),
    ({"partys": []}, "partys"),  # typo'd keys are caught, not ignored
])
def test_invalid_sheets_are_rejected(tmp_path, change, message):
    with pytest.raises(FactSheetError, match=message):
        load_fact_sheets(write(tmp_path, [GOOD | change]))


def test_all_problems_reported_at_once(tmp_path):
    with pytest.raises(FactSheetError) as info:
        load_fact_sheets(write(tmp_path, [GOOD | {"doc_type": "bill"}, GOOD | {"parties": []}, GOOD]))
    assert "2 problem(s)" in str(info.value) or "3 problem(s)" in str(info.value)
    assert "sheet #1" in str(info.value) and "sheet #2" in str(info.value)


def test_duplicate_doc_ids_are_rejected(tmp_path):
    with pytest.raises(FactSheetError, match="duplicate doc_id"):
        load_fact_sheets(write(tmp_path, [GOOD, GOOD]))


def test_hash_changes_with_any_fact():
    a = FactSheet.model_validate(GOOD)
    b = FactSheet.model_validate(GOOD | {"amounts": [{"amount": 20, "currency": "GBP", "label": "total"}]})
    assert a.sha256() == FactSheet.model_validate(GOOD).sha256()
    assert a.sha256() != b.sha256()
