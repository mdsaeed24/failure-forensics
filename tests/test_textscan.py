from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from forensics.textscan import find_dates, find_money, format_amount, format_date


@pytest.mark.parametrize("text, amount, currency", [
    ("Total £1,028.40 due", "1028.40", "GBP"),
    ("GBP 1,028.40", "1028.40", "GBP"),
    ("1.028,40 €", "1028.40", "EUR"),
    ("fee: $25", "25", "USD"),
    ("C$500", "500", "CAD"),
    ("CHF 3,000.00", "3000.00", "CHF"),
    ("¥5,000", "5000", "JPY"),
    ("₹1,41,600.00", "141600.00", "INR"),  # Indian lakh grouping
])
def test_money_forms(text, amount, currency):
    [m] = find_money(text)
    assert (m.amount, m.currency) == (Decimal(amount), currency)


@pytest.mark.parametrize("text", [
    "12 packs", "VAT 20%", "Invoice INV-2026-0142", "Ref USD-0142", "costs 3.141 units",
])
def test_not_money(text):
    assert find_money(text) == []


def test_reference_number_next_to_a_code_is_not_money():
    assert [(m.amount, m.currency) for m in find_money("INV-2026-0142 USD 50")] == [
        (Decimal("50"), "USD")
    ]


@pytest.mark.parametrize("text, expected", [
    ("3 March 2026", dt.date(2026, 3, 3)),
    ("3rd of March, 2026", dt.date(2026, 3, 3)),
    ("March 3, 2026", dt.date(2026, 3, 3)),
    ("Mar. 3rd 2026", dt.date(2026, 3, 3)),
    ("2026-03-03", dt.date(2026, 3, 3)),
    ("3 SEPT 2026", dt.date(2026, 9, 3)),
])
def test_date_forms(text, expected):
    [d] = find_dates(text)
    assert d.value == expected and not d.ambiguous


def test_numeric_dates_are_ambiguous_and_impossible_dates_have_no_value():
    numeric, impossible = find_dates("on 03/04/2026 or 31 February 2026")
    assert numeric.ambiguous and numeric.value is None
    assert not impossible.ambiguous and impossible.value is None


@pytest.mark.parametrize("text", ["March 2026", "Market 5 2026", "5 pallets in 2026"])
def test_not_full_dates(text):
    assert find_dates(text) == []


def test_formatting():
    assert format_amount(Decimal("1028.4"), "GBP") == "£1,028.40"
    assert format_amount(Decimal("5000"), "JPY") == "¥5,000"
    assert format_amount(Decimal("7"), "CHF") == "CHF 7.00"
    assert format_date(dt.date(2026, 3, 3)) == "3 March 2026"
