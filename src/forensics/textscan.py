"""Find money amounts and calendar dates written in free text. No LLM.

Used by the Phase 2 document verifier; the Phase 4 checks can reuse it.

Supported ways of writing them:
- money: a currency symbol or ISO code right before or after the number, e.g.
  "£1,028.40", "GBP 1,028.40", "1.028,40 €". A bare number is not money.
- dates: "3 March 2026", "3rd of March, 2026", "March 3, 2026", "Mar. 3 2026",
  "2026-03-03". Numeric dates like 03/04/2026 are found but flagged as
  ambiguous (3 April or 4 March?), so they have no value.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from decimal import Decimal

# Currency -> symbols that may stand for it, preferred first. The ISO code always works.
# A symbol is listed under one currency only, so a match is never ambiguous:
# "$" means USD; Canadian dollars must be written C$, CA$ or CAD.
CURRENCY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "USD": ("$", "US$"),
    "GBP": ("£",),
    "EUR": ("€",),
    "JPY": ("¥",),
    "INR": ("₹", "Rs."),
    "CAD": ("C$", "CA$"),
    "AUD": ("A$", "AU$"),
    "NZD": ("NZ$",),
    "SGD": ("S$",),
    "HKD": ("HK$",),
    "BRL": ("R$",),
    "KRW": ("₩",),
    "CHF": (),
    "CNY": (),
    "SEK": (),
    "NOK": (),
    "DKK": (),
    "PLN": (),
    "ZAR": (),
    "AED": (),
    "SAR": (),
    "MXN": (),
}
SUPPORTED_CURRENCIES = frozenset(CURRENCY_SYMBOLS)
ZERO_DECIMAL_CURRENCIES = frozenset({"JPY", "KRW"})

_MARKER_TO_CURRENCY = {code: code for code in CURRENCY_SYMBOLS} | {
    symbol: code for code, symbols in CURRENCY_SYMBOLS.items() for symbol in symbols
}
_MARKER = "(?:" + "|".join(
    re.escape(m) for m in sorted(_MARKER_TO_CURRENCY, key=len, reverse=True)
) + ")"
# Broad on purpose; _parse_number decides whether the digits form a valid amount.
# Not preceded by a word character or "-", so "INV-2026-0142" is not a number.
_NUMBER = r"(?<![\w.,-])\d(?:[\d.,]*\d)?"
_MONEY = re.compile(
    rf"(?<![A-Za-z])(?P<pre>{_MARKER})(?![A-Za-z])\s?(?P<n1>{_NUMBER})"
    rf"|(?P<n2>{_NUMBER})\s?(?<![A-Za-z])(?P<post>{_MARKER})(?![A-Za-z])"
)

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_ORD = r"(?:st|nd|rd|th)?"
_DATE = re.compile(
    rf"\b(?P<d1>\d{{1,2}}){_ORD}(?:\s+of)?\s+(?P<m1>{_MONTH}),?\s+(?P<y1>\d{{4}})\b"
    rf"|\b(?P<m2>{_MONTH})\s+(?P<d2>\d{{1,2}}){_ORD},?\s+(?P<y2>\d{{4}})\b"
    r"|\b(?P<y3>\d{4})-(?P<m3>\d{2})-(?P<d3>\d{2})\b"
    r"|\b(?P<numeric>\d{1,2}[/.]\d{1,2}[/.]\d{2,4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MoneyMention:
    amount: Decimal
    currency: str
    surface: str  # exactly as written
    start: int


@dataclass(frozen=True)
class DateMention:
    value: dt.date | None  # None: ambiguous numeric format, or not a real date (31 February)
    surface: str
    start: int
    ambiguous: bool = False


def find_money(text: str) -> list[MoneyMention]:
    found = []
    for m in _MONEY.finditer(text):
        number = m.group("n1") or m.group("n2")
        amount = _parse_number(number)
        if amount is None:
            continue
        currency = _MARKER_TO_CURRENCY[m.group("pre") or m.group("post")]
        found.append(MoneyMention(amount, currency, m.group(0), m.start()))
    return found


def find_dates(text: str) -> list[DateMention]:
    found = []
    for m in _DATE.finditer(text):
        if m.group("numeric"):
            found.append(DateMention(None, m.group(0), m.start(), ambiguous=True))
            continue
        if m.group("y3"):
            year, month, day = int(m.group("y3")), int(m.group("m3")), int(m.group("d3"))
        else:
            year = int(m.group("y1") or m.group("y2"))
            month = _MONTHS[(m.group("m1") or m.group("m2")).rstrip(".").lower()]
            day = int(m.group("d1") or m.group("d2"))
        try:
            value = dt.date(year, month, day)
        except ValueError:
            value = None
        found.append(DateMention(value, m.group(0), m.start()))
    return found


def _parse_number(s: str) -> Decimal | None:
    """'1,028.40' and '1.028,40' are both 1028.40; Indian '1,41,600.00' is 141600.00.
    More than two decimals is not money."""
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d{1,2})?"
                    r"|\d+(?:\.\d{1,2})?", s):
        return Decimal(s.replace(",", ""))
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+,\d{1,2}", s):
        return Decimal(s.replace(".", "").replace(",", "."))
    return None


def format_amount(amount: Decimal, currency: str) -> str:
    """The suggested way to write an amount: '£1,028.40', 'CHF 1,028.40', '¥5,000'."""
    places = 0 if currency in ZERO_DECIMAL_CURRENCIES else 2
    number = f"{amount:,.{places}f}"
    symbols = CURRENCY_SYMBOLS.get(currency, ())
    return f"{symbols[0]}{number}" if symbols else f"{currency} {number}"


def format_date(value: dt.date) -> str:
    return f"{value.day} {value:%B %Y}"


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def normalize_name(name: str) -> str:
    """Case, punctuation and spacing don't matter: "Harbor Lane Printing Ltd." ~ "harbor lane printing ltd"."""
    return " ".join(re.sub(r"[^\w\s]", " ", name.casefold()).split())


def same_name(a: str, b: str) -> bool:
    """Equal, or one is a whole-word part of the other: "Acme" ~ "Acme Ltd", but "Lim" !~ "Climate"."""
    a, b = normalize_name(a), normalize_name(b)
    return bool(a and b) and (f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def name_in_text(name: str, text: str) -> bool:
    n = normalize_name(name)
    return bool(n) and f" {n} " in f" {normalize_name(text)} "


def parse_iso_date(value: str) -> dt.date | None:
    """Strict YYYY-MM-DD, as the pipeline is asked to produce."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None
