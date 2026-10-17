"""Human-readable views: the verification report and the side-by-side review screen."""

from __future__ import annotations

import datetime as dt
import hashlib
import textwrap
from itertools import zip_longest

from forensics.documents.fact_sheets import FactSheet
from forensics.documents.store import DocumentStore, ManifestEntry
from forensics.documents.verifier import check_deterministic
from forensics.textscan import format_amount


def verification_report(store: DocumentStore, sheets: list[FactSheet]) -> str:
    rows, details = [], []
    counts = {"verified": 0, "failed": 0, "approved": 0, "not generated": 0, "stale": 0}
    for sheet in sheets:
        entry = store.entries.get(sheet.doc_id)
        if entry is None:
            counts["not generated"] += 1
            rows.append(f"| {sheet.doc_id} | {sheet.doc_type} | not generated | | | | | |")
            continue
        stale = not store.is_current(sheet)
        counts["stale" if stale else entry.status] += 1
        counts["approved"] += entry.review.status == "approved" and not stale
        v = entry.verification
        rows.append(
            f"| {sheet.doc_id} | {sheet.doc_type} | {'stale' if stale else entry.status} "
            f"| {entry.attempts} | {len(v.errors)} | {len(v.warnings)} | {entry.review.status} "
            f"| {_cost(entry.cost_usd)} |"
        )
        details.append(_detail(sheet, entry))

    total_cost = [e.cost_usd for e in store.entries.values()]
    lines = [
        "# Document verification report",
        "",
        f"Generated {dt.datetime.now(dt.UTC):%Y-%m-%d %H:%M} UTC. "
        f"{len(sheets)} fact sheets: {counts['verified']} verified, {counts['failed']} failed "
        f"verification, {counts['stale']} stale, {counts['not generated']} not generated. "
        f"{counts['approved']} approved by a reviewer and in the ground truth.",
        "",
        f"Total generation cost so far: "
        f"{_cost(None if None in total_cost else sum(total_cost))}",  # type: ignore[arg-type]
        "",
        "| doc_id | type | status | attempts | errors | warnings | review | cost |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        *details,
    ]
    return "\n".join(lines)


def _detail(sheet: FactSheet, entry: ManifestEntry) -> str:
    v = entry.verification
    out = [f"## {sheet.doc_id}: {entry.status} after {entry.attempts} attempt(s)", ""]
    for n, problems in enumerate(entry.rejected_attempts, start=1):
        out.append(f"Attempt {n} rejected:")
        out += [f"- {p}" for p in problems]
        out.append("")
    out.append(f"Final attempt: {len(v.found)} fact(s) found.")
    out += [f"- **{i.severity}** ({i.check}): {i.message}" for i in v.issues]
    if v.audit:
        out.append(f"- auditor reads it as **{v.audit.doc_type}**: {v.audit.doc_type_reason}")
    if entry.review.note:
        out.append(f"- reviewer ({entry.review.status}): {entry.review.note}")
    out.append("")
    return "\n".join(out)


# --- side-by-side review screen ------------------------------------------------

_GREEN, _RED, _YELLOW, _BOLD, _DIM, _RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[2m", "\033[0m"
)


def side_by_side(
    sheet: FactSheet, entry: ManifestEntry, text: str, width: int = 120, color: bool = True
) -> str:
    """Document on the left with facts highlighted, fact sheet and checks on the right.

    Presence is re-checked against the text on disk, so the screen is right even if
    the .md file was edited after generation.
    """
    def c(code: str, s: str) -> str:
        return f"{code}{s}{_RESET}" if color else s

    issues, found = check_deterministic(sheet, text)
    right_w = max(36, min(56, width // 2 - 4))
    left_w = max(40, width - right_w - 3)

    left: list[tuple[str, str]] = []  # (plain padded line, display line)
    surfaces = sorted({f.surface for f in found}, key=len, reverse=True)
    for n, line in enumerate(text.rstrip("\n").split("\n"), start=1):
        pieces = textwrap.wrap(line, left_w - 5) or [""]
        for k, piece in enumerate(pieces):
            plain = (f"{n:>3}  " if k == 0 else "     ") + piece
            plain = plain[:left_w].ljust(left_w)
            shown = plain
            for s in surfaces:
                shown = shown.replace(s, c(_GREEN + _BOLD, s))
            left.append((plain, c(_DIM, shown[:5]) + shown[5:] if color else shown))

    hits = {(f.kind, f.index): f for f in found}
    right: list[str] = []

    def add(s: str = "", code: str | None = None, indent: int = 0) -> None:
        for piece in textwrap.wrap(s, right_w - indent) or [""]:
            right.append((" " * indent) + (c(code, piece) if code else piece))

    def fact(kind: str, i: int, label: str) -> None:
        hit = hits.get((kind, i))
        add(("✓ " if hit else "✗ ") + label, _GREEN if hit else _RED)
        if hit:
            add(f'"{hit.surface}" · line {hit.line}', _DIM, indent=4)

    stale = entry.doc_sha256 != _sha(text)
    add(f"{sheet.doc_id} · {sheet.doc_type}", _BOLD)
    add(f"generated: {entry.status}, {entry.attempts} attempt(s)")
    add(f"review: {entry.review.status}" + (f" ({entry.review.note})" if entry.review.note else ""))
    if stale:
        add("document edited since generation", _YELLOW)
    add()
    add("PARTIES", _BOLD)
    for i, p in enumerate(sheet.parties):
        fact("party", i, f"{p.name} ({p.role})")
    if sheet.dates:
        add("DATES", _BOLD)
        for i, d in enumerate(sheet.dates):
            fact("date", i, f"{d.date.isoformat()}  {d.label}")
    if sheet.amounts:
        add("AMOUNTS", _BOLD)
        for i, a in enumerate(sheet.amounts):
            fact("amount", i, f"{format_amount(a.amount, a.currency)}  {a.label}")
    add()
    add("CHECKS", _BOLD)
    extras = [i for i in issues if i.check != "missing_fact"]
    if not extras:
        add("✓ no extra or ambiguous dates/amounts", _GREEN)
    for issue in extras:
        add(f"✗ {issue.message}", _RED)
    audit = entry.verification.audit
    if audit:
        ok = audit.doc_type == sheet.doc_type
        add(f"{'✓' if ok else '✗'} auditor reads it as {audit.doc_type}", _GREEN if ok else _RED)
        add(audit.doc_type_reason, _DIM, indent=4)
    for issue in entry.verification.issues:
        if issue.check in ("extra_name", "doc_type") or issue.severity == "warning":
            add(f"{'!' if issue.severity == 'warning' else '✗'} {issue.message}",
                _YELLOW if issue.severity == "warning" else _RED)

    rows = []
    for l, r in zip_longest(left, right, fillvalue=None):
        shown = l[1] if l else " " * left_w
        rows.append(f"{shown} │ {r or ''}")
    return "\n".join(rows)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cost(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"
