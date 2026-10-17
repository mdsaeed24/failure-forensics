"""Review generated documents side by side with their fact sheets: approve or reject.

    python scripts/review_documents.py                    # step through documents awaiting review
    python scripts/review_documents.py --list
    python scripts/review_documents.py --show inv_001
    python scripts/review_documents.py --approve inv_001 rem_001
    python scripts/review_documents.py --reject inv_001 --note "reads like a quotation"

Only approved documents enter data/ground_truth.json. A rejection note is passed to
the generator next time you run generate_documents.py. No API calls, no cost.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

from forensics.config import load_settings
from forensics.documents.fact_sheets import FactSheet, FactSheetError, load_fact_sheets
from forensics.documents.report import side_by_side
from forensics.documents.store import DocumentStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="list every document and its state")
    parser.add_argument("--show", metavar="DOC_ID", help="show one document, no prompt")
    parser.add_argument("--approve", nargs="+", metavar="DOC_ID")
    parser.add_argument("--reject", metavar="DOC_ID")
    parser.add_argument("--note", help="why it was rejected (required with --reject)")
    parser.add_argument("--include-failed", action="store_true",
                        help="also step through documents that failed verification")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    try:
        sheets = {s.doc_id: s for s in load_fact_sheets(settings.resolve(settings.paths.fact_sheets))}
    except FactSheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    store = DocumentStore(settings.resolve(settings.paths.documents_dir),
                          settings.resolve(settings.paths.ground_truth))
    color = sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    width = shutil.get_terminal_size((120, 40)).columns

    def require(doc_id: str) -> FactSheet | None:
        if doc_id not in store.entries or doc_id not in sheets:
            print(f"{doc_id}: no generated document with a current fact sheet", file=sys.stderr)
            return None
        return sheets[doc_id]

    if args.list:
        for doc_id, sheet in sheets.items():
            entry = store.entries.get(doc_id)
            if entry is None:
                print(f"  {doc_id:<20} not generated")
                continue
            state = "stale" if not store.is_current(sheet) else entry.status
            print(f"  {doc_id:<20} {state:<9} review: {entry.review.status:<9}"
                  + (f" note: {entry.review.note}" if entry.review.note else ""))
        return 0

    if args.show:
        sheet = require(args.show)
        if sheet is None:
            return 1
        print(_screen(store, sheet, width, color))
        return 0

    if args.approve or args.reject:
        if args.reject and not args.note:
            print("--reject needs --note: the note tells the generator what to fix.", file=sys.stderr)
            return 1
        decisions = [(d, "approved", None) for d in args.approve or []]
        if args.reject:
            decisions.append((args.reject, "rejected", args.note))
        for doc_id, status, note in decisions:
            if require(doc_id) is None:
                return 1
            store.set_review(doc_id, status, note)  # type: ignore[arg-type]
            print(f"{doc_id}: {status}")
        return _finish(store, list(sheets.values()))

    queue = [
        s for d, s in sheets.items()
        if d in store.entries and store.is_current(s)
        and store.entries[d].review.status == "pending"
        and (store.entries[d].status == "verified" or args.include_failed)
    ]
    if not queue:
        print("Nothing awaiting review. (--list shows every document.)")
        return 0
    for n, sheet in enumerate(queue, start=1):
        print("\n" + _screen(store, sheet, width, color))
        print(f"\n[{n}/{len(queue)}] {sheet.doc_id}")
        answer = _ask("[a]pprove  [r]eject  [s]kip  [q]uit > ", "arsq")
        if answer == "q":
            break
        if answer == "a":
            store.set_review(sheet.doc_id, "approved")
        elif answer == "r":
            note = ""
            while not note:
                note = input("What's wrong? (passed to the generator) > ").strip()
            store.set_review(sheet.doc_id, "rejected", note)
    return _finish(store, list(sheets.values()))


def _screen(store: DocumentStore, sheet: FactSheet, width: int, color: bool) -> str:
    text = store.read_document(sheet.doc_id) or ""
    return side_by_side(sheet, store.entries[sheet.doc_id], text, width=width, color=color)


def _ask(prompt: str, choices: str) -> str:
    while True:
        answer = input(prompt).strip().lower()[:1]
        if answer and answer in choices:
            return answer


def _finish(store: DocumentStore, sheets: list[FactSheet]) -> int:
    included, _ = store.build_ground_truth(sheets)
    print(f"Ground truth: {included} approved document(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
