"""Fact sheets → documents, each verified automatically and rewritten until it passes.

    python scripts/generate_documents.py                 # next document that needs generating
    python scripts/generate_documents.py --limit 3
    python scripts/generate_documents.py --doc-id inv_001 --doc-id rem_001
    python scripts/generate_documents.py --all

A document needs generating if it doesn't exist yet, its fact sheet changed, or a
reviewer rejected it (the rejection note is passed to the generator). Documents
that already failed verification are only retried when named with --doc-id.

Writes data/documents/*.md, data/documents/manifest.json, data/ground_truth.json
and reports/document_verification.md.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from forensics.config import PROJECT_ROOT, Settings, load_settings
from forensics.documents.fact_sheets import FactSheet, FactSheetError, load_fact_sheets
from forensics.documents.generator import GenerationOutcome, generate
from forensics.documents.report import verification_report
from forensics.documents.store import DocumentStore
from forensics.llm.client import LLMClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc-id", action="append", default=[], help="generate this document (repeatable)")
    parser.add_argument("--limit", type=int, default=1, help="max documents to generate (default: 1; ignored with --doc-id)")
    parser.add_argument("--all", action="store_true", help="ignore --limit")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every LLM call")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    settings = load_settings()
    if settings.generation is None:
        print("config.yaml has no 'generation' section.", file=sys.stderr)
        return 1
    try:
        sheets = load_fact_sheets(settings.resolve(settings.paths.fact_sheets))
    except FactSheetError as exc:
        print(exc, file=sys.stderr)
        return 1
    store = DocumentStore(settings.resolve(settings.paths.documents_dir),
                          settings.resolve(settings.paths.ground_truth))

    todo = _select(sheets, store, args.doc_id)
    if todo is None:
        return 1
    waiting = len(todo)
    if not args.all and not args.doc_id:  # documents named explicitly are all generated
        todo = todo[: args.limit]
    if not todo:
        print("Every document is up to date. Nothing to generate.")
    else:
        try:
            llm = LLMClient.from_settings(settings)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Generating {len(todo)} document(s) with {settings.generation.generator.model}...")
        asyncio.run(_generate_all(todo, store, llm, settings))
        if waiting > len(todo):
            print(f"{waiting - len(todo)} more need generating; use --limit N or --all.")

    included, _ = store.build_ground_truth(sheets)
    report_path = settings.resolve(settings.paths.reports_dir) / "document_verification.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(verification_report(store, sheets) + "\n", encoding="utf-8")
    print(f"\nReport: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"Ground truth: {included} approved document(s). "
          "Review new documents with: python scripts/review_documents.py")
    return 0


def _select(sheets: list[FactSheet], store: DocumentStore, doc_ids: list[str]) -> list[FactSheet] | None:
    by_id = {s.doc_id: s for s in sheets}
    if doc_ids:
        unknown = [d for d in doc_ids if d not in by_id]
        if unknown:
            print(f"Not in the fact sheets: {', '.join(unknown)}", file=sys.stderr)
            return None
        return [by_id[d] for d in doc_ids]
    todo = []
    for sheet in sheets:
        entry = store.entries.get(sheet.doc_id)
        if not store.is_current(sheet) or (entry and entry.review.status == "rejected"):
            todo.append(sheet)
    return todo


async def _generate_all(todo: list[FactSheet], store: DocumentStore, llm: LLMClient,
                        settings: Settings) -> None:
    assert settings.generation is not None
    gen = settings.generation

    async def one(sheet: FactSheet) -> tuple[FactSheet, GenerationOutcome]:
        entry = store.entries.get(sheet.doc_id)
        note = entry.review.note if entry and entry.review.status == "rejected" else None
        return sheet, await generate(sheet, llm, gen, reviewer_note=note)

    for next_done in asyncio.as_completed([one(s) for s in todo]):
        sheet, outcome = await next_done
        store.record(sheet, outcome, gen.generator.model)
        _print_outcome(outcome)


def _print_outcome(outcome: GenerationOutcome) -> None:
    cost = "unknown" if outcome.cost_usd is None else f"${outcome.cost_usd:.4f}"
    if outcome.status == "error":
        print(f"  ✗ {outcome.doc_id:<20} API error: {outcome.error}")
        return
    v = outcome.verification
    assert v is not None
    mark = "✓" if outcome.status == "verified" else "✗"
    print(f"  {mark} {outcome.doc_id:<20} {outcome.status} after {outcome.attempts} attempt(s), "
          f"{len(v.warnings)} warning(s), {cost}")
    for issue in v.errors:
        print(f"      error: {issue.message}")


if __name__ == "__main__":
    sys.exit(main())
