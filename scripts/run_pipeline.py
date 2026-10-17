"""Run documents through the 4-step pipeline and print each step's typed output.

    python scripts/run_pipeline.py --doc data/samples/sample_invoice.md
    python scripts/run_pipeline.py                 # first document in data/documents
    python scripts/run_pipeline.py --limit 5       # first 5 documents
    python scripts/run_pipeline.py --all           # every document (costs money)

Every run is traced to traces/ (one JSON file per document, plus traces/index.sqlite).
Browse them with: python scripts/list_traces.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

from forensics.config import PROJECT_ROOT, Settings, load_settings
from forensics.llm.client import LLMClient
from forensics.pipeline.models import RawDocument
from forensics.pipeline.runner import PipelineOptions, PipelineResult, StepRecord, run_pipeline
from forensics.tracing.setup import configure_tracing
from forensics.tracing.store import TraceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", type=Path, action="append", help="a document to run (repeatable)")
    parser.add_argument("--dir", type=Path, help="folder of *.md documents (default: data/documents)")
    parser.add_argument("--limit", type=int, default=1, help="max documents to run (default: 1)")
    parser.add_argument("--all", action="store_true", help="ignore --limit and run every document")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every LLM call")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings()

    paths = args.doc or sorted(settings.resolve(args.dir or settings.paths.documents_dir).glob("*.md"))
    if not args.all:
        paths = paths[: args.limit]
    if not paths:
        print("No documents found. Phase 2 generates them; for now try:\n"
              "  python scripts/run_pipeline.py --doc data/samples/sample_invoice.md", file=sys.stderr)
        return 1

    try:
        llm = LLMClient.from_settings(settings)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    provider = configure_tracing(settings)
    batch_id = uuid.uuid4().hex[:12]
    try:
        results = asyncio.run(_run_all(paths, llm, settings, PipelineOptions(batch_id=batch_id)))
    finally:
        provider.shutdown()
    store = TraceStore(settings.resolve(settings.paths.traces_dir))
    for result in results:
        _print_result(result)
        if result.trace_id:
            print(f"Trace: {store.path_of(result.trace_id).relative_to(PROJECT_ROOT)}")
    if len(results) > 1:
        ok = sum(r.status == "ok" for r in results)
        print(f"\n{ok}/{len(results)} documents completed. Total cost: {_cost(_sum_cost(results))}")
        print(f"Batch {batch_id}: python scripts/list_traces.py --batch {batch_id}")
    return 0 if all(r.status == "ok" for r in results) else 2


async def _run_all(paths: list[Path], llm: LLMClient, settings: Settings,
                   options: PipelineOptions) -> list[PipelineResult]:
    docs = [RawDocument.from_file(p) for p in paths]
    # Documents run concurrently; the client's semaphore caps requests in flight.
    return await asyncio.gather(*(run_pipeline(d, llm, settings, options) for d in docs))


def _print_result(result: PipelineResult) -> None:
    print(f"\n{'=' * 72}\n{result.doc_id}   status: {result.status}")
    outputs = {
        "intake": result.intake,
        "extraction": result.extraction,
        "classification": result.classification,
        "summarization": result.summary,
    }
    for i, record in enumerate(result.steps, start=1):
        print(f"\n[{i}] {record.step:<15} {_step_stats(record)}")
        if record.status == "error":
            print(f"    FAILED: {record.error}")
            continue
        output = outputs[record.step]
        exclude = {"text"} if record.step == "intake" else None  # the full text is the input file
        print(f"    {type(output).__name__}" + (" (text omitted)" if exclude else ""))
        print(_indent(json.dumps(output.model_dump(mode="json", exclude=exclude), indent=2, ensure_ascii=False)))
    print(f"\nTotal: {result.total_tokens} tokens, {_cost(result.total_cost_usd)}")


def _step_stats(record: StepRecord) -> str:
    parts = [f"{record.duration_ms:,.0f} ms"]
    for call in record.llm_calls:
        parts.append(
            f"model={call.model} tokens={call.prompt_tokens}→{call.completion_tokens}"
            f" ({call.reasoning_tokens} reasoning) "
            f"cost={_cost(call.cost_usd)} attempts={call.attempts} finish={call.finish_reason}"
            + (" REPAIRED" if call.repaired else "")
            + (" TRUNCATED" if call.truncated else "")
        )
    return "  ".join(parts)


def _sum_cost(results: list[PipelineResult]) -> float | None:
    costs = [r.total_cost_usd for r in results]
    return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]


def _cost(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


if __name__ == "__main__":
    sys.exit(main())
