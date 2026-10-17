"""Find the root cause of a bad pipeline run.

    python scripts/analyze_trace.py 368da3          # a trace (a unique id prefix is enough)
    python scripts/analyze_trace.py --latest
    python scripts/analyze_trace.py 368da3 --force  # analyze even if the output looks right

If the document is in the ground truth, the final output is scored first and only a
bad run is analyzed (the judge costs money). Documents without ground truth are
always analyzed, as a human flag would be in production.

The diagnosis is saved to traces/diagnoses/<trace_id>.json and the index.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from forensics.analyzer.root_cause import Diagnosis, analyze
from forensics.config import PROJECT_ROOT, load_settings
from forensics.llm.client import LLMClient
from forensics.scoring.final_score import load_ground_truth, score_trace
from forensics.tracing.store import TraceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", nargs="?", help="trace id or unique prefix")
    parser.add_argument("--latest", action="store_true", help="the most recent trace")
    parser.add_argument("--force", action="store_true", help="analyze even if the output matches ground truth")
    args = parser.parse_args()
    if not args.trace and not args.latest:
        parser.error("give a trace id, or --latest")

    settings = load_settings()
    store = TraceStore(settings.resolve(settings.paths.traces_dir))
    try:
        if args.latest:
            rows = store.list_traces(limit=1)
            if not rows:
                print("No traces yet.", file=sys.stderr)
                return 1
            trace_id = rows[0]["trace_id"]
        else:
            trace_id = store.resolve_id(args.trace)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1
    trace_doc = store.load(trace_id)
    print(f"trace {trace_id}  doc {trace_doc['doc_id']}  status {trace_doc['status']}")

    symptom: list[str] = []
    truth = load_ground_truth(settings.resolve(settings.paths.ground_truth)).get(trace_doc["doc_id"])
    if truth is None:
        print("No ground truth for this document: analyzing as a flagged run.")
    else:
        score = score_trace(trace_doc, truth)
        symptom = [d.message for d in score.discrepancies]
        if score.good:
            print("Final output matches the ground truth.")
            if not args.force:
                print("Nothing to analyze (use --force to analyze anyway).")
                return 0
        else:
            print(f"Final output is BAD ({len(symptom)} problem(s)):")
            for line in symptom:
                print(f"  - {line}")

    try:
        llm = LLMClient.from_settings(settings)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"\nAnalyzing with {settings.llm.judge_model}...")
    diagnosis = asyncio.run(analyze(trace_doc, llm, settings, symptom))
    path = store.save_diagnosis(diagnosis.model_dump(mode="json"))
    _print(diagnosis)
    print(f"\nSaved: {path.relative_to(PROJECT_ROOT)}")
    return 0


def _print(d: Diagnosis) -> None:
    print(f"\n{d.headline}\n")
    print("Evidence chain:")
    for line in d.evidence:
        print(f"  {line}")
    cost = "unknown" if d.cost_usd is None else f"${d.cost_usd:.4f}"
    print(f"\nJudge calls: {d.judge_calls}, analysis cost: {cost}")


if __name__ == "__main__":
    sys.exit(main())
