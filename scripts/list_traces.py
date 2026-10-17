"""Browse recorded traces (reads traces/index.sqlite; no API calls).

    python scripts/list_traces.py                    # 20 most recent runs
    python scripts/list_traces.py --doc inv_001      # runs of one document
    python scripts/list_traces.py --status error
    python scripts/list_traces.py --batch 3f9c2a1b7e4d
    python scripts/list_traces.py --show 4bf92f      # per-step signals of one trace (id prefix)
"""

from __future__ import annotations

import argparse
import sys

from forensics.config import PROJECT_ROOT, load_settings
from forensics.tracing.store import TraceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", help="only this doc_id")
    parser.add_argument("--status", choices=["ok", "error"])
    parser.add_argument("--batch", help="only this batch_id")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--show", metavar="TRACE_ID", help="show one trace's steps (a unique prefix is enough)")
    args = parser.parse_args()

    settings = load_settings()
    store = TraceStore(settings.resolve(settings.paths.traces_dir))

    if args.show:
        try:
            trace_id = store.resolve_id(args.show)
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 1
        row = store.get(trace_id)
        print(f"trace {trace_id}  doc {row['doc_id']}  status {row['status']}"
              + (f" (failed at {row['failed_step']})" if row["failed_step"] else ""))
        print(f"file  {store.path_of(trace_id).relative_to(PROJECT_ROOT)}\n")
        print(f"{'#':<2} {'step':<15} {'status':<7} {'ms':>8} {'conf':>4} {'tokens':>7} "
              f"{'reason':>6} {'cost':>9} {'att':>3} {'flags':<10} signals")
        for s in store.steps(trace_id):
            flags = " ".join(f for f, on in (("TRUNC", s["truncated"]), ("REPAIR", s["repaired"])) if on)
            signals = ", ".join(f"{k}={v}" for k, v in s["signals"].items() if k != "confidence")
            print(f"{s['step_index']:<2} {s['step']:<15} {s['status']:<7} {s['duration_ms']:>8,.0f} "
                  f"{_v(s['confidence']):>4} {s['tokens']:>7} {s['reasoning_tokens']:>6} "
                  f"{_cost(s['cost_usd']):>9} {s['attempts']:>3} {flags:<10} {signals}")
            if s["error"]:
                print(f"   error: {s['error']}")
        return 0

    rows = store.list_traces(doc_id=args.doc, status=args.status, batch_id=args.batch, limit=args.limit)
    if not rows:
        print("No traces yet. Run: python scripts/run_pipeline.py --doc data/documents/inv_001.md")
        return 0
    print(f"{'trace':<10} {'started (UTC)':<20} {'doc_id':<16} {'status':<7} {'failed at':<15} "
          f"{'ms':>7} {'tokens':>7} {'cost':>9}")
    for r in rows:
        print(f"{r['trace_id'][:8]:<10} {r['started_at'][:19].replace('T', ' '):<20} {r['doc_id']:<16} "
              f"{r['status']:<7} {r['failed_step'] or '':<15} {r['duration_ms']:>7,.0f} "
              f"{r['total_tokens']:>7} {_cost(r['cost_usd']):>9}")
    return 0


def _v(value: object) -> str:
    return "" if value is None else str(value)


def _cost(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.5f}"


if __name__ == "__main__":
    sys.exit(main())
