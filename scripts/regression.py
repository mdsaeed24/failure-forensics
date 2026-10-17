"""Re-run eval cases and report which known issues are resolved.

    python scripts/regression.py                # open and regressed cases
    python scripts/regression.py --all          # also re-check resolved ones
    python scripts/regression.py --case case_002

Each case costs one pipeline run plus at most one judge call.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys

from forensics.config import load_settings
from forensics.feedback.eval_cases import Feedback
from forensics.feedback.regression import apply_result, check_case
from forensics.llm.client import LLMClient
from forensics.scoring.final_score import load_ground_truth
from forensics.tracing.setup import configure_tracing
from forensics.tracing.store import TraceStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="include resolved cases")
    parser.add_argument("--case", action="append", default=[], help="only this case (repeatable)")
    args = parser.parse_args()

    settings = load_settings()
    traces = TraceStore(settings.resolve(settings.paths.traces_dir))
    feedback = Feedback(traces, settings.resolve(settings.paths.eval_cases))
    cases = feedback.load_cases()
    if args.case:
        chosen = [c for c in cases if c.case_id in args.case]
    else:
        chosen = [c for c in cases if args.all or c.status != "resolved"]
    if not chosen:
        print("No eval cases to check." if not cases else "Every case is resolved (use --all to re-check).")
        return 0

    try:
        llm = LLMClient.from_settings(settings)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    truth = load_ground_truth(settings.resolve(settings.paths.ground_truth))
    batch_id = f"regression-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M}"
    provider = configure_tracing(settings)

    async def run_all():
        return await asyncio.gather(*(check_case(c, llm, settings, traces, truth, batch_id) for c in chosen))

    try:
        results = asyncio.run(run_all())
    finally:
        provider.shutdown()

    updated = {c.case_id: apply_result(c, r) for c, r in zip(chosen, results)}
    feedback.save_cases([updated.get(c.case_id, c) for c in cases])
    for before, result in zip(chosen, results):
        after = updated[before.case_id]
        mark = "✓" if result.passed else "✗"
        change = f"{before.status} -> {after.status}" if before.status != after.status else after.status
        print(f"{mark} {after.case_id}  {after.doc_id:<9} {after.root_cause_step}/{after.failure_type.value:<16} "
              f"{change:<20} {result.reason}")

    final = feedback.load_cases()
    resolved = sum(c.status == "resolved" for c in final)
    print(f"\nKnown issues resolved: {resolved}/{len(final)}"
          + (f"; regressed: {sum(c.status == 'regressed' for c in final)}" if any(c.status == "regressed" for c in final) else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
