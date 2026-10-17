"""Measure the analyzer: clean runs + planted faults -> reports/diagnosis_accuracy.md

    python scripts/evaluate_diagnosis.py --docs 4 --per-fault 1     # smoke test (~2 cents)
    python scripts/evaluate_diagnosis.py --per-fault 5              # full evaluation
    python scripts/evaluate_diagnosis.py --resume                   # finish an interrupted one
    python scripts/evaluate_diagnosis.py --resume --redo shift_date # rerun one fault after a fix
    python scripts/evaluate_diagnosis.py --report-only              # rebuild the latest report

Uses only documents in data/ground_truth.json. Fault labels go to
traces/evaluation.sqlite, which the analyzer never reads.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import random
import sys
import uuid

from forensics.config import PROJECT_ROOT, load_settings
from forensics.evaluation.labels import EvalStore, RunRecord
from forensics.evaluation.metrics import compute
from forensics.evaluation.report import render
from forensics.evaluation.runner import EvalDoc, run_evaluation
from forensics.faults.registry import FAULTS, get_fault
from forensics.llm.client import LLMClient
from forensics.scoring.final_score import load_ground_truth
from forensics.tracing.setup import configure_tracing
from forensics.tracing.store import TraceStore

# Rough per-run costs seen so far, for the estimate printed before starting.
_PIPELINE_RUN = 0.0013
_ANALYSIS_CLEAN = 0.0015
_ANALYSIS_INJECTED = 0.0008


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, help="use a random sample of N documents (default: all)")
    parser.add_argument("--per-fault", type=int, default=5, help="documents per fault (default: 5)")
    parser.add_argument("--faults", help="comma-separated fault names (default: all)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=8, help="API requests in flight (default: 8)")
    parser.add_argument("--resume", nargs="?", const="latest", metavar="EVAL_ID",
                        help="continue an evaluation (default: the latest)")
    parser.add_argument("--redo", metavar="FAULT", action="append", default=[],
                        help="with --resume: discard and rerun this fault's runs (repeatable)")
    parser.add_argument("--report-only", nargs="?", const="latest", metavar="EVAL_ID",
                        help="rebuild the report without running anything")
    parser.add_argument("-y", "--yes", action="store_true", help="don't ask before spending")
    args = parser.parse_args()

    settings = load_settings()
    labels = EvalStore(settings.resolve(settings.paths.traces_dir) / "evaluation.sqlite")
    report_path = settings.resolve(settings.paths.reports_dir) / "diagnosis_accuracy.md"

    if args.report_only:
        eval_id = labels.latest() if args.report_only == "latest" else args.report_only
        if eval_id is None:
            print("No evaluation has been run yet.", file=sys.stderr)
            return 1
        return _write_report(labels, eval_id, report_path)

    if args.resume:
        eval_id = labels.latest() if args.resume == "latest" else args.resume
        if eval_id is None:
            print("Nothing to resume.", file=sys.stderr)
            return 1
        config = labels.config(eval_id)
        for fault in args.redo:
            print(f"Discarded {labels.forget_fault(eval_id, get_fault(fault).name)} run(s) of {fault}.")
        args.docs, args.per_fault, args.seed = config["docs_requested"], config["per_fault"], config["seed"]
        args.faults = ",".join(config["faults"])
    else:
        eval_id = f"eval-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M}-{uuid.uuid4().hex[:4]}"

    truth = load_ground_truth(settings.resolve(settings.paths.ground_truth))
    docs_dir = settings.resolve(settings.paths.documents_dir)
    docs = [EvalDoc(d, (docs_dir / f"{d}.md").read_text(encoding="utf-8"), t) for d, t in sorted(truth.items())]
    if args.docs:
        docs = sorted(random.Random(args.seed).sample(docs, min(args.docs, len(docs))), key=lambda d: d.doc_id)
    try:
        faults = [get_fault(n.strip()) for n in args.faults.split(",")] if args.faults else list(FAULTS.values())
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    injected = sum(min(args.per_fault, len(docs)) for _ in faults)
    estimate = len(docs) * (_PIPELINE_RUN + _ANALYSIS_CLEAN) + injected * (_PIPELINE_RUN + _ANALYSIS_INJECTED)
    print(f"{eval_id}: {len(docs)} clean runs + up to {injected} injected runs "
          f"({len(faults)} faults x {args.per_fault}). Estimated cost ~${estimate:.2f}.")
    if not args.yes and not args.resume and input("Start? [y/N] ").strip().lower() != "y":
        return 0

    settings = settings.model_copy(update={"llm": settings.llm.model_copy(update={"concurrency": args.concurrency})})
    try:
        llm = LLMClient.from_settings(settings)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    labels.start(eval_id, dt.datetime.now(dt.UTC).isoformat(), {
        "eval_id": eval_id, "docs": len(docs), "docs_requested": args.docs, "per_fault": args.per_fault,
        "seed": args.seed, "faults": [f.name for f in faults],
        "pipeline_model": settings.llm.pipeline_model, "judge_model": settings.llm.judge_model,
    })
    traces = TraceStore(settings.resolve(settings.paths.traces_dir))
    provider = configure_tracing(settings)
    try:
        records = asyncio.run(run_evaluation(eval_id, docs, faults, args.per_fault, args.seed,
                                             llm, settings, traces, labels, _progress))
    finally:
        provider.shutdown()
    errors = sum(1 for r in records if r.error)
    if errors:
        print(f"\n{errors} run(s) hit API errors; finish them with: python scripts/evaluate_diagnosis.py --resume")
    return _write_report(labels, eval_id, report_path)


_count = 0


def _progress(r: RunRecord) -> None:
    global _count
    _count += 1
    label = r.fault or "clean"
    if r.error:
        outcome = f"ERROR {r.error[:80]}"
    elif r.kind == "clean":
        said = f"{r.diag_step}/{r.diag_type}" if r.diag_step else "nothing"
        outcome = f"{'good' if r.final_good else 'BAD '}  analyzer blamed: {said}"
    elif not r.applied:
        outcome = "not applied"
    elif not r.took_effect:
        outcome = "applied, did not take (output still correct)"
    else:
        right = (r.diag_step, r.diag_type) == (r.expected_step, r.expected_type)
        outcome = (f"took -> {r.diag_step}/{r.diag_type} [{r.diag_confidence}, {r.decided_by}] "
                   f"{'RIGHT' if right else f'WRONG (planted {r.expected_step}/{r.expected_type})'}")
    print(f"[{_count:>3}] {label:<24} {r.doc_id:<9} {outcome}", flush=True)


def _write_report(labels: EvalStore, eval_id: str, path) -> int:
    config = labels.config(eval_id)
    metrics = compute(labels.records(eval_id))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(metrics, config), encoding="utf-8")
    print(f"\nReport: {path.relative_to(PROJECT_ROOT)}")
    print(f"Right step {metrics.step_accuracy.hits}/{metrics.step_accuracy.n}, "
          f"right step+type {metrics.exact_accuracy.hits}/{metrics.exact_accuracy.n}, "
          f"false alarms {metrics.false_alarms.hits}/{metrics.false_alarms.n}, "
          f"cost ${metrics.total_cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
