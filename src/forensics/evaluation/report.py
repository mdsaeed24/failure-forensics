"""Writes reports/diagnosis_accuracy.md from evaluation metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any

from forensics.analyzer.taxonomy import DEFINITIONS, FailureType
from forensics.evaluation.metrics import NONE, STEP_LABELS, TYPE_LABELS, Metrics, Rate


def render(m: Metrics, config: dict[str, Any]) -> str:
    out: list[str] = []
    w = out.append
    cost = f"${m.total_cost:.2f}" + ("" if m.cost_complete else " (some prices unknown)")

    w("# Diagnosis accuracy\n")
    w(f"Evaluation `{config['eval_id']}`, started {config['started_at'][:16].replace('T', ' ')} UTC. "
      f"Pipeline model `{config['pipeline_model']}`, judge `{config['judge_model']}`. "
      f"{config['docs']} documents, up to {config['per_fault']} per fault, seed {config['seed']}. "
      f"Total cost {cost}.\n")

    w("## Headline\n")
    w(f"- **Right step: {_pct(m.step_accuracy)}** of {m.step_accuracy.n} planted failures that "
      f"actually broke the final output {_ci(m.step_accuracy)}.")
    w(f"- **Right step and failure type: {_pct(m.exact_accuracy)}** {_ci(m.exact_accuracy)}.")
    alarm_levels = Counter(r.diag_confidence for r in m.clean_good if r.diag_step)
    by_level = ", ".join(f"{n} {lvl}" for lvl in ("high", "medium", "low") if (n := alarm_levels[lvl]))
    w(f"- **False alarms: {_pct(m.false_alarms)}** of {m.false_alarms.n} clean runs with a correct "
      f"output were blamed on a step anyway {_ci(m.false_alarms)}"
      + (f"; by confidence: {by_level}." if by_level else "."))
    w(f"- **Natural failures:** {len(m.natural_failures)} clean run(s) were wrong with no fault "
      "planted (listed below).")
    if m.errors:
        w(f"- {len(m.errors)} run(s) could not be completed (API errors) and are excluded.")
    w("")
    w("Accuracy counts only runs where the planted fault *took*: it changed something and the "
      "final output was really wrong. A fault that the next step happened to undo is a "
      "non-event, not a miss.\n")

    w("## By fault\n")
    w("| fault | breaks | type | how | runs | applied | took | right step | right step+type | decided by |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for f in m.by_fault:
        deciders = ", ".join(f"{k} {v}" for k, v in f.decided_by.most_common()) or "-"
        w(f"| `{f.fault}` | {f.expected_step} | {f.expected_type} | {f.mechanism} | {f.runs} | "
          f"{f.applied} | {f.took} | {_frac(f.step_ok, f.took)} | {_frac(f.exact, f.took)} | {deciders} |")
    w("")

    w("## By what decided\n")
    w("| decided by | runs | right step+type |")
    w("|---|---|---|")
    for name, rate in m.by_decider.items():
        if rate.n:
            w(f"| {name} | {rate.n} | {_pct(rate)} |")
    w("\nDeterministic checks decide most extraction faults for free; classification and some "
      "summary faults need the judge.\n")

    w("## Precision and recall by failure type\n")
    w("Rows include clean correct runs, whose right answer is `none`.\n")
    w("| type | precision | recall |")
    w("|---|---|---|")
    for label in TYPE_LABELS:
        p, r = m.precision_recall[label]
        if p.n or r.n:
            w(f"| {label} | {_pct(p)} ({p.hits}/{p.n}) | {_pct(r)} ({r.hits}/{r.n}) |")
    w("")

    w("## Confusion matrix: failure type\n")
    w(_matrix(m.type_confusion, TYPE_LABELS, "planted ↓ / diagnosed →"))
    w("## Confusion matrix: step\n")
    w(_matrix(m.step_confusion, STEP_LABELS, "planted ↓ / diagnosed →"))

    w("## Is the analyzer's confidence meaningful?\n")
    w("| analyzer confidence | times it named a cause | right step+type |")
    w("|---|---|---|")
    for level, rate in m.analyzer_calibration.items():
        if rate.n:
            w(f"| {level} | {rate.n} | {_pct(rate)} |")
    w("\n(False alarms count against the level the analyzer gave them.)\n")

    w("## Is the pipeline model's self-reported confidence meaningful?\n")
    w("The pipeline model rates its own output 1-5. Only outputs it really produced count "
      "(clean runs and prompt faults). If confidence were informative, low ratings would "
      "come with more wrong outputs.\n")
    for step, levels in m.self_confidence.items():
        if not levels:
            continue
        w(f"**{step}**\n")
        w("| self-confidence | outputs | wrong |")
        w("|---|---|---|")
        for c, rate in levels.items():
            w(f"| {c} | {rate.n} | {_pct(rate)} ({rate.hits}) |")
        w("")

    w("## Natural failures (no fault planted)\n")
    if not m.natural_failures:
        w("None: every clean run matched the ground truth.\n")
    else:
        w("These are the pipeline's own mistakes. There's no label to score the diagnosis "
          "against; they are worth a human look.\n")
        w("| doc | what was wrong | analyzer says |")
        w("|---|---|---|")
        for r in m.natural_failures:
            said = f"{r.diag_step} / {r.diag_type} ({r.diag_confidence})" if r.diag_step else "no step-level failure"
            w(f"| {r.doc_id} | {'; '.join(r.discrepancies)} | {said} |")
        w("")

    misses = [r for r in m.took if (r.diag_step, r.diag_type) != (r.expected_step, r.expected_type)]
    w("## Misdiagnoses\n")
    if not misses:
        w("None.\n")
    else:
        w("| fault | doc | planted | diagnosed | change made | trace |")
        w("|---|---|---|---|---|---|")
        for r in misses:
            got = f"{r.diag_step or NONE} / {r.diag_type or NONE}"
            w(f"| `{r.fault}` | {r.doc_id} | {r.expected_step} / {r.expected_type} | {got} | "
              f"{r.detail} | `{(r.trace_id or '')[:8]}` |")
        w("")

    alarms = [r for r in m.clean_good if r.diag_step]
    if alarms:
        w("## False alarms\n")
        w("| doc | blamed | trace |")
        w("|---|---|---|")
        for r in alarms:
            w(f"| {r.doc_id} | {r.diag_step} / {r.diag_type} ({r.diag_confidence}) | `{(r.trace_id or '')[:8]}` |")
        w("")

    w("## Failure types\n")
    for t in FailureType:
        w(f"- **{t.value}**: {DEFINITIONS[t]}")
    w("")
    w("## Limitations\n")
    w("- The final-output score checks document type, parties, dates and amounts only. A "
      "summary that invents a claim in words (\"payable under the contractor's standard "
      "terms\") or garbles a sentence still scores as correct. So some \"false alarms\" "
      "may be real defects the scoring can't see, and a prompt fault can \"not take\" "
      "while still degrading the output. Check the false-alarm traces before trusting the rate.")
    w("- Hook faults edit outputs in clean, mechanical ways (a transposed digit, an invented "
      "name). Real model errors are often subtler, so deterministic-check accuracy here is "
      "likely an upper bound.")
    w("- Per-fault samples are small; read the intervals, not just the percentages.")
    w("- The judge is also an LLM, and its verdicts can vary between runs.")
    w("- Documents were generated from fact sheets: clean and well-formatted compared to "
      "real scans or emails.")
    return "\n".join(out) + "\n"


def _pct(rate: Rate) -> str:
    return "n/a" if rate.value is None else f"{rate.value:.0%}"


def _ci(rate: Rate) -> str:
    interval = rate.wilson()
    return "" if interval is None else f"(95% CI {interval[0]:.0%}–{interval[1]:.0%})"


def _frac(hits: int, n: int) -> str:
    return "-" if not n else f"{hits}/{n}"


def _matrix(confusion: dict[str, Counter], labels: list[str], corner: str) -> str:
    rows = [label for label in labels if sum(confusion.get(label, Counter()).values())]
    cols = [label for label in labels if any(confusion.get(r, Counter())[label] for r in rows)]
    if not rows:
        return "No data.\n"
    lines = [f"| {corner} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        cells = [str(confusion[r][c] or "·") if r != c else f"**{confusion[r][c]}**" for c in cols]
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
