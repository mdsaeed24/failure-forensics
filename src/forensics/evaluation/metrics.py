"""Accuracy numbers from evaluation records. Plain functions over lists, no LLM.

Counted runs:
- accuracy: injected runs where the fault applied AND the final output was bad;
- false alarms: clean runs whose final output was good (the analyzer should find nothing);
- natural failures: clean runs whose output was bad anyway (no label; listed for review).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from forensics.analyzer.taxonomy import FailureType
from forensics.evaluation.labels import RunRecord
from forensics.pipeline.runner import STEP_ORDER

NONE = "none"
TYPE_LABELS = [t.value for t in FailureType] + [NONE]
STEP_LABELS = [*STEP_ORDER, NONE]


@dataclass
class Rate:
    hits: int
    n: int

    @property
    def value(self) -> float | None:
        return self.hits / self.n if self.n else None

    def wilson(self, z: float = 1.96) -> tuple[float, float] | None:
        """95% interval: with small n the point estimate alone overstates certainty."""
        if not self.n:
            return None
        p, n = self.hits / self.n, self.n
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class FaultRow:
    fault: str
    expected_step: str
    expected_type: str
    mechanism: str
    runs: int = 0
    applied: int = 0
    took: int = 0
    step_ok: int = 0
    exact: int = 0
    decided_by: Counter = field(default_factory=Counter)


@dataclass
class Metrics:
    took: list[RunRecord]
    clean_good: list[RunRecord]
    natural_failures: list[RunRecord]
    errors: list[RunRecord]
    step_accuracy: Rate
    exact_accuracy: Rate
    false_alarms: Rate
    by_fault: list[FaultRow]
    type_confusion: dict[str, Counter]
    step_confusion: dict[str, Counter]
    precision_recall: dict[str, tuple[Rate, Rate]]
    analyzer_calibration: dict[str, Rate]
    by_decider: dict[str, Rate]
    self_confidence: dict[str, dict[int, Rate]]  # step -> confidence -> rate of WRONG outputs
    total_cost: float
    cost_complete: bool  # False if some model had no configured price


def _step_rank(step: str) -> int:
    return STEP_ORDER.index(step) if step in STEP_ORDER else len(STEP_ORDER)


def _exact(r: RunRecord) -> bool:
    return r.diag_step == r.expected_step and r.diag_type == r.expected_type


def compute(records: list[RunRecord]) -> Metrics:
    ok = [r for r in records if r.error is None]
    took = [r for r in ok if r.took_effect]
    clean = [r for r in ok if r.kind == "clean"]
    clean_good = [r for r in clean if r.final_good]
    natural = [r for r in clean if r.final_good is False]

    rows: dict[str, FaultRow] = {}
    for r in ok:
        if r.kind != "injected":
            continue
        row = rows.setdefault(r.fault, FaultRow(r.fault, r.expected_step or "", r.expected_type or "", r.mechanism))
        row.runs += 1
        row.applied += r.applied
        if r.took_effect:
            row.took += 1
            row.step_ok += r.diag_step == r.expected_step
            row.exact += _exact(r)
            row.decided_by[r.decided_by or NONE] += 1

    # Type labels: planted faults that took, plus clean good runs whose answer is "none".
    pairs = [(r.expected_type or NONE, r.diag_type or NONE) for r in took]
    pairs += [(NONE, r.diag_type or NONE) for r in clean_good]
    type_confusion: dict[str, Counter] = defaultdict(Counter)
    for expected, predicted in pairs:
        type_confusion[expected][predicted] += 1
    precision_recall = {}
    for label in TYPE_LABELS:
        tp = sum(1 for e, p in pairs if e == p == label)
        precision_recall[label] = (Rate(tp, sum(1 for _, p in pairs if p == label)),
                                   Rate(tp, sum(1 for e, _ in pairs if e == label)))

    step_confusion: dict[str, Counter] = defaultdict(Counter)
    for r in took:
        step_confusion[r.expected_step or NONE][r.diag_step or NONE] += 1

    calibration: dict[str, Rate] = {}
    for level in ("high", "medium", "low"):
        named = [r for r in took if r.diag_confidence == level]
        alarms = [r for r in clean_good if r.diag_confidence == level]
        calibration[level] = Rate(sum(_exact(r) for r in named), len(named) + len(alarms))

    by_decider: dict[str, Rate] = {}
    for decider in ("deterministic", "judge", "crash"):
        group = [r for r in took if r.decided_by == decider]
        by_decider[decider] = Rate(sum(_exact(r) for r in group), len(group))

    # Is the pipeline model's own confidence informative? Only outputs it really produced
    # count: clean runs and prompt faults (a hook rewrites output but keeps its confidence).
    genuine = [r for r in ok if r.kind == "clean" or r.mechanism == "prompt"]
    self_confidence: dict[str, dict[int, Rate]] = {}
    for step, conf, correct in (("extraction", "extraction_confidence", "extraction_correct"),
                                ("classification", "classification_confidence", "classification_correct")):
        levels: dict[int, list[bool]] = defaultdict(list)
        for r in genuine:
            c, right = getattr(r, conf), getattr(r, correct)
            if c is not None and right is not None:
                levels[c].append(not right)
        self_confidence[step] = {c: Rate(sum(v), len(v)) for c, v in sorted(levels.items())}

    costs = [r.pipeline_cost for r in ok] + [r.analysis_cost for r in ok if r.analyzed]
    return Metrics(
        took=took, clean_good=clean_good, natural_failures=natural,
        errors=[r for r in records if r.error],
        step_accuracy=Rate(sum(r.diag_step == r.expected_step for r in took), len(took)),
        exact_accuracy=Rate(sum(_exact(r) for r in took), len(took)),
        false_alarms=Rate(sum(1 for r in clean_good if r.diag_step), len(clean_good)),
        by_fault=sorted(rows.values(), key=lambda f: (_step_rank(f.expected_step), f.fault)),
        type_confusion=type_confusion, step_confusion=step_confusion,
        precision_recall=precision_recall, analyzer_calibration=calibration,
        by_decider=by_decider, self_confidence=self_confidence,
        total_cost=sum(c for c in costs if c is not None),
        cost_complete=None not in costs,
    )
