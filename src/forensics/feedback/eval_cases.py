"""The feedback loop: a human flags a bad trace, confirms or corrects the diagnosis,
and the confirmed failure becomes an eval case that regression re-checks.

    flag_trace()   open a flag on a trace, with a note saying what looks wrong
    confirm()      accept the analyzer's diagnosis, or override step/type -> EvalCase
    dismiss()      close a flag without a case (the output was fine after all)

Flags live in traces/index.sqlite next to the traces they point at. Eval cases live
in data/eval_cases.json, committed with the repo: they're the project's memory of
every failure a human has confirmed.

The UI and the HTTP API both call these functions, so the rules are the same
whichever way a flag comes in.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from forensics.analyzer.taxonomy import FailureType
from forensics.pipeline.runner import STEP_ORDER
from forensics.tracing.store import TraceStore


class FeedbackError(ValueError):
    """A request that can't be carried out (unknown trace, flag already closed, ...)."""


class Flag(BaseModel):
    trace_id: str
    doc_id: str
    note: str
    flagged_by: str = "anonymous"
    flagged_at: dt.datetime
    status: Literal["open", "confirmed", "dismissed"] = "open"
    closed_at: dt.datetime | None = None
    closing_note: str | None = None
    case_id: str | None = None


class RegressionResult(BaseModel):
    run_at: dt.datetime
    trace_id: str | None
    passed: bool
    reason: str


class EvalCase(BaseModel):
    case_id: str
    doc_id: str
    source_trace_id: str
    created_at: dt.datetime
    note: str  # what the human said was wrong
    root_cause_step: str
    failure_type: FailureType
    diagnosis_agreed: bool  # False: the human overrode the analyzer
    analyzer_said: str | None = None  # the analyzer's step/type, kept when overridden
    status: Literal["open", "resolved", "regressed"] = "open"
    history: list[RegressionResult] = []


_FLAGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS flags (
    trace_id TEXT PRIMARY KEY,
    doc_id   TEXT,
    status   TEXT,
    record   TEXT
)"""


class Feedback:
    def __init__(self, traces: TraceStore, cases_path: Path) -> None:
        self.traces = traces
        self.cases_path = cases_path

    # --- flags ---------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        self.traces.dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.traces.index_path)
        conn.execute(_FLAGS_SCHEMA)
        return conn

    def _save_flag(self, flag: Flag) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO flags VALUES (?, ?, ?, ?)",
                         (flag.trace_id, flag.doc_id, flag.status, flag.model_dump_json()))

    def get_flag(self, trace_id: str) -> Flag | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT record FROM flags WHERE trace_id = ?", (trace_id,)).fetchone()
        return Flag.model_validate_json(row[0]) if row else None

    def list_flags(self, status: str | None = None) -> list[Flag]:
        query, params = "SELECT record FROM flags", ()
        if status:
            query, params = query + " WHERE status = ?", (status,)
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return sorted((Flag.model_validate_json(r[0]) for r in rows),
                      key=lambda f: f.flagged_at, reverse=True)

    def flag_trace(self, trace_id: str, note: str, flagged_by: str = "anonymous") -> Flag:
        note = note.strip()
        if not note:
            raise FeedbackError("say what looks wrong: the note becomes part of the eval case")
        try:
            row = self.traces.get(trace_id)
        except KeyError:
            raise FeedbackError(f"no trace {trace_id!r}") from None
        existing = self.get_flag(trace_id)
        if existing and existing.status == "open":
            raise FeedbackError(f"trace {trace_id[:8]} is already flagged")
        flag = Flag(trace_id=trace_id, doc_id=row["doc_id"], note=note, flagged_by=flagged_by or "anonymous",
                    flagged_at=_now())
        self._save_flag(flag)
        return flag

    def dismiss(self, trace_id: str, note: str = "") -> Flag:
        flag = self._open_flag(trace_id)
        flag.status, flag.closed_at, flag.closing_note = "dismissed", _now(), note.strip() or None
        self._save_flag(flag)
        return flag

    def confirm(
        self, trace_id: str, root_cause_step: str | None = None,
        failure_type: str | None = None, note: str = "",
    ) -> EvalCase:
        """Accept the stored diagnosis, or override it by giving step and/or type."""
        flag = self._open_flag(trace_id)
        diagnosis = self.traces.load_diagnosis(trace_id)
        said_step = diagnosis.get("root_cause_step") if diagnosis else None
        said_type = diagnosis.get("failure_type") if diagnosis else None

        step = root_cause_step or said_step
        kind = failure_type or said_type
        if step is None or kind is None:
            raise FeedbackError(
                "no diagnosis to confirm: analyze the trace first, or give both the step and the failure type"
            )
        if step not in STEP_ORDER:
            raise FeedbackError(f"unknown step {step!r}; use one of {', '.join(STEP_ORDER)}")
        try:
            kind_value = FailureType(kind)
        except ValueError:
            raise FeedbackError(f"unknown failure type {kind!r}; use one of "
                                f"{', '.join(t.value for t in FailureType)}") from None

        agreed = (step, kind_value.value) == (said_step, said_type)
        if agreed:
            said = None
        elif diagnosis is None:
            said = "not analyzed"
        elif said_step is None:
            said = "no failure found"
        else:
            said = f"{said_step} / {said_type}"
        cases = self.load_cases()
        case = EvalCase(
            case_id=f"case_{len(cases) + 1:03d}",
            doc_id=flag.doc_id,
            source_trace_id=trace_id,
            created_at=_now(),
            note=" / ".join(n for n in (flag.note, note.strip()) if n),
            root_cause_step=step,
            failure_type=kind_value,
            diagnosis_agreed=agreed,
            analyzer_said=said,
        )
        self.save_cases([*cases, case])
        flag.status, flag.closed_at, flag.case_id = "confirmed", _now(), case.case_id
        flag.closing_note = note.strip() or None
        self._save_flag(flag)
        return case

    def _open_flag(self, trace_id: str) -> Flag:
        flag = self.get_flag(trace_id)
        if flag is None:
            raise FeedbackError(f"trace {trace_id[:8]} has not been flagged")
        if flag.status != "open":
            raise FeedbackError(f"the flag on {trace_id[:8]} is already {flag.status}")
        return flag

    # --- eval cases ----------------------------------------------------------------

    def load_cases(self) -> list[EvalCase]:
        if not self.cases_path.is_file():
            return []
        return [EvalCase.model_validate(c) for c in json.loads(self.cases_path.read_text(encoding="utf-8"))]

    def save_cases(self, cases: list[EvalCase]) -> None:
        self.cases_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cases_path.with_suffix(".tmp")
        data: list[dict[str, Any]] = [c.model_dump(mode="json") for c in cases]
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self.cases_path)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
