"""The answer key for evaluation runs: which fault was planted where.

A separate database file (traces/evaluation.sqlite), not a table in the trace index:
the analyzer reads traces through TraceStore, which has no way to reach this file.
Only evaluation code imports this module. The analyzer can't peek at the answer key.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class RunRecord(BaseModel):
    eval_id: str
    kind: Literal["clean", "injected"]
    doc_id: str
    fault: str = ""  # "" for clean runs
    mechanism: Literal["hook", "prompt", ""] = ""
    expected_step: str | None = None  # None: no fault planted
    expected_type: str | None = None
    applied: bool = False  # the fault actually changed something
    detail: str = ""
    trace_id: str | None = None
    pipeline_status: str | None = None
    final_good: bool | None = None  # False = the fault "took" (or a natural failure, if clean)
    discrepancies: list[str] = []
    analyzed: bool = False
    diag_step: str | None = None
    diag_type: str | None = None
    diag_confidence: str | None = None
    decided_by: Literal["deterministic", "judge", "crash", ""] = ""
    judge_calls: int = 0
    # The pipeline model's own 1-5 confidence per step, for calibration.
    extraction_confidence: int | None = None
    classification_confidence: int | None = None
    extraction_correct: bool | None = None  # vs ground truth: no party/date/amount discrepancy
    classification_correct: bool | None = None
    pipeline_cost: float | None = None
    analysis_cost: float | None = None
    error: str | None = None  # the evaluation itself failed for this run (e.g. API outage)

    @property
    def took_effect(self) -> bool:
        return self.kind == "injected" and self.applied and self.final_good is False


class EvalStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.execute("""CREATE TABLE IF NOT EXISTS evals (
            eval_id TEXT PRIMARY KEY, started_at TEXT, config TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            eval_id TEXT, doc_id TEXT, fault TEXT, record TEXT,
            PRIMARY KEY (eval_id, doc_id, fault))""")
        return conn

    def start(self, eval_id: str, started_at: str, config: dict[str, Any]) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR IGNORE INTO evals VALUES (?, ?, ?)",
                         (eval_id, started_at, json.dumps(config)))

    def config(self, eval_id: str) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT started_at, config FROM evals WHERE eval_id = ?",
                               (eval_id,)).fetchone()
        if row is None:
            raise KeyError(f"no evaluation {eval_id!r}")
        return {"started_at": row[0], **json.loads(row[1])}

    def latest(self) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT eval_id FROM evals ORDER BY started_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def save(self, record: RunRecord) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)",
                         (record.eval_id, record.doc_id, record.fault, record.model_dump_json()))

    def forget_fault(self, eval_id: str, fault: str) -> int:
        """Drop a fault's runs, e.g. after fixing it, so --resume runs it again."""
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM runs WHERE eval_id = ? AND fault = ?",
                                (eval_id, fault)).rowcount

    def records(self, eval_id: str) -> list[RunRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT record FROM runs WHERE eval_id = ?", (eval_id,)).fetchall()
        return [RunRecord.model_validate_json(r[0]) for r in rows]
