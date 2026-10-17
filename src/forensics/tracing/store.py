"""Trace storage: one JSON file per trace, plus a SQLite index for querying.

    traces/<started>_<doc_id>_<trace8>.json   the full span tree, human-readable
    traces/index.sqlite                        one row per trace and per step

The JSON file is the source of truth; the index only holds what's worth
filtering or sorting on, so it can always be rebuilt from the files.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id         TEXT PRIMARY KEY,
    doc_id           TEXT,
    batch_id         TEXT,
    started_at       TEXT,
    duration_ms      REAL,
    status           TEXT,
    failed_step      TEXT,
    complete         INTEGER,
    llm_calls        INTEGER,
    total_tokens     INTEGER,
    reasoning_tokens INTEGER,
    cost_usd         REAL,
    any_truncated    INTEGER,
    any_repaired     INTEGER,
    path             TEXT
);
CREATE INDEX IF NOT EXISTS traces_by_doc ON traces (doc_id, started_at);
CREATE INDEX IF NOT EXISTS traces_by_batch ON traces (batch_id);

CREATE TABLE IF NOT EXISTS diagnoses (
    trace_id         TEXT PRIMARY KEY,
    doc_id           TEXT,
    analyzed_at      TEXT,
    root_cause_step  TEXT,
    failure_type     TEXT,
    confidence       TEXT,
    judge_calls      INTEGER,
    cost_usd         REAL,
    path             TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    trace_id         TEXT,
    step             TEXT,
    step_index       INTEGER,
    status           TEXT,
    duration_ms      REAL,
    error            TEXT,
    confidence       INTEGER,
    llm_calls        INTEGER,
    attempts         INTEGER,
    tokens           INTEGER,
    reasoning_tokens INTEGER,
    cost_usd         REAL,
    finish_reason    TEXT,
    truncated        INTEGER,
    repaired         INTEGER,
    signals          TEXT,
    PRIMARY KEY (trace_id, step)
);
"""

_TRACE_COLUMNS = (
    "trace_id", "doc_id", "batch_id", "started_at", "duration_ms", "status", "failed_step",
    "complete", "llm_calls", "total_tokens", "reasoning_tokens", "cost_usd", "any_truncated",
    "any_repaired", "path",
)
_STEP_COLUMNS = (
    "trace_id", "step", "step_index", "status", "duration_ms", "error", "confidence",
    "llm_calls", "attempts", "tokens", "reasoning_tokens", "cost_usd", "finish_reason",
    "truncated", "repaired", "signals",
)


class TraceStore:
    def __init__(self, traces_dir: Path) -> None:
        self.dir = traces_dir
        self.index_path = traces_dir / "index.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self.dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.index_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def save(self, trace_doc: dict[str, Any], steps: list[dict[str, Any]]) -> Path:
        """Writes the trace file, then its index rows (replacing any for the same trace)."""
        path = self.dir / trace_doc["file"]
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(trace_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)

        row = {**{k: trace_doc.get(k) for k in _TRACE_COLUMNS}, **trace_doc["summary"],
               "path": trace_doc["file"]}
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM steps WHERE trace_id = ?", (trace_doc["trace_id"],))
            conn.execute(
                f"INSERT OR REPLACE INTO traces ({', '.join(_TRACE_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _TRACE_COLUMNS)})",
                [_sql(row.get(c)) for c in _TRACE_COLUMNS],
            )
            conn.executemany(
                f"INSERT INTO steps ({', '.join(_STEP_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _STEP_COLUMNS)})",
                [[_sql(s.get(c)) for c in _STEP_COLUMNS] for s in steps],
            )
        return path

    def save_diagnosis(self, diagnosis: dict[str, Any]) -> Path:
        """One diagnosis per trace; re-analyzing replaces it."""
        path = self.dir / "diagnoses" / f"{diagnosis['trace_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        row = {**diagnosis, "path": str(path.relative_to(self.dir))}
        columns = ("trace_id", "doc_id", "analyzed_at", "root_cause_step", "failure_type",
                   "confidence", "judge_calls", "cost_usd", "path")
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO diagnoses ({', '.join(columns)}) "
                f"VALUES ({', '.join('?' for _ in columns)})",
                [_sql(row.get(c)) for c in columns],
            )
        return path

    def load_diagnosis(self, trace_id: str) -> dict[str, Any] | None:
        path = self.dir / "diagnoses" / f"{trace_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def list_traces(
        self, doc_id: str | None = None, status: str | None = None,
        batch_id: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses, params = [], []
        for column, value in (("doc_id", doc_id), ("status", status), ("batch_id", batch_id)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT * FROM traces {where} ORDER BY started_at DESC LIMIT ?", [*params, limit]
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, trace_id: str) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            raise KeyError(f"no trace {trace_id!r}")
        return dict(row)

    def steps(self, trace_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM steps WHERE trace_id = ? ORDER BY step_index", (trace_id,)
            ).fetchall()
        return [dict(r) | {"signals": json.loads(r["signals"] or "{}")} for r in rows]

    def resolve_id(self, prefix: str) -> str:
        """Full trace id from a unique prefix (like git's short hashes)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT trace_id FROM traces WHERE trace_id LIKE ? LIMIT 2", (prefix + "%",)
            ).fetchall()
        if len(rows) != 1:
            raise KeyError(f"{'no' if not rows else 'more than one'} trace matches {prefix!r}")
        return rows[0]["trace_id"]

    def path_of(self, trace_id: str) -> Path:
        return self.dir / self.get(trace_id)["path"]

    def load(self, trace_id: str) -> dict[str, Any]:
        return json.loads(self.path_of(trace_id).read_text(encoding="utf-8"))


def _sql(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value
