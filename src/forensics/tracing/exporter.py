"""Custom SpanExporter: finished spans → one JSON file per trace + SQLite index rows.

OpenTelemetry hands over spans as each one ends, children before parents. Spans
are held per trace until the root span ("pipeline") ends, then the whole tree is
written at once. If the process stops before a root ends, shutdown() writes what
it has, marked complete=false.

An exporter must never break the program it observes: errors are logged, not raised.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode, format_span_id, format_trace_id

from forensics.tracing import spans as attr
from forensics.tracing.store import TraceStore

log = logging.getLogger(__name__)


class ForensicsExporter(SpanExporter):
    def __init__(self, traces_dir: Path) -> None:
        self.store = TraceStore(traces_dir)
        self._pending: dict[int, list[ReadableSpan]] = defaultdict(list)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        finished: list[list[ReadableSpan]] = []
        with self._lock:
            for span in spans:
                self._pending[span.context.trace_id].append(span)
                if span.parent is None:  # the root ends last, so the trace is complete
                    finished.append(self._pending.pop(span.context.trace_id))
        ok = all(self._write(trace_spans, complete=True) for trace_spans in finished)
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        with self._lock:
            leftovers, self._pending = list(self._pending.values()), defaultdict(list)
        for trace_spans in leftovers:
            self._write(trace_spans, complete=False)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True

    def _write(self, trace_spans: list[ReadableSpan], complete: bool) -> bool:
        try:
            trace_doc, steps = build_trace(trace_spans, complete)
            self.store.save(trace_doc, steps)
            return True
        except Exception:
            log.exception("could not write trace")
            return False


def build_trace(trace_spans: list[ReadableSpan], complete: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nodes = {s.context.span_id: _node(s) for s in trace_spans}
    roots = []
    for span in sorted(trace_spans, key=lambda s: s.start_time or 0):
        node = nodes[span.context.span_id]
        parent = nodes.get(span.parent.span_id) if span.parent else None
        (parent["children"] if parent else roots).append(node)
    root = roots[0]  # a complete trace has exactly one; a partial one takes the earliest

    steps = [_step_row(n) for n in root["children"] if attr.STEP_NAME in n["attributes"]]
    llm = [n for n in _walk(root) if attr.GENAI_REQUEST_MODEL in n["attributes"]]
    costs = [n["attributes"].get(attr.LLM_COST) for n in llm]
    a = root["attributes"]
    trace_id = format_trace_id(trace_spans[0].context.trace_id)
    started = root["start"]
    doc_id = a.get(attr.DOC_ID, "unknown")

    trace_doc = {
        "trace_id": trace_id,
        "doc_id": doc_id,
        "batch_id": a.get(attr.BATCH_ID),
        "started_at": started,
        "duration_ms": root["duration_ms"],
        "status": a.get(attr.STATUS, root["status"]),
        "failed_step": a.get(attr.FAILED_STEP),
        "complete": complete,
        "summary": {
            "llm_calls": len(llm),
            "total_tokens": sum(_tokens(n) for n in llm),
            "reasoning_tokens": sum(n["attributes"].get(attr.LLM_REASONING_TOKENS, 0) for n in llm),
            "cost_usd": None if None in costs else round(sum(costs), 8),
            "any_truncated": any(n["attributes"].get(attr.LLM_TRUNCATED) for n in llm),
            "any_repaired": any(n["attributes"].get(attr.LLM_REPAIRED) for n in llm),
        },
        "file": _file_name(started, doc_id, trace_id),
        "root": root,
    }
    for step in steps:
        step["trace_id"] = trace_id
    return trace_doc, steps


def _node(span: ReadableSpan) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for key, value in (span.attributes or {}).items():
        if key in attr.JSON_ATTRIBUTES and isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        attributes[key] = list(value) if isinstance(value, tuple) else value
    error = span.status.status_code == StatusCode.ERROR
    return {
        "name": span.name,
        "span_id": format_span_id(span.context.span_id),
        "start": _iso(span.start_time),
        "duration_ms": _ms(span.start_time, span.end_time),
        "status": "error" if error else "ok",
        "error": span.status.description if error else None,
        "attributes": attributes,
        "events": [
            {"name": e.name, "time": _iso(e.timestamp), "attributes": dict(e.attributes or {})}
            for e in span.events
        ],
        "children": [],
    }


def _step_row(node: dict[str, Any]) -> dict[str, Any]:
    a = node["attributes"]
    calls = [c["attributes"] for c in node["children"] if attr.GENAI_REQUEST_MODEL in c["attributes"]]
    costs = [c.get(attr.LLM_COST) for c in calls]
    signals = {k[len(attr.SIGNAL_PREFIX):]: v for k, v in a.items() if k.startswith(attr.SIGNAL_PREFIX)}
    finish = [(c.get(attr.GENAI_FINISH_REASONS) or [None])[-1] for c in calls]
    return {
        "step": a[attr.STEP_NAME],
        "step_index": a.get(attr.STEP_INDEX),
        "status": node["status"],
        "duration_ms": node["duration_ms"],
        "error": node["error"],
        "confidence": signals.get("confidence"),
        "llm_calls": len(calls),
        "attempts": sum(c.get(attr.LLM_ATTEMPTS, 0) for c in calls),
        "tokens": sum(_tokens({"attributes": c}) for c in calls),
        "reasoning_tokens": sum(c.get(attr.LLM_REASONING_TOKENS, 0) for c in calls),
        "cost_usd": None if None in costs else sum(costs),
        "finish_reason": finish[-1] if finish else None,
        "truncated": any(c.get(attr.LLM_TRUNCATED) for c in calls),
        "repaired": any(c.get(attr.LLM_REPAIRED) for c in calls),
        "signals": signals,
    }


def _walk(node: dict[str, Any]):
    yield node
    for child in node["children"]:
        yield from _walk(child)


def _tokens(node: dict[str, Any]) -> int:
    a = node["attributes"]
    return a.get(attr.GENAI_INPUT_TOKENS, 0) + a.get(attr.GENAI_OUTPUT_TOKENS, 0)


def _iso(ns: int | None) -> str | None:
    if ns is None:
        return None
    return dt.datetime.fromtimestamp(ns / 1e9, dt.UTC).isoformat(timespec="milliseconds")


def _ms(start: int | None, end: int | None) -> float | None:
    return None if start is None or end is None else round((end - start) / 1e6, 3)


def _file_name(started: str | None, doc_id: str, trace_id: str) -> str:
    stamp = (started or "unknown")[:19].replace("-", "").replace(":", "").replace("T", "-")
    safe_doc = re.sub(r"[^A-Za-z0-9_-]", "_", doc_id)
    return f"{stamp}_{safe_doc}_{trace_id[:8]}.json"
