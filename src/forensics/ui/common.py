"""Shared state and helpers for the Streamlit pages."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import streamlit as st

from forensics.config import Settings, load_settings
from forensics.feedback.eval_cases import Feedback
from forensics.llm.client import LLMClient
from forensics.scoring.final_score import load_ground_truth
from forensics.tracing.setup import configure_tracing
from forensics.tracing.store import TraceStore

# Single-series charts use categorical slot 1 (validated reference palette), stepped per theme.
SERIES = {"light": "#2a78d6", "dark": "#3987e5"}
STATUS_ICON = {"ok": "✅", "error": "❌", "open": "🚩", "confirmed": "✔️", "dismissed": "➖",
               "resolved": "✅", "regressed": "⚠️"}


@st.cache_resource
def settings() -> Settings:
    return load_settings()


@st.cache_resource
def _tracing() -> None:
    """Regression runs from the UI must be traced. One provider per process, never shut
    down: our exporter writes synchronously, so there is nothing to flush."""
    configure_tracing(settings())


def traces() -> TraceStore:
    s = settings()
    return TraceStore(s.resolve(s.paths.traces_dir))


def feedback() -> Feedback:
    s = settings()
    return Feedback(traces(), s.resolve(s.paths.eval_cases))


def ground_truth() -> dict[str, dict[str, Any]]:
    s = settings()
    return load_ground_truth(s.resolve(s.paths.ground_truth))


def run_with_llm(work: Callable[[LLMClient], Any], traced: bool = False) -> Any:
    """Runs an async job that needs the API; shows a clear error instead of a traceback."""
    try:
        llm = LLMClient.from_settings(settings())
    except RuntimeError as exc:
        st.error(str(exc))
        return None
    if traced:
        _tracing()
    return asyncio.run(work(llm))


def series_color() -> str:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return SERIES["dark" if theme == "dark" else "light"]


def money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.4f}"


def short(trace_id: str | None) -> str:
    return (trace_id or "")[:8]


def status_label(status: str) -> str:
    return f"{STATUS_ICON.get(status, '•')} {status}"


def open_trace(trace_id: str) -> None:
    """Jump to the Traces page showing this trace."""
    st.session_state["selected_trace"] = trace_id
    st.switch_page(PAGES["traces"], query_params={"trace": trace_id})


PAGES: dict[str, Any] = {}  # filled in by app.py, so pages can link to each other
