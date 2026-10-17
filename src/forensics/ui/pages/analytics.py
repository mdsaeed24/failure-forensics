"""Analytics: what's failing, where, how much it costs, and how good the analyzer is.

Every chart here is a single series in one validated color (no legend needed; the
title names it), with a hover tooltip and a table view underneath.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from forensics.evaluation.labels import EvalStore
from forensics.evaluation.metrics import compute
from forensics.pipeline.runner import STEP_ORDER
from forensics.ui import common

st.title("📊 Analytics")
store = common.traces()
rows = store.list_traces(limit=100_000)
if not rows:
    st.info("No traces yet.")
    st.stop()

diagnoses = [d for r in rows if (d := store.load_diagnosis(r["trace_id"]))]
flags = common.feedback().list_flags()
cases = common.feedback().load_cases()
errors = sum(r["status"] == "error" for r in rows)

tiles = st.columns(5)
tiles[0].metric("Pipeline runs", f"{len(rows):,}")
tiles[1].metric("Crashed runs", errors, help="A step raised an error (e.g. invalid output after the retry).")
tiles[2].metric("Analyzed", len(diagnoses))
tiles[3].metric("Open flags", sum(f.status == "open" for f in flags))
tiles[4].metric("Known issues resolved", f"{sum(c.status == 'resolved' for c in cases)}/{len(cases)}")

color = common.series_color()


def bar(data: pd.DataFrame, category: str, value: str, title: str, value_title: str,
        sort: list[str] | str = "-x", fmt: str = ",") -> None:
    """Horizontal bars: thin, rounded data end, recessive axes, tooltip on hover."""
    if data.empty:
        st.caption(f"{title}: no data yet.")
        return
    chart = (
        alt.Chart(data, title=alt.Title(title, anchor="start", fontSize=14, fontWeight=600))
        .mark_bar(color=color, cornerRadiusEnd=4, size=14)
        .encode(
            y=alt.Y(f"{category}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=180, ticks=False, domain=False)),
            x=alt.X(f"{value}:Q", title=value_title,
                    axis=alt.Axis(grid=True, gridOpacity=0.15, domain=False, ticks=False, format=fmt)),
            tooltip=[alt.Tooltip(f"{category}:N", title=category),
                     alt.Tooltip(f"{value}:Q", title=value_title, format=fmt)],
        )
        .properties(height=max(90, 26 * len(data)))
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, width="stretch")
    with st.expander("Table"):
        st.dataframe(data, hide_index=True, width="stretch")


st.markdown("### Where runs fail")
named = [d for d in diagnoses if d["root_cause_step"]]
left, right = st.columns(2, gap="large")
with left:
    by_step = Counter(d["root_cause_step"] for d in named)
    bar(pd.DataFrame({"step": list(by_step), "runs": list(by_step.values())}), "step", "runs",
        "Diagnosed root cause, by step", "runs", sort=[s for s in STEP_ORDER if s in by_step])
with right:
    by_type = Counter(d["failure_type"] or "unclassified" for d in named)
    bar(pd.DataFrame({"failure type": list(by_type), "runs": list(by_type.values())}), "failure type", "runs",
        "Diagnosed root cause, by failure type", "runs")
st.caption(f"{len(named)} of {len(diagnoses)} analyzed runs had a root cause. Analyzed runs include "
           "evaluation runs with planted faults, so this reflects testing, not production traffic.")

st.markdown("### Cost and speed")
left, right = st.columns(2, gap="large")
with left:
    cost: dict[str, float] = Counter()
    for r in rows:
        cost[r["batch_id"] or "no batch"] += r["cost_usd"] or 0
    bar(pd.DataFrame({"batch": list(cost), "cost": list(cost.values())}), "batch", "cost",
        "Pipeline cost, by batch", "USD", fmt="$,.3f")
with right:
    durations: dict[str, list[float]] = {s: [] for s in STEP_ORDER}
    for r in rows[:2000]:  # recent runs are plenty for typical latency
        for s in store.steps(r["trace_id"]):
            durations[s["step"]].append(s["duration_ms"] / 1000)
    median = {s: sorted(v)[len(v) // 2] for s, v in durations.items() if v}
    bar(pd.DataFrame({"step": list(median), "seconds": list(median.values())}), "step", "seconds",
        "Median step duration", "seconds", sort=[s for s in STEP_ORDER if s in median], fmt=",.1f")

st.markdown("### How good is the analyzer?")
s = common.settings()
evals = EvalStore(s.resolve(s.paths.traces_dir) / "evaluation.sqlite")
eval_id = evals.latest() if evals.path.is_file() else None
if eval_id is None:
    st.caption("No evaluation yet: `python scripts/evaluate_diagnosis.py --per-fault 6`")
    st.stop()
m = compute(evals.records(eval_id))


def pct(rate: Any) -> str:
    return "n/a" if rate.value is None else f"{rate.value:.0%}"


tiles = st.columns(4)
tiles[0].metric("Right step", pct(m.step_accuracy), help=f"{m.step_accuracy.hits}/{m.step_accuracy.n} planted faults that took")
tiles[1].metric("Right step + type", pct(m.exact_accuracy), help=f"{m.exact_accuracy.hits}/{m.exact_accuracy.n}")
tiles[2].metric("False alarms", pct(m.false_alarms), help=f"{m.false_alarms.hits}/{m.false_alarms.n} correct clean runs blamed anyway")
tiles[3].metric("Evaluation cost", f"${m.total_cost:.2f}")
st.dataframe(pd.DataFrame([{
    "fault": f.fault, "breaks": f.expected_step, "type": f.expected_type, "how": f.mechanism,
    "took": f"{f.took}/{f.runs}", "right step+type": f"{f.exact}/{f.took}" if f.took else "-",
    "decided by": ", ".join(f"{k} {v}" for k, v in f.decided_by.most_common()) or "-",
} for f in m.by_fault]), hide_index=True, width="stretch")
st.caption(f"Evaluation {eval_id}. Full report with confusion matrices, calibration and limitations: "
           "reports/diagnosis_accuracy.md")
