"""Traces: find a run, open it, analyze it, flag it."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from forensics.ui import common
from forensics.ui.components import trace_view

st.title("🔎 Traces")
store = common.traces()
fb = common.feedback()

with st.sidebar:
    st.header("Filter")
    doc_id = st.text_input("Document id", placeholder="e.g. inv_007").strip() or None
    status = st.selectbox("Status", ["any", "ok", "error"])
    batch = st.text_input("Batch id").strip() or None
    only_flagged = st.toggle("Flagged only")
    only_diagnosed = st.toggle("Diagnosed only")
    limit = st.slider("Show at most", 20, 1000, 200, step=20)

rows = store.list_traces(doc_id=doc_id, status=None if status == "any" else status, batch_id=batch, limit=limit)
flags = {f.trace_id: f for f in fb.list_flags()}
table = []
for r in rows:
    diag = store.load_diagnosis(r["trace_id"])
    if only_flagged and r["trace_id"] not in flags or only_diagnosed and diag is None:
        continue
    table.append({
        "trace": common.short(r["trace_id"]),
        "started (UTC)": r["started_at"][:19].replace("T", " "),
        "document": r["doc_id"],
        "status": common.status_label(r["status"]),
        "failed at": r["failed_step"] or "",
        "seconds": round(r["duration_ms"] / 1000, 1),
        "tokens": r["total_tokens"],
        "cost $": r["cost_usd"],
        "root cause": (f"{diag['root_cause_step']} / {diag['failure_type']}"
                       if diag and diag["root_cause_step"] else "none found" if diag else ""),
        "flag": common.status_label(flags[r["trace_id"]].status) if r["trace_id"] in flags else "",
        "_id": r["trace_id"],
    })

if not table:
    st.info("No traces match. Run some documents first: `python scripts/run_pipeline.py --limit 3`")
    st.stop()

frame = pd.DataFrame(table)
st.caption(f"{len(frame)} run(s). Click a row to open it.")
picked = st.dataframe(
    frame.drop(columns="_id"), hide_index=True, width="stretch", height=280,
    on_select="rerun", selection_mode="single-row",
    column_config={"cost $": st.column_config.NumberColumn(format="$%.5f")},
)
if picked.selection.rows:
    st.session_state["selected_trace"] = frame.iloc[picked.selection.rows[0]]["_id"]

selected = st.session_state.get("selected_trace") or st.query_params.get("trace")
if selected:
    try:
        selected = store.resolve_id(selected)
    except KeyError:
        st.warning(f"Trace {selected} not found.")
        st.stop()
    st.query_params["trace"] = selected
    st.divider()
    trace_view(selected)
