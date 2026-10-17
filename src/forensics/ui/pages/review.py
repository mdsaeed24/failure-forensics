"""Review: open flags awaiting a decision, and the eval cases they became."""

from __future__ import annotations

import asyncio
import datetime as dt

import pandas as pd
import streamlit as st

from forensics.feedback.regression import apply_result, check_case
from forensics.ui import common
from forensics.ui.components import confirm_form, show_diagnosis

st.title("🚩 Review")
fb = common.feedback()
store = common.traces()

st.markdown("### Open flags")
open_flags = fb.list_flags("open")
if not open_flags:
    st.caption("Nothing waiting. Flag a run from the Traces page, or via the API.")
for flag in open_flags:
    with st.expander(f"🚩 {flag.doc_id} · trace {common.short(flag.trace_id)} · {flag.note}", expanded=True):
        diagnosis = store.load_diagnosis(flag.trace_id)
        if diagnosis:
            show_diagnosis(diagnosis)
        else:
            st.caption("Not analyzed yet: open the trace to analyze it.")
        if st.button("Open trace", key=f"open-{flag.trace_id}"):
            common.open_trace(flag.trace_id)
        confirm_form(flag.trace_id, flag.note, flag.flagged_by, diagnosis)

st.markdown("### Eval cases")
cases = fb.load_cases()
if not cases:
    st.caption("No eval cases yet. Confirming a flag creates one.")
    st.stop()

resolved = sum(c.status == "resolved" for c in cases)
c1, c2, c3 = st.columns(3)
c1.metric("Known issues", len(cases))
c2.metric("Resolved", f"{resolved}/{len(cases)}")
c3.metric("Regressed", sum(c.status == "regressed" for c in cases))

st.dataframe(pd.DataFrame([{
    "case": c.case_id,
    "document": c.doc_id,
    "root cause": f"{c.root_cause_step} / {c.failure_type.value}",
    "analyzer agreed": "yes" if c.diagnosis_agreed else f"no (said {c.analyzer_said or 'nothing'})",
    "status": common.status_label(c.status),
    "last check": (f"{'pass' if c.history[-1].passed else 'fail'}: {c.history[-1].reason}"
                   if c.history else "never run"),
    "note": c.note,
} for c in cases]), hide_index=True, width="stretch")

pending = [c for c in cases if c.status != "resolved"]
include_resolved = st.toggle("Also re-check resolved cases", help="Catches a fixed issue coming back.")
targets = cases if include_resolved else pending
if st.button(f"Run regression on {len(targets)} case(s)", type="primary", disabled=not targets,
             help="Re-runs each document through the current pipeline, then checks the failing step "
                  "(about $0.002 per case)."):
    truth = common.ground_truth()
    batch_id = f"regression-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M}"

    async def run_all(llm):
        return await asyncio.gather(*(check_case(c, llm, common.settings(), store, truth, batch_id)
                                      for c in targets))

    with st.spinner("Re-running..."):
        results = common.run_with_llm(run_all, traced=True)
    if results is not None:
        updated = {c.case_id: apply_result(c, r) for c, r in zip(targets, results)}
        fb.save_cases([updated.get(c.case_id, c) for c in cases])
        st.rerun()
