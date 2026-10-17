"""Reusable pieces: the trace view, the diagnosis panel, the feedback panel."""

from __future__ import annotations

from typing import Any

import streamlit as st

from forensics.analyzer.root_cause import analyze, step_nodes
from forensics.analyzer.taxonomy import FailureType
from forensics.feedback.eval_cases import FeedbackError
from forensics.pipeline.runner import STEP_ORDER
from forensics.scoring.final_score import score_trace
from forensics.tracing import spans as attr
from forensics.ui import common


def trace_view(trace_id: str) -> None:
    store = common.traces()
    trace_doc = store.load(trace_id)
    diagnosis = store.load_diagnosis(trace_id)
    summary = trace_doc["summary"]

    st.subheader(f"{trace_doc['doc_id']} · trace `{common.short(trace_id)}`")
    st.caption(f"{trace_doc['started_at'][:19].replace('T', ' ')} UTC · batch {trace_doc.get('batch_id') or '-'}"
               + ("" if trace_doc["complete"] else " · INCOMPLETE (the run stopped early)"))
    cols = st.columns(6)
    cols[0].metric("Status", common.status_label(trace_doc["status"]))
    cols[1].metric("Failed at", trace_doc.get("failed_step") or "-")
    cols[2].metric("Duration", f"{trace_doc['duration_ms'] / 1000:.1f} s")
    cols[3].metric("Tokens", f"{summary['total_tokens']:,}", help=f"{summary['reasoning_tokens']:,} hidden reasoning")
    cols[4].metric("Cost", common.money(summary["cost_usd"]))
    cols[5].metric("LLM calls", summary["llm_calls"])

    if st.button("↔️ Compare with another run of this document", key=f"compare-{trace_id}"):
        st.session_state["compare_doc"] = trace_doc["doc_id"]
        st.switch_page(common.PAGES["compare"])

    truth = common.ground_truth().get(trace_doc["doc_id"])
    symptom: list[str] = []
    if truth:
        score = score_trace(trace_doc, truth)
        symptom = [d.message for d in score.discrepancies]
        if score.good:
            st.success("✅ Final output matches the ground truth.")
        else:
            st.error("❌ Final output is wrong:\n" + "\n".join(f"- {m}" for m in symptom))
    else:
        st.info("No ground truth for this document: judge the output yourself, and flag it if it's wrong.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("#### Steps")
        _steps(trace_doc, diagnosis)
    with right:
        _diagnosis_panel(trace_id, trace_doc, diagnosis, symptom)
        _feedback_panel(trace_id, diagnosis)


def _steps(trace_doc: dict[str, Any], diagnosis: dict[str, Any] | None) -> None:
    nodes = step_nodes(trace_doc)
    blamed = diagnosis.get("root_cause_step") if diagnosis else None
    for index, name in enumerate(STEP_ORDER, start=1):
        node = nodes.get(name)
        if node is None:
            st.markdown(f"{index}. **{name}** · not run")
            continue
        a = node["attributes"]
        signals = {k[len(attr.SIGNAL_PREFIX):]: v for k, v in a.items() if k.startswith(attr.SIGNAL_PREFIX)}
        icon = "❌" if node["status"] == "error" else "🎯" if name == blamed else "✅"
        title = f"{icon} {index}. {name} · {node['duration_ms']:,.0f} ms"
        if "confidence" in signals:
            title += f" · self-confidence {signals['confidence']}/5"
        if name == blamed:
            title += " · ROOT CAUSE"
        with st.expander(title, expanded=node["status"] == "error" or name == blamed):
            if node["error"]:
                st.error(node["error"])
            if signals:
                st.caption(" · ".join(f"{k} = {v}" for k, v in signals.items() if k != "confidence"))
            tab_output, tab_input = st.tabs(["Output", "Input"])
            with tab_output:
                if attr.STEP_OUTPUT in a:
                    st.json(a[attr.STEP_OUTPUT], expanded=2)
                else:
                    st.caption("No output.")
            with tab_input:
                st.json(a.get(attr.STEP_INPUT, {}), expanded=1)
            for call in node["children"]:
                _llm_call(call)


def _llm_call(node: dict[str, Any]) -> None:
    a = node["attributes"]
    flags = [f for f, on in (("TRUNCATED", a.get(attr.LLM_TRUNCATED)), ("REPAIRED", a.get(attr.LLM_REPAIRED))) if on]
    st.markdown(
        f"**{node['name']}** · {a.get(attr.GENAI_INPUT_TOKENS, 0):,} → {a.get(attr.GENAI_OUTPUT_TOKENS, 0):,} tokens "
        f"({a.get(attr.LLM_REASONING_TOKENS, 0):,} reasoning) · {common.money(a.get(attr.LLM_COST))} · "
        f"{a.get(attr.LLM_ATTEMPTS, 0)} attempt(s) · finish {', '.join(a.get(attr.GENAI_FINISH_REASONS) or ['-'])}"
        + (f" · **{' '.join(flags)}**" if flags else "")
    )
    prompt, reply, reasoning = st.tabs(["Prompt", "Raw reply", "Reasoning"])
    with prompt:
        for message in a.get(attr.LLM_MESSAGES, []):
            st.caption(message["role"])
            st.code(message["content"], language=None, wrap_lines=True)
    with reply:
        st.code(a.get(attr.LLM_RAW_OUTPUT, ""), language="json", wrap_lines=True)
    with reasoning:
        text = a.get(attr.LLM_REASONING)
        if text:
            st.text(text)
        else:
            st.caption("No reasoning returned.")


def _diagnosis_panel(trace_id: str, trace_doc: dict[str, Any], diagnosis: dict[str, Any] | None,
                     symptom: list[str]) -> None:
    st.markdown("#### Diagnosis")
    if diagnosis:
        show_diagnosis(diagnosis)
    else:
        st.caption("Not analyzed yet.")
    label = "Re-analyze" if diagnosis else "Analyze"
    if st.button(label, key=f"analyze-{trace_id}",
                 help="Runs deterministic checks, then the LLM judge step by step (about $0.001-0.003)."):
        flag = common.feedback().get_flag(trace_id)
        context = symptom + ([f"flagged: {flag.note}"] if flag else [])
        with st.spinner("Analyzing..."):
            result = common.run_with_llm(lambda llm: analyze(trace_doc, llm, common.settings(), context))
        if result is not None:
            common.traces().save_diagnosis(result.model_dump(mode="json"))
            st.rerun()


def show_diagnosis(diagnosis: dict[str, Any]) -> None:
    box = {"high": st.error, "medium": st.warning, "low": st.info}.get(diagnosis["confidence"], st.info)
    if diagnosis["root_cause_step"] is None:
        st.success(diagnosis["headline"])
    else:
        box(diagnosis["headline"])
    with st.expander("Evidence chain", expanded=True):
        st.markdown("\n".join(f"- {line.strip()}" if not line.startswith("    ") else f"    - {line.strip()[2:]}"
                              for line in diagnosis["evidence"]))
        st.caption(f"{diagnosis['judge_calls']} judge call(s) with {diagnosis['judge_model']} · "
                   f"{common.money(diagnosis['cost_usd'])} · {diagnosis['analyzed_at'][:16].replace('T', ' ')} UTC")


def _feedback_panel(trace_id: str, diagnosis: dict[str, Any] | None) -> None:
    fb = common.feedback()
    flag = fb.get_flag(trace_id)
    st.markdown("#### Feedback")
    if flag is None or flag.status != "open":
        if flag is not None:
            st.caption(f"{common.status_label(flag.status)} on {flag.closed_at:%Y-%m-%d}"
                       + (f" as {flag.case_id}" if flag.case_id else "")
                       + (f": {flag.closing_note}" if flag.closing_note else ""))
        with st.form(f"flag-{trace_id}", clear_on_submit=True):
            note = st.text_area("What's wrong with this output?",
                                placeholder="e.g. the summary mixes up the donation amount and the date")
            who = st.text_input("Your name (optional)")
            if st.form_submit_button("🚩 Flag as bad"):
                try:
                    fb.flag_trace(trace_id, note, who or "anonymous")
                    st.rerun()
                except FeedbackError as exc:
                    st.error(str(exc))
        return
    confirm_form(trace_id, flag.note, flag.flagged_by, diagnosis)


def confirm_form(trace_id: str, note: str, flagged_by: str, diagnosis: dict[str, Any] | None) -> None:
    fb = common.feedback()
    st.warning(f"🚩 Flagged by {flagged_by}: {note}")
    said_step = diagnosis.get("root_cause_step") if diagnosis else None
    said_type = diagnosis.get("failure_type") if diagnosis else None
    steps, types = list(STEP_ORDER), [t.value for t in FailureType]
    with st.form(f"confirm-{trace_id}"):
        st.caption("Confirm the analyzer's diagnosis, or correct it. Either way it becomes an eval case."
                   if said_step else "No diagnosis yet: analyze first, or set the step and type yourself.")
        step = st.selectbox("Root-cause step", steps, index=steps.index(said_step) if said_step in steps else 0)
        kind = st.selectbox("Failure type", types, index=types.index(said_type) if said_type in types else 0)
        extra = st.text_input("Note (optional)")
        confirm, dismiss = st.columns(2)
        if confirm.form_submit_button("✔️ Confirm → eval case", type="primary"):
            try:
                case = fb.confirm(trace_id, step, kind, extra)
                st.toast(f"Saved {case.case_id}")
                st.rerun()
            except FeedbackError as exc:
                st.error(str(exc))
        if dismiss.form_submit_button("Dismiss (output was fine)"):
            fb.dismiss(trace_id, extra)
            st.rerun()

