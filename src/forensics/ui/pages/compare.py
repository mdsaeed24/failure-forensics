"""Compare: two runs of the same document side by side, or a run against the ground truth.

Useful for: a planted fault vs the clean run, before vs after a prompt change, and a
regression run vs the run that was flagged.
"""

from __future__ import annotations

import difflib
import html
from decimal import Decimal
from typing import Any

import pandas as pd
import streamlit as st

from forensics.analyzer.root_cause import step_nodes
from forensics.scoring.final_score import step_outputs
from forensics.textscan import format_amount, normalize_name
from forensics.tracing import spans as attr
from forensics.ui import common

TRUTH = "Ground truth"

st.title("↔️ Compare")
store = common.traces()
truth = common.ground_truth()

docs = sorted({r["doc_id"] for r in store.list_traces(limit=100_000)})
if not docs:
    st.info("No traces yet.")
    st.stop()
default_doc = st.session_state.get("compare_doc")
doc_id = st.selectbox("Document", docs, index=docs.index(default_doc) if default_doc in docs else 0)
runs = store.list_traces(doc_id=doc_id, limit=200)


def label(r: dict[str, Any]) -> str:
    return (f"{r['started_at'][:19].replace('T', ' ')} · {common.short(r['trace_id'])} · {r['status']}"
            f" · {r['batch_id'] or 'no batch'}")


labels = {label(r): r["trace_id"] for r in runs}
choices_b = list(labels) + ([TRUTH] if doc_id in truth else [])
col_a, col_b = st.columns(2)
a_label = col_a.selectbox("Run A", list(labels), index=0)
b_label = col_b.selectbox("B", choices_b, index=min(1, len(choices_b) - 1) if len(choices_b) > 1 else 0)
if a_label == b_label:
    st.info("Pick two different runs, or the ground truth for B.")
    st.stop()

trace_a = store.load(labels[a_label])
out_a = step_outputs(trace_a)
if b_label == TRUTH:
    t = truth[doc_id]
    out_b = {"classification": {"doc_type": t["doc_type"]},
             "extraction": {"parties": t["parties"], "dates": t["dates"], "amounts": t["amounts"]}}
    trace_b = None
else:
    trace_b = store.load(labels[b_label])
    out_b = step_outputs(trace_b)


# --- classification ---------------------------------------------------------------
st.markdown("#### Classification")
type_a = out_a.get("classification", {}).get("doc_type", "-")
type_b = out_b.get("classification", {}).get("doc_type", "-")
c1, c2 = st.columns(2)
c1.metric("A", type_a)
c2.metric("B", type_b, delta=None if type_a == type_b else "differs", delta_color="off")


# --- extracted facts --------------------------------------------------------------
def facts(extraction: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """(kind, comparable key) -> how it reads. Repeated facts collapse, as in scoring."""
    if not extraction:
        return {}
    found: dict[tuple[str, str], str] = {}
    for p in extraction.get("parties", []):
        found[("party", normalize_name(p["name"]))] = p["name"]
    for d in extraction.get("dates", []):
        found[("date", d["date"])] = d["date"]
    for a in extraction.get("amounts", []):
        amount = Decimal(str(a["amount"]))
        shown = format_amount(amount, a["currency"]) if len(a["currency"]) == 3 and a["currency"].isupper() \
            else f"{amount} {a['currency']}"
        found[("amount", f"{amount.normalize()} {a['currency']}")] = shown
    return found


st.markdown("#### Extracted facts")
fa, fb_ = facts(out_a.get("extraction")), facts(out_b.get("extraction"))
rows = []
for key in sorted(set(fa) | set(fb_), key=lambda k: ("party", "date", "amount").index(k[0])):
    rows.append({"kind": key[0], "A": fa.get(key, "—"), "B": fb_.get(key, "—"),
                 "": "" if key in fa and key in fb_ else "only A" if key in fa else "only B"})
only_diff = st.toggle("Differences only", value=False)
frame = pd.DataFrame(rows)
if only_diff and not frame.empty:
    frame = frame[frame[""] != ""]
if frame.empty:
    st.success("Same facts." if rows else "No facts extracted.")
else:
    st.dataframe(frame, hide_index=True, width="stretch")


# --- summary ----------------------------------------------------------------------
st.markdown("#### Summary")
sum_a = out_a.get("summarization", {}).get("summary")
sum_b = out_b.get("summarization", {}).get("summary") if trace_b else None
if sum_a and sum_b:
    words_a, words_b = sum_a.split(), sum_b.split()
    parts = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=words_a, b=words_b).get_opcodes():
        if op == "equal":
            parts.append(html.escape(" ".join(words_a[i1:i2])))
        if op in ("delete", "replace"):
            parts.append(f'<del title="only in A">{html.escape(" ".join(words_a[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            parts.append(f'<ins title="only in B">{html.escape(" ".join(words_b[j1:j2]))}</ins>')
    st.html(
        "<style>.fx-diff{line-height:1.7}"
        ".fx-diff del{background:rgba(227,73,72,.22);text-decoration:line-through;padding:0 2px;border-radius:3px}"
        ".fx-diff ins{background:rgba(27,175,122,.25);text-decoration:none;padding:0 2px;border-radius:3px}</style>"
        f'<div class="fx-diff">{" ".join(parts)}</div>'
    )
    st.caption("Struck through: only in A. Highlighted: only in B.")
elif sum_a:
    st.write(sum_a)
    if trace_b is None:
        st.caption("The ground truth has no summary to compare with.")


# --- per-step signals -------------------------------------------------------------
if trace_b is not None:
    st.markdown("#### Per-step signals")
    na, nb = step_nodes(trace_a), step_nodes(trace_b)
    signal_rows = []
    for step in ("intake", "extraction", "classification", "summarization"):
        for side, nodes in (("A", na), ("B", nb)):
            node = nodes.get(step)
            if node is None:
                continue
            calls = [c["attributes"] for c in node["children"]]
            signal_rows.append({
                "step": step, "run": side, "status": common.status_label(node["status"]),
                "ms": round(node["duration_ms"]),
                "tokens": sum(c.get(attr.GENAI_INPUT_TOKENS, 0) + c.get(attr.GENAI_OUTPUT_TOKENS, 0) for c in calls),
                "reasoning": sum(c.get(attr.LLM_REASONING_TOKENS, 0) for c in calls),
                "self-confidence": node["attributes"].get(attr.SIGNAL_PREFIX + "confidence"),
            })
    st.dataframe(pd.DataFrame(signal_rows), hide_index=True, width="stretch")

c1, c2 = st.columns(2)
if c1.button("Open run A"):
    common.open_trace(trace_a["trace_id"])
if trace_b is not None and c2.button("Open run B"):
    common.open_trace(trace_b["trace_id"])
