"""Failure Forensics UI.

    streamlit run src/forensics/ui/app.py

Pages: Traces (explore a run, analyze it, flag it), Compare (two runs side by side,
or a run against the ground truth), Review (open flags and eval cases), Analytics.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from forensics.ui import common

st.set_page_config(page_title="Failure Forensics", page_icon="🔎", layout="wide")

_HERE = Path(__file__).parent / "pages"
common.PAGES.update({
    "traces": st.Page(_HERE / "traces.py", title="Traces", icon="🔎", default=True),
    "compare": st.Page(_HERE / "compare.py", title="Compare", icon="↔️"),
    "review": st.Page(_HERE / "review.py", title="Review", icon="🚩"),
    "analytics": st.Page(_HERE / "analytics.py", title="Analytics", icon="📊"),
})
st.navigation(list(common.PAGES.values())).run()
