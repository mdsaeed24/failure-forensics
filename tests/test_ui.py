"""Headless smoke test of every UI page with Streamlit's AppTest: each must render
without an exception, on empty data and on real traces."""

from __future__ import annotations

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from forensics.config import PROJECT_ROOT
from test_analyzer import OK, diagnose, hook, make_trace

APP = str(PROJECT_ROOT / "src" / "forensics" / "ui" / "app.py")
PAGES = ["traces", "compare", "review", "analytics"]


@pytest.fixture
def ui_config(tmp_path, trace_store, monkeypatch):
    config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    config["paths"] |= {"traces_dir": str(trace_store.dir), "eval_cases": str(tmp_path / "cases.json"),
                        "ground_truth": str(tmp_path / "truth.json")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv("FORENSICS_CONFIG", str(path))
    from forensics.ui import common
    common.settings.clear()  # cached per process


def run_page(page: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    if page != "traces":
        at.switch_page(f"pages/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_no_data(ui_config, page):
    at = run_page(page)
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_traces_a_diagnosis_and_a_flag(ui_config, page, make_llm, settings, trace_store):
    from forensics.feedback.eval_cases import Feedback
    trace_doc = make_trace(make_llm, settings, trace_store, hooks=[hook("classification", doc_type="quotation")])
    make_trace(make_llm, settings, trace_store)
    d, _ = diagnose(make_llm, settings, trace_doc, OK, {"score": 1, "failure_type": "misclassification", "reason": "r"})
    trace_store.save_diagnosis(d.model_dump(mode="json"))
    from forensics.ui import common
    Feedback(trace_store, common.settings().resolve(common.settings().paths.eval_cases)).flag_trace(
        trace_doc["trace_id"], "wrong type")

    at = run_page(page)
    if page == "traces":
        at.session_state["selected_trace"] = trace_doc["trace_id"]
        at.run()
        assert any("ROOT CAUSE" in e.label for e in at.expander)
    assert not at.exception, at.exception
