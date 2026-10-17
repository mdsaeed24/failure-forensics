from __future__ import annotations

import asyncio

from fakes import pipeline_or_judge_reply
from forensics.evaluation.labels import EvalStore, RunRecord
from forensics.evaluation.metrics import compute
from forensics.evaluation.report import render
from forensics.evaluation.runner import EvalDoc, run_evaluation
from forensics.faults.registry import FAULTS
from test_analyzer import TRUTH
from test_checks import SOURCE


def evaluate(make_llm, settings, trace_store, tmp_path, faults, replies=200):
    llm, fake = make_llm(*[pipeline_or_judge_reply] * replies)
    labels = EvalStore(tmp_path / "evaluation.sqlite")
    labels.start("e1", "2026-10-17T00:00:00", {})
    seen = []
    records = asyncio.run(run_evaluation(
        "e1", [EvalDoc("inv_test", SOURCE, TRUTH)], [FAULTS[f] for f in faults], 1, 7,
        llm, settings, trace_store, labels, seen.append,
    ))
    return records, labels, fake, seen


def test_end_to_end_with_a_judge_that_sees_nothing_wrong(make_llm, settings, trace_store, tmp_path):
    records, labels, _, seen = evaluate(make_llm, settings, trace_store, tmp_path,
                                        ["transpose_amount", "confusable_label"])
    clean, transpose, confusable = sorted(records, key=lambda r: r.fault)[0], *[
        next(r for r in records if r.fault == f) for f in ("transpose_amount", "confusable_label")]

    assert clean.kind == "clean" and clean.final_good and clean.analyzed and clean.diag_step is None
    assert transpose.took_effect and (transpose.diag_step, transpose.diag_type) == ("extraction", "wrong_value")
    assert transpose.decided_by == "deterministic" and transpose.judge_calls == 0
    # The always-approving fake judge can't spot the swapped label: a measured miss.
    assert confusable.took_effect and confusable.diag_step is None
    assert len(seen) == 3 and len(labels.records("e1")) == 3

    m = compute(records)
    assert (m.step_accuracy.hits, m.step_accuracy.n) == (1, 2)
    assert (m.false_alarms.hits, m.false_alarms.n) == (0, 1)
    assert m.type_confusion["misclassification"]["none"] == 1
    report = render(m, {"eval_id": "e1", "started_at": "2026-10-17T00:00", "pipeline_model": "p",
                        "judge_model": "j", "docs": 1, "per_fault": 1, "seed": 7})
    assert "Right step: 50%" in report and "`confusable_label`" in report


def test_resume_skips_finished_runs(make_llm, settings, trace_store, tmp_path):
    _, labels, fake, _ = evaluate(make_llm, settings, trace_store, tmp_path, ["invent_party"])
    calls_first = len(fake.requests)
    llm, fake2 = make_llm()
    records = asyncio.run(run_evaluation(
        "e1", [EvalDoc("inv_test", SOURCE, TRUTH)], [FAULTS["invent_party"]], 1, 7,
        llm, settings, trace_store, labels, lambda r: None,
    ))
    assert len(records) == 2 and fake2.requests == [] and calls_first > 0


def test_fault_that_does_not_take_is_not_analyzed_or_counted(make_llm, settings, trace_store, tmp_path):
    records, _, _, _ = evaluate(make_llm, settings, trace_store, tmp_path, ["prompt_embellish"])
    # The fake summarizer ignores the weaker prompt, so the output stays correct.
    [r] = [r for r in records if r.fault]
    assert r.applied and r.final_good and not r.took_effect and not r.analyzed
    assert compute(records).step_accuracy.n == 0


def test_self_confidence_uses_only_genuine_outputs():
    base = dict(eval_id="e", doc_id="d", final_good=True, extraction_confidence=5,
                extraction_correct=True, classification_confidence=2, classification_correct=False)
    records = [
        RunRecord(kind="clean", **base),
        RunRecord(kind="injected", fault="x", mechanism="hook", applied=True, **base),
        RunRecord(kind="injected", fault="y", mechanism="prompt", applied=True, **base),
    ]
    m = compute(records)
    assert m.self_confidence["classification"][2].n == 2  # clean + prompt, not hook
    assert m.self_confidence["classification"][2].hits == 2
