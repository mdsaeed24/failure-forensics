from __future__ import annotations

import asyncio
import json

from opentelemetry import trace

from fakes import CLASSIFICATION, EXTRACTION, SUMMARY, FakeReply, pipeline_reply
from forensics.config import PROJECT_ROOT
from forensics.pipeline.models import RawDocument
from forensics.pipeline.runner import PipelineOptions, run_pipeline

SAMPLE = RawDocument.from_file(PROJECT_ROOT / "data" / "samples" / "sample_invoice.md")


def run(llm, settings, options=None, doc=SAMPLE):
    return asyncio.run(run_pipeline(doc, llm, settings, options))


def step_nodes(trace_doc):
    return {n["attributes"]["forensics.step.name"]: n for n in trace_doc["root"]["children"]}


def test_one_run_writes_one_trace_file_and_index_rows(make_llm, settings, trace_store):
    llm, _ = make_llm(
        FakeReply(EXTRACTION, prompt_tokens=600, completion_tokens=900, reasoning_tokens=700,
                  reasoning="The total is on the last line."),
        CLASSIFICATION, SUMMARY,
    )
    result = run(llm, settings, PipelineOptions(batch_id="b1"))

    [row] = trace_store.list_traces()
    assert row["trace_id"] == result.trace_id
    assert (row["doc_id"], row["batch_id"], row["status"], row["complete"]) == ("sample_invoice", "b1", "ok", 1)
    assert (row["llm_calls"], row["total_tokens"], row["reasoning_tokens"]) == (3, 1500 + 300, 700)
    assert trace_store.path_of(result.trace_id).name.endswith(f"_sample_invoice_{result.trace_id[:8]}.json")

    steps = trace_store.steps(result.trace_id)
    assert [s["step"] for s in steps] == ["intake", "extraction", "classification", "summarization"]
    assert steps[1]["confidence"] == 5 and steps[1]["signals"]["parties_count"] == 3
    assert steps[2]["signals"]["doc_type"] == "invoice"
    assert steps[0]["llm_calls"] == 0 and steps[0]["signals"]["word_count"] == 140


def test_trace_tree_holds_inputs_outputs_and_llm_evidence(make_llm, settings, trace_store):
    llm, _ = make_llm(FakeReply(EXTRACTION, reasoning="thinking..."), CLASSIFICATION, SUMMARY)
    result = run(llm, settings)
    trace_doc = trace_store.load(result.trace_id)

    assert trace_doc["root"]["name"] == "pipeline"
    nodes = step_nodes(trace_doc)
    extraction = nodes["extraction"]
    assert extraction["attributes"]["forensics.step.input"]["doc_id"] == "sample_invoice"  # decoded JSON
    assert extraction["attributes"]["forensics.step.output"]["parties"][0]["name"] == "Harbor Lane Printing Ltd."

    [chat] = extraction["children"]
    a = chat["attributes"]
    assert chat["name"] == "chat fake-flash"
    assert a["gen_ai.request.model"] == "fake-flash" and a["gen_ai.response.finish_reasons"] == ["stop"]
    assert a["forensics.llm.reasoning"] == "thinking..."
    assert json.loads(a["forensics.llm.raw_output"]) == EXTRACTION
    assert a["forensics.llm.messages"][1]["role"] == "user"
    assert "Harbor Lane Printing Ltd." in a["forensics.llm.messages"][1]["content"]

    summarization_input = nodes["summarization"]["attributes"]["forensics.step.input"]
    assert summarization_input["classification"]["doc_type"] == "invoice"


def test_failed_run_records_the_failing_step_and_the_bad_output(make_llm, settings, trace_store):
    llm, _ = make_llm(EXTRACTION, "not json", "still not json")
    result = run(llm, settings)

    [row] = trace_store.list_traces(status="error")
    assert row["failed_step"] == "classification" and row["any_repaired"] == 1
    trace_doc = trace_store.load(result.trace_id)
    assert trace_doc["root"]["status"] == "error"

    classification = step_nodes(trace_doc)["classification"]
    assert classification["status"] == "error" and "LLMOutputError" in classification["error"]
    assert classification["events"][0]["name"] == "exception"
    assert "forensics.step.output" not in classification["attributes"]
    [chat] = classification["children"]
    assert chat["attributes"]["forensics.llm.raw_output"] == "still not json"
    assert "summarization" not in step_nodes(trace_doc)


def test_hook_changes_are_what_the_trace_shows(make_llm, settings, trace_store):
    def drop_amounts(step, output):
        return output.model_copy(update={"amounts": []}) if step == "extraction" else output

    llm, _ = make_llm(EXTRACTION, CLASSIFICATION, SUMMARY)
    result = run(llm, settings, PipelineOptions(hooks=[drop_amounts]))
    steps = trace_store.steps(result.trace_id)
    assert steps[1]["signals"]["amounts_count"] == 0
    trace_text = trace_store.path_of(result.trace_id).read_text()
    assert "drop_amounts" not in trace_text  # the trace never says a hook ran


def test_concurrent_runs_get_separate_traces(make_llm, settings, trace_store):
    llm, _ = make_llm(*[pipeline_reply] * 9)
    docs = [SAMPLE.model_copy(update={"doc_id": f"doc_{i}"}) for i in range(3)]

    async def all_docs():
        return await asyncio.gather(*(run_pipeline(d, llm, settings) for d in docs))

    results = asyncio.run(all_docs())
    assert len({r.trace_id for r in results}) == 3
    for r in results:
        trace_doc = trace_store.load(r.trace_id)
        assert trace_doc["doc_id"] == r.doc_id
        assert [len(n["children"]) for n in trace_doc["root"]["children"]] == [0, 1, 1, 1]


def test_unfinished_trace_is_written_as_incomplete_on_shutdown(exporter, trace_store):
    tracer = trace.get_tracer("test")
    root = tracer.start_span("pipeline", attributes={"forensics.doc_id": "crashed"})
    with trace.use_span(root):
        with tracer.start_as_current_span("step intake", attributes={"forensics.step.name": "intake"}):
            pass
    exporter.shutdown()  # root never ended
    [row] = trace_store.list_traces()
    assert row["complete"] == 0 and row["doc_id"] == "unknown"  # the root's attributes were never exported


def test_spans_outside_a_pipeline_run_are_not_stored(make_llm, trace_store):
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    llm, _ = make_llm({"value": 1})
    asyncio.run(llm.complete_json(system="json", user="?", schema=Answer))  # e.g. a judge call
    assert trace_store.list_traces() == []


def test_trace_ids_resolve_from_a_prefix(make_llm, settings, trace_store):
    llm, _ = make_llm(EXTRACTION, CLASSIFICATION, SUMMARY)
    result = run(llm, settings)
    assert trace_store.resolve_id(result.trace_id[:6]) == result.trace_id
