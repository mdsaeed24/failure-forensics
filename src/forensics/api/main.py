"""HTTP API for the feedback loop, so other systems can flag traces too.

    uvicorn forensics.api.main:app --port 8000      # docs at http://localhost:8000/docs

    GET  /traces                       list (filters: doc_id, status, limit)
    GET  /traces/{id}                  trace + diagnosis + flag (a unique id prefix works)
    POST /traces/{id}/analyze          run the root-cause analyzer (calls the judge: costs money)
    POST /traces/{id}/flag             {"note": "...", "flagged_by": "..."}
    GET  /flags                        (filter: status)
    POST /flags/{id}/confirm           {} accepts the diagnosis; or {"root_cause_step", "failure_type"}
    POST /flags/{id}/dismiss           {"note": "..."}
    GET  /eval-cases

A thin layer over forensics.feedback: the Streamlit UI calls the same functions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from forensics.analyzer.root_cause import analyze
from forensics.config import Settings, load_settings
from forensics.feedback.eval_cases import EvalCase, Feedback, FeedbackError, Flag
from forensics.llm.client import LLMClient
from forensics.scoring.final_score import load_ground_truth, score_trace
from forensics.tracing.store import TraceStore


class FlagRequest(BaseModel):
    note: str
    flagged_by: str = "anonymous"


class ConfirmRequest(BaseModel):
    root_cause_step: str | None = None
    failure_type: str | None = None
    note: str = ""


class DismissRequest(BaseModel):
    note: str = ""


def create_app(
    settings: Settings | None = None,
    llm_factory: Callable[[Settings], LLMClient] = LLMClient.from_settings,
) -> FastAPI:
    settings = settings or load_settings()
    traces = TraceStore(settings.resolve(settings.paths.traces_dir))
    feedback = Feedback(traces, settings.resolve(settings.paths.eval_cases))
    app = FastAPI(title="Failure Forensics", version="0.1.0",
                  description="Flag bad pipeline runs, confirm diagnoses, build eval cases.")

    def resolve(prefix: str) -> str:
        try:
            return traces.resolve_id(prefix)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from None

    def bad_request(exc: FeedbackError) -> HTTPException:
        conflict = "already" in str(exc)
        return HTTPException(409 if conflict else 400, str(exc))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/traces")
    def list_traces(doc_id: str | None = None, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return traces.list_traces(doc_id=doc_id, status=status, limit=limit)

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        full = resolve(trace_id)
        flag = feedback.get_flag(full)
        return {"trace": traces.load(full), "diagnosis": traces.load_diagnosis(full),
                "flag": flag.model_dump(mode="json") if flag else None}

    @app.post("/traces/{trace_id}/analyze")
    async def analyze_trace(trace_id: str) -> dict[str, Any]:
        full = resolve(trace_id)
        trace_doc = traces.load(full)
        truth = load_ground_truth(settings.resolve(settings.paths.ground_truth)).get(trace_doc["doc_id"])
        symptom = [d.message for d in score_trace(trace_doc, truth).discrepancies] if truth else []
        flag = feedback.get_flag(full)
        if flag:
            symptom.append(f"flagged: {flag.note}")
        try:
            llm = llm_factory(settings)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from None
        diagnosis = await analyze(trace_doc, llm, settings, symptom)
        data = diagnosis.model_dump(mode="json")
        traces.save_diagnosis(data)
        return data

    @app.post("/traces/{trace_id}/flag", status_code=201)
    def flag_trace(trace_id: str, body: FlagRequest) -> Flag:
        try:
            return feedback.flag_trace(resolve(trace_id), body.note, body.flagged_by)
        except FeedbackError as exc:
            raise bad_request(exc) from None

    @app.get("/flags")
    def list_flags(status: str | None = None) -> list[Flag]:
        return feedback.list_flags(status)

    @app.post("/flags/{trace_id}/confirm", status_code=201)
    def confirm(trace_id: str, body: ConfirmRequest) -> EvalCase:
        try:
            return feedback.confirm(resolve(trace_id), body.root_cause_step, body.failure_type, body.note)
        except FeedbackError as exc:
            raise bad_request(exc) from None

    @app.post("/flags/{trace_id}/dismiss")
    def dismiss(trace_id: str, body: DismissRequest) -> Flag:
        try:
            return feedback.dismiss(resolve(trace_id), body.note)
        except FeedbackError as exc:
            raise bad_request(exc) from None

    @app.get("/eval-cases")
    def eval_cases() -> list[EvalCase]:
        return feedback.load_cases()

    return app


def __getattr__(name: str) -> Any:
    """`uvicorn forensics.api.main:app` builds the app on first access, so importing this
    module (in tests, say) doesn't read config.yaml."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
