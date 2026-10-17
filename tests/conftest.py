from __future__ import annotations

from typing import Any

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from fakes import FakeLLM
from forensics.config import (
    GenerationSettings,
    LLMSettings,
    ModelCallSettings,
    ModelPrice,
    Settings,
)
from forensics.llm.client import LLMClient
from forensics.tracing.exporter import ForensicsExporter
from forensics.tracing.store import TraceStore


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests NEVER call the real API: building a real client fails loudly."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not call the real API; use the make_llm fixture.")

    monkeypatch.setattr("forensics.llm.client.AsyncOpenAI", refuse)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm=LLMSettings(
            pipeline_model="fake-flash",
            judge_model="fake-pro",
            concurrency=2,
            max_retries=2,
            backoff_base_seconds=0.0,  # no real waiting in tests
        ),
        prices={"fake-flash": ModelPrice(input_per_million=1.0, output_per_million=2.0)},
        generation=GenerationSettings(
            max_attempts=3,
            generator=ModelCallSettings(model="fake-pro", thinking=False, temperature=0.9),
            verifier=ModelCallSettings(model="fake-pro", max_output_tokens=8000),
        ),
    )


@pytest.fixture
def make_llm(settings: Settings):
    """make_llm(reply, reply, ...) -> (LLMClient, FakeLLM) with replies scripted in order."""

    def make(*replies: Any) -> tuple[LLMClient, FakeLLM]:
        fake = FakeLLM(list(replies))
        return LLMClient(fake, settings), fake

    return make


class _SwitchableProcessor(SpanProcessor):
    """OpenTelemetry allows one global TracerProvider per process, so tests share one
    and swap where its spans go. With no target, spans are created and dropped."""

    target: SpanProcessor | None = None

    def on_start(self, span, parent_context=None):
        if self.target:
            self.target.on_start(span, parent_context)

    def on_end(self, span):
        if self.target:
            self.target.on_end(span)


_switch = _SwitchableProcessor()


@pytest.fixture(scope="session", autouse=True)
def _tracer_provider():
    provider = TracerProvider()
    provider.add_span_processor(_switch)
    trace.set_tracer_provider(provider)


@pytest.fixture
def exporter(tmp_path) -> ForensicsExporter:
    """Spans made during the test are exported to tmp_path/traces."""
    exporter = ForensicsExporter(tmp_path / "traces")
    _switch.target = SimpleSpanProcessor(exporter)
    yield exporter
    _switch.target = None


@pytest.fixture
def trace_store(exporter) -> TraceStore:
    return exporter.store
