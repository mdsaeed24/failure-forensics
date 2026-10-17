"""OpenTelemetry wiring: one TracerProvider per process, exporting to traces/."""

from __future__ import annotations

from collections.abc import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter

from forensics.config import Settings
from forensics.tracing.exporter import ForensicsExporter


def configure_tracing(
    settings: Settings, extra_exporters: Sequence[SpanExporter] = ()
) -> TracerProvider:
    """Call once at the start of a script, and `provider.shutdown()` at the end.

    Our exporter runs synchronously (SimpleSpanProcessor), so a trace's file exists the
    moment its pipeline run returns. Any standard exporter (OTLP to Jaeger, Langfuse,
    ...) can be added through `extra_exporters`; those are batched in the background.
    """
    provider = TracerProvider(resource=Resource.create({"service.name": "failure-forensics"}))
    provider.add_span_processor(
        SimpleSpanProcessor(ForensicsExporter(settings.resolve(settings.paths.traces_dir)))
    )
    for exporter in extra_exporters:
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider
