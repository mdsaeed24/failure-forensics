"""Phase 3: OpenTelemetry tracing for the pipeline.

Span tree for one pipeline run (one trace):

    pipeline                      doc_id, batch_id, status, failed_step
    ├── step intake               input, output, signals
    ├── step extraction           input, output, signals
    │   └── chat deepseek-flash   tokens, cost, finish_reason, messages, raw output, reasoning
    ├── step classification
    │   └── chat deepseek-flash
    └── step summarization
        └── chat deepseek-flash
"""
