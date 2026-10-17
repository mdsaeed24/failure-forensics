# Shortcuts. Each target is one plain command, so on Windows without `make`
# you can run the command on the right-hand side directly.

PY ?= python

.PHONY: install test run check-models generate review traces analyze

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

run:
	$(PY) scripts/run_pipeline.py --doc data/samples/sample_invoice.md

check-models:
	$(PY) scripts/check_models.py

generate:
	$(PY) scripts/generate_documents.py --limit 3

review:
	$(PY) scripts/review_documents.py

traces:
	$(PY) scripts/list_traces.py

analyze:
	$(PY) scripts/analyze_trace.py --latest
