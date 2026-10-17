# Shortcuts. Each target is one plain command, so on Windows without `make`
# you can run the command on the right-hand side directly.

PY ?= python

.PHONY: install test run check-models

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

run:
	$(PY) scripts/run_pipeline.py --doc data/samples/sample_invoice.md

check-models:
	$(PY) scripts/check_models.py
