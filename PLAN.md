# Failure Forensics — Build Plan

An observability and root-cause tool for a 4-step LLM document pipeline.
It traces every step. When the final output is bad, it walks back through the
steps, finds the **first** step that failed, classifies the failure, shows the
evidence, and turns confirmed failures into eval cases. Its diagnosis accuracy
is **measured** on planted failures whose answers are known.

---

## 1. The big picture

```
            ┌──────────── pipeline (Phase 1) ────────────┐
document →  │ 1 Intake → 2 Extraction → 3 Classify → 4 Summarize │ → result
            └───────────────┬────────────────────────────┘
                            │ every step emits a span (Phase 3)
                            ▼
              OpenTelemetry → custom exporter → traces/*.json + SQLite index
                            │
         final output bad?  ▼   (vs ground truth, or a human flag)
              Root-cause analyzer (Phase 4)
              deterministic checks first → LLM judge per step → first bad step
                            │
                            ▼
              Diagnosis + evidence chain → human confirms → eval_cases.json
                                                              │
                                     scripts/regression.py ◄──┘ (Phase 6)

  Fault injection (Phase 5) plants known failures → measures how often the
  analyzer names the right step and failure type.
```

## 2. Folder structure

```
Failure Forensics/
├── PLAN.md                    this file
├── README.md                  case study (Phase 7)
├── pyproject.toml             dependencies + pytest settings
├── Makefile                   shortcuts; every target also has a plain `python ...` command
├── config.yaml                models, token caps, concurrency, prices, paths
├── .env.example               DEEPSEEK_API_KEY=...
├── Dockerfile, docker-compose.yml           (Phase 7)
│
├── data/
│   ├── samples/               hand-written sample docs for smoke runs (no fact sheet)
│   ├── fact_sheets.yaml       YOU write the facts per test document (Phase 2)
│   ├── documents/             generated *.md documents + manifest.json (verification, review state)
│   ├── ground_truth.json      facts per doc_id, built from fact sheets
│   └── eval_cases.json        grows from confirmed human flags (Phase 6)
├── traces/                    one JSON file per trace (gitignored)
├── reports/
│   ├── document_verification.md  output of generate_documents.py (Phase 2)
│   └── diagnosis_accuracy.md  output of the Phase 5 evaluation
│
├── scripts/                   thin command-line entry points; logic lives in src/
│   ├── run_pipeline.py        run docs through the pipeline (--limit)
│   ├── check_models.py        verify config.yaml model IDs against DeepSeek /models
│   ├── list_traces.py         browse traces: list, filter, per-step signals (Phase 3)
│   ├── analyze_trace.py       score a trace vs ground truth; if bad, find the root cause (Phase 4)
│   ├── generate_documents.py  fact sheet → document → automatic fact verification
│   ├── review_documents.py    side-by-side doc vs fact sheet, approve/reject
│   ├── evaluate_diagnosis.py  clean + injected runs → accuracy report
│   ├── regression.py          re-run eval cases, track "known issues resolved"
│   └── seed_demo.py           fill the UI with data (Docker)
│
├── src/forensics/
│   ├── config.py              typed Settings loaded from config.yaml + .env
│   ├── textscan.py            find money amounts and dates in text (no LLM); shared by
│   │                          the Phase 2 verifier and the Phase 4 checks
│   ├── documents/             Phase 2: ground truth by construction
│   │   ├── fact_sheets.py     strict fact sheet model + loader
│   │   ├── generator.py       fact sheet → document, rewritten with feedback until verified
│   │   ├── verifier.py        deterministic checks, then an independent LLM audit
│   │   ├── store.py           documents, manifest, ground_truth.json (approved only)
│   │   └── report.py          verification report + side-by-side review screen
│   ├── llm/
│   │   ├── client.py          DeepSeek via openai SDK: JSON mode, Pydantic validation,
│   │   │                      retry-once on bad JSON, backoff, concurrency limit,
│   │   │                      max tokens, finish_reason logging, truncation flag
│   │   └── cost.py            tokens → dollars
│   ├── pipeline/
│   │   ├── models.py          Pydantic input/output model for each step
│   │   ├── prompts.py         prompt templates (with named variants, for fault injection)
│   │   ├── intake.py          step 1 (no LLM)
│   │   ├── extraction.py      step 2
│   │   ├── classification.py  step 3
│   │   ├── summarization.py   step 4
│   │   └── runner.py          the ONLY place that knows step order; has hook points
│   │                          between steps where faults can be injected
│   ├── tracing/
│   │   ├── setup.py           OpenTelemetry TracerProvider wiring
│   │   ├── spans.py           traced_step(...) context manager + attribute names (gen_ai.* /
│   │   │                      forensics.*); a context manager, not a decorator, so the
│   │   │                      runner's hooks run inside the step's span
│   │   ├── exporter.py        custom SpanExporter → JSON file + SQLite index
│   │   └── store.py           read/query traces from SQLite + JSON
│   ├── checks/                deterministic checks, no LLM; each step vs its OWN input
│   │   ├── intake.py          was the text handed on intact?
│   │   ├── grounding.py       does every extracted name/date/amount appear in the source?
│   │   ├── dates.py           do extracted dates parse?
│   │   ├── currency.py        are amounts + currency codes consistent?
│   │   └── summary_entities.py does the summary mention only extracted entities?
│   ├── scoring/
│   │   └── final_score.py     compare final result to ground truth → good / bad
│   ├── analyzer/
│   │   ├── taxonomy.py        the 5 failure types: hallucination, omission, wrong_value,
│   │   │                      misclassification, format_error
│   │   ├── judge.py           LLM judge: "is this step's output a reasonable
│   │   │                      transformation of ITS OWN input?" (1–5 + reason)
│   │   ├── root_cause.py      walks steps in order, picks first significant drop
│   │   └── evidence.py        builds the plain-English evidence chain
│   ├── faults/
│   │   ├── injectors.py       one class per fault, each with its expected label
│   │   └── registry.py
│   ├── evaluation/
│   │   ├── metrics.py         step/type accuracy, precision/recall, confusion matrix,
│   │   │                      false alarms, confidence calibration
│   │   └── report.py          writes reports/diagnosis_accuracy.md
│   ├── feedback/
│   │   └── eval_cases.py      confirmed flag → eval case
│   ├── api/
│   │   └── main.py            FastAPI: flag a trace, confirm/override a diagnosis
│   └── ui/
│       ├── app.py             Streamlit: trace list + trace view
│       └── pages/             diff view, analytics
│
└── tests/
    ├── conftest.py            FakeLLM fixture: tests NEVER call the real API
    └── test_*.py              tests added in every phase
```

## 3. Phases

Each phase ends with: run it → show it working → 3–5 design-decision bullets → **stop for your OK**.

| Phase | Builds | You'll see it work by |
|---|---|---|
| 1 | LLM client + 4-step pipeline | running 1 sample doc, printing each step's typed output |
| 2 | Fact sheets, document generator + auto-verifier, review tool | generating 3 docs from 3 example fact sheets; verification report. **Then you write ~40–50 fact sheets.** |
| 3 | OpenTelemetry tracing, custom exporter, per-step signals | running a doc; opening its JSON trace and SQLite row |
| 4 | Root-cause analyzer + evidence chain | analyzing a deliberately broken trace |
| 5 | Fault injectors + diagnosis accuracy eval | `reports/diagnosis_accuracy.md` with real numbers |
| 6 | Streamlit explorer, diff view, FastAPI flagging, eval loop, regression | clicking through the UI; flagging a trace end-to-end |
| 7 | Docker + README case study | `docker compose up` and the UI already has data |

## 4. Key design choices (decided up front)

1. **Steps are pure, the runner is the glue.** Each step takes a Pydantic input
   and returns a Pydantic output. Only `runner.py` knows the order. That gives
   fault injection a clean place to plug in (between steps) without touching
   step code.

2. **The analyzer is blind to the injected answer.** The fault label is stored
   in a separate SQLite table that only the evaluation code reads. If the
   analyzer could see it, the accuracy number would be meaningless.
   *(Simple version: a student can't peek at the answer key.)*

3. **"Did the planted fault actually take?"** Some faults (e.g. an odd
   formatting the summarizer *might* still read correctly) don't always cause a
   bad output. The eval reports how often each fault took effect, and measures
   diagnosis accuracy only on runs where the final output really was bad —
   counted honestly, not assumed.

4. **Cheap and reliable before expensive and fuzzy.** Deterministic checks
   (string grounding, date parsing, currency rules) run before any LLM judge.
   They're free, repeatable, and catch hallucinated entities with certainty.

5. **Judge ≠ model under test.** Pipeline uses the fast model; the judge uses a
   stronger, different model, so it doesn't share the same blind spots.

6. **Self-confidence is a weak signal.** We record the model's 1–5 confidence,
   but then *measure* whether it actually predicts failures (calibration) rather
   than trusting it.

## 5. Concepts you'll meet (one-line versions)

- **Trace / span** — a trace is one full pipeline run; a span is one step inside
  it, with a start time, end time, and attached data. Like a receipt with one
  line item per step.
- **SpanExporter** — the plug-in that decides where finished spans go (a file,
  a database, Jaeger…). We write our own; because it's standard OpenTelemetry,
  other exporters can be added alongside it later.
- **Ground truth by construction** — you write the facts first, then generate
  the document from them, so the correct answer is known exactly.
- **Precision / recall** — of the times the analyzer said "Misclassification",
  how often was it right (precision)? Of all real misclassifications, how many
  did it catch (recall)?
- **Confusion matrix** — a table of "true label vs predicted label"; mistakes
  show up off the diagonal.
- **Calibration** — when the model says "confidence 5", is it actually right
  more often than when it says "2"?

## 6. Commands (work on Windows and macOS)

Set up once:

```
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS:     source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env      (Windows)   |   cp .env.example .env   (macOS)
```

Each phase adds its own commands, always as `python scripts/xxx.py ...`,
with a matching `make` target for convenience.

## 7. Open items to confirm in Phase 1

- **Model IDs.** `deepseek-flash` and `deepseek-v4-pro` are set in
  `config.yaml`. In Phase 1 I'll check them against DeepSeek's `/models`
  endpoint with your key and adjust if the exact names differ.
- **Cost guardrails defaults:** concurrency 4, max output tokens 4000 per
  step (raised from 1500 in Phase 1: deepseek-flash's hidden reasoning tokens count
  against the cap), 2 retries with exponential backoff, `--limit` on every script that
  calls the API.
