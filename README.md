# Failure Forensics

**When a multi-step LLM pipeline gives a wrong answer, which step broke it, and how?**
Failure Forensics traces every step of a 4-step document pipeline, walks back from a
bad output to the *first* step that failed, names the failure type, shows the
evidence, and turns confirmed failures into regression tests. Its diagnosis accuracy
is **measured** on failures planted on purpose, not assumed.

| Measured on 45 documents, 16 kinds of planted failure | Result |
|---|---|
| Right step, on planted failures that broke the final output | **82 / 82** (95% CI 96–100%) |
| Right step **and** failure type | **82 / 82** |
| False alarms on correct runs | **3 / 44** (7%, CI 2–18%): 1 medium-confidence, 2 low |
| Cost of the whole evaluation (≈140 pipeline runs + analysis) | **$0.20** |

Read the 100% with the caveats in [What the numbers don't say](#what-the-numbers-dont-say):
most of those failures were caught by exact checks, and the two hardest kinds of
fault never actually took effect.

```
docker compose up        # UI at http://localhost:8501, preloaded with the real runs above
```

---

## The problem

A document pipeline here is four steps: **intake → extraction → classification →
summarization**. When the final summary says the wrong total, the error could have
started anywhere: intake may have cut the document short, extraction may have read a
digit wrong, classification may have called an invoice a quotation, or the summarizer
may have invented a fee. Every step after the first mistake usually looks wrong too,
because it faithfully carried the bad value forward. Blaming the last step, or every
step, doesn't help anyone fix it.

## How it works

```
           ┌───────────────────── pipeline ─────────────────────┐
document → │ intake → extraction → classification → summarization │ → result
           └──────────────────────────┬─────────────────────────┘
                                      │ OpenTelemetry spans: every step's input, output,
                                      ▼ tokens, cost, retries, prompt, raw reply, reasoning
                    custom exporter → traces/*.json + SQLite index
                                      │
                 output bad? (vs ground truth, or a human flag)
                                      ▼
            root-cause analyzer: step by step, in order
            1. did the step crash?                          → failed  (high confidence)
            2. deterministic checks vs its OWN input        → failed  (high)
            3. LLM judge: faithful transformation of input? → failed ≤2 (medium), suspect 3 (low)
            stop at the first failure; follow the bad value downstream
                                      ▼
            diagnosis + evidence chain → human confirms/overrides → eval case
                                                                      ▼
                                              regression: re-run, is the issue gone?
```

**The key idea: every step is judged against its own input, never against the final
answer.** A summarizer that faithfully repeats a wrong amount it was given did its job.
That's what lets the analyzer find the *first* failure instead of blaming everything
downstream. And it's why the analyzer never needs the ground truth: it works the same
on a production trace a user has flagged.

An evidence chain, from the real demo data (a quotation headed "Invoice / Proposal",
with its classification forced to `invoice`):

```
Symptom: classified as "invoice", should be "quotation"
Step 1 intake: OK (text handed on intact)
Step 2 extraction: OK (every value grounded in the document, nothing missed, judge 5/5)
Step 3 classification: FAILED (misclassification), judge 2/5
    - judge: The document is a price offer requiring acceptance and signature, so it
      should be classified as 'quotation', not 'invoice'.
    - judge quotes: "This document is a price offer only. Nothing is payable until the
      client accepts and signs this proposal."
Step 4 summarization: not checked (after the root cause)
Downstream: summarization was told the document type is "invoice", and the summary
repeats it: "This invoice is issued by Atelier Lumière..."
```

### The five failure types

| type | meaning |
|---|---|
| hallucination | the output states something its input doesn't support |
| omission | the output leaves out something its input clearly contains |
| wrong_value | the right fact, wrongly carried over (a transposed digit, the wrong currency) |
| misclassification | the document got the wrong type |
| format_error | malformed output: unusable, cut off, a non-ISO date or currency |

The deterministic checks separate *wrong* from *invented*: an amount whose quoted text
exists in the document but says something else is `wrong_value`; an amount with no
support anywhere is `hallucination`.

## Measuring it honestly

You can't measure a diagnoser on real failures, because nobody knows their true root
cause. So the project builds data where the answer is known:

1. **Ground truth by construction.** 45 fact sheets (all 7 document types, 9
   currencies, 10 deliberate near-misses such as a receipt headed "INVOICE") are written
   first. A stronger model writes each document *from* its sheet. Code checks that
   every fact is present and nothing else is (no extra names, dates or amounts), an
   independent LLM audit checks the type, and a human approves each one.
2. **16 planted faults with known labels.** 13 edit one step's output before the next
   step sees it (a transposed digit, an invented party, dates in the wrong format, a
   swapped label...). 3 are weakened prompts a team might plausibly ship ("keep it
   short: only the main facts"), whose effect is up to the model.
3. **The analyzer can't see the answer key.** Fault labels live in a separate database
   file that the analyzer's code has no way to read. Neither hooks nor prompt variants
   are recorded in the trace (a test asserts it).
4. **Only faults that took count.** A fault is counted only if it changed something *and*
   the final output really was wrong, on a document whose clean run was correct.

Full report: [reports/diagnosis_accuracy.md](reports/diagnosis_accuracy.md), with
per-fault tables, precision/recall, confusion matrices and calibration.

## What the numbers don't say

Digging into the results mattered more than the headline:

- **Exact checks did most of the work.** 70 of the 82 diagnoses were decided by
  deterministic checks, for free and with certainty. The edit-the-output faults are clean
  and mechanical. Real model errors are subtler, so this is an upper bound.
- **The judge was tested on easy cases only.** Its 12 cases (a swapped label, a summary
  with its amounts deleted) were all right. But the realistic version, a weakened
  "classify by the title" prompt run on documents with *misleading* titles, never took:
  `deepseek-flash` still classified by purpose, 4 out of 4. The judge has never faced a
  subtle misclassification.
- **The scoring has a blind spot.** It checks type, names, dates and amounts. The
  weakened summary prompt *did* make summaries invent things ("the remaining balance
  payable under the contractor's standard terms"), but claims made in words pass, so
  those runs scored as fine.
- **One "false alarm" was a real defect.** A clean run's summary read *"The donation
  amount of 25000 INR is recorded as the receipt date of 2026-03-15."* The scoring called
  it correct; the judge flagged it. Of the other two false alarms, one was the judge
  misreading the extraction ("omits the subtotal", which was there), and one was
  debatable.
- **Faults don't always propagate.** A transposed total in the extraction was *undone*
  by the summarizer, which read the true value from the extracted quote. Counting only
  faults that took is what keeps this from showing up as a miss.
- **The model's self-confidence is useless.** It rated 60 of 63 extractions 5/5,
  including all 7 that were wrong. The project records it and measures it, but never
  trusts it.
- **One natural failure was missed.** With no fault planted, extraction listed "All Staff"
  (a memo's audience) as a party. The name is in the text, so grounding passed, and the
  judge didn't object. A human flag caught it, and it's now an open eval case that
  regression, told what to look for, does catch (see below).

## The feedback loop

Flag a bad run in the UI or over HTTP, confirm or correct the diagnosis, and it becomes
an eval case in [data/eval_cases.json](data/eval_cases.json), committed with the code.
Regression re-runs each case's document through the current pipeline and checks the
failing step again, this time *telling* the judge what the human reported:

```
✓ case_001  rct_004  summarization/wrong_value  open -> resolved  the step now passes its checks and the judge
✗ case_002  oth_001  extraction/hallucination   open              judge 2/5: The output includes 'All Staff' as a
                                                                  party, but the document uses it only as the
                                                                  memo's audience, not a named person or organization.
Known issues resolved: 1/2
```

The analyzer has to stay blind so its accuracy means something. A regression check
should look for the specific problem. Note that `case_001` "resolved" because the garbled
sentence didn't happen again on the re-run: nothing was fixed. A flaky issue needs
several passes in a row before it's really closed.

## Design decisions

1. **Steps are pure; the runner is the only place that knows their order.** Fault
   injection plugs into its hook points and prompt variants without touching step code.
2. **Schemas check shape, not correctness.** A date like `2026-02-30` passes the
   Pydantic model on purpose, so a wrong answer shows up in a trace as the failure it is,
   instead of turning into a retry or a crash.
3. **The trace records what the next step actually received.** Step spans live in the
   runner, around the hooks, not in a decorator on the step functions. Otherwise a
   planted fault would be invisible in the trace.
4. **Cheap and certain before expensive and fuzzy.** Deterministic checks run first, and
   the walk stops at the first failed step, so most diagnoses cost nothing.
5. **The judge is a different, stronger model** (`deepseek-v4-pro`; the pipeline runs on
   `deepseek-flash`). It sees the step's job, input and output, but not the step's
   prompt (a weakened prompt is what it should catch) or its self-confidence (which would
   anchor it).
6. **Hidden reasoning is evidence.** `deepseek-flash` spends most of its output tokens on
   reasoning (827 of 1,213 in one call), which counts against the token cap: the cap
   had to go from 1,500 to 4,000. Each call's reasoning text is kept in the trace.
7. **Standard OpenTelemetry.** LLM spans use the `gen_ai.*` semantic conventions, so the
   same spans can go to Jaeger or another backend with one extra exporter.

## Run it

**Docker.** Starts the UI and API with the demo data, and needs no API key to browse:

```
docker compose up
# UI   http://localhost:8501
# API  http://localhost:8000/docs
# ports taken?  UI_PORT=8601 API_PORT=8100 docker compose up
```

To analyze traces or run regression from the containers, put your key in `.env` first.

**Locally** (Python 3.11+):

```
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                   # add DEEPSEEK_API_KEY
python scripts/seed_demo.py                            # optional: load the demo traces
python -m streamlit run src/forensics/ui/app.py
```

| command | what it does | cost |
|---|---|---|
| `python -m pytest` | 193 tests; they never call the API (a fake LLM) | free |
| `python scripts/check_models.py` | checks the model ids in `config.yaml` against the API | free |
| `python scripts/run_pipeline.py --doc data/documents/inv_007.md` | runs one document, prints every step, writes its trace | ≈ $0.001 |
| `python scripts/list_traces.py --show <id>` | per-step signals of a trace | free |
| `python scripts/analyze_trace.py <id>` | scores a trace; if it's bad, finds the root cause | ≈ $0.001–0.003 |
| `python scripts/generate_documents.py --all` | fact sheets → verified documents | ≈ $0.04 for 45 |
| `python scripts/review_documents.py` | approve or reject documents side by side with their fact sheets | free |
| `python scripts/evaluate_diagnosis.py --per-fault 6` | the full accuracy evaluation | ≈ $0.25 |
| `python scripts/regression.py` | re-checks every open eval case | ≈ $0.002 per case |
| `python -m uvicorn forensics.api.main:app` | HTTP API: flag, analyze, confirm, list | |

Scripts that can run many documents take a `--limit` (default 1), and the evaluation
prints an estimate and asks before spending. A `Makefile` has a target for each
command, and each target is one plain command, for Windows.

## Project layout

```
src/forensics/
  llm/          DeepSeek client: JSON mode, Pydantic validation, one repair retry,
                backoff, concurrency cap, truncation flag, cost, reasoning capture
  pipeline/     the 4 steps, their models and prompts (+ weakened variants), the runner
  tracing/      OpenTelemetry setup, span helpers, custom exporter, trace store
  documents/    Phase 2: fact sheets, generator, verifier, review screen
  checks/       deterministic checks: intake integrity, grounding, dates, currency, summary
  analyzer/     failure taxonomy, LLM judge, root-cause walk, evidence chain
  scoring/      final output vs ground truth
  faults/       16 fault injectors with their labels
  evaluation/   evaluation runner, label store, metrics, report
  feedback/     flags → eval cases, regression
  api/          FastAPI
  ui/           Streamlit: traces, compare, review, analytics
scripts/        thin command-line entry points
data/           fact sheets, generated documents, ground truth, eval cases
demo/           snapshot of real traces for the UI (no local paths in it)
reports/        document verification and diagnosis accuracy reports
```

## Limitations and next steps

- **Harder faults for the judge.** The judge needs faults that are subtle but still take:
  a near-miss document relabeled with a plausible reason, or a summary that mixes up two
  parties. These would need a more capable pipeline model to be broken in realistic
  ways.
- **Scoring beyond numbers.** Checking a summary's claims, not just its dates and
  amounts, would catch the embellishments above, probably with an LLM grader that is
  itself measured.
- **Repeat runs.** The pipeline isn't fully deterministic. Running each evaluation case
  several times, and requiring regression cases to pass N times in a row, would separate
  flaky failures from fixed ones.
- **Generated documents are clean.** Real inputs (scans, email threads, OCR noise) would
  stress intake and extraction far more.
- **Prices in `config.yaml` are placeholders.** Costs above use them; check them against
  DeepSeek's current pricing.
