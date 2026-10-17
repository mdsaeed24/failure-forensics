# Diagnosis accuracy

Evaluation `eval-20261017-1114-272d`, started 2026-10-17 11:14 UTC. Pipeline model `deepseek-flash`, judge `deepseek-v4-pro`. 45 documents, up to 6 per fault, seed 7. Total cost $0.20.

## Headline

- **Right step: 100%** of 82 planted failures that actually broke the final output (95% CI 96%–100%).
- **Right step and failure type: 100%** (95% CI 96%–100%).
- **False alarms: 7%** of 44 clean runs with a correct output were blamed on a step anyway (95% CI 2%–18%); by confidence: 1 medium, 2 low.
- **Natural failures:** 1 clean run(s) were wrong with no fault planted (listed below).

Accuracy counts only runs where the planted fault *took*: it changed something and the final output was really wrong. A fault that the next step happened to undo is a non-event, not a miss.

## By fault

| fault | breaks | type | how | runs | applied | took | right step | right step+type | decided by |
|---|---|---|---|---|---|---|---|---|---|
| `truncate_document` | intake | omission | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `drop_amount` | extraction | omission | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `invent_amount` | extraction | hallucination | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `invent_party` | extraction | hallucination | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `numeric_dates` | extraction | format_error | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `prompt_main_facts_only` | extraction | omission | prompt | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `shift_date` | extraction | wrong_value | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `swap_currency` | extraction | wrong_value | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `symbol_currency` | extraction | format_error | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `transpose_amount` | extraction | wrong_value | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `confusable_label` | classification | misclassification | hook | 6 | 6 | 6 | 6/6 | 6/6 | judge 6 |
| `prompt_title_only` | classification | misclassification | prompt | 4 | 4 | 0 | - | - | - |
| `prompt_embellish` | summarization | hallucination | prompt | 6 | 6 | 0 | - | - | - |
| `summary_drops_amounts` | summarization | omission | hook | 6 | 6 | 6 | 6/6 | 6/6 | judge 6 |
| `summary_invents_fee` | summarization | hallucination | hook | 6 | 6 | 6 | 6/6 | 6/6 | deterministic 6 |
| `summary_wrong_amount` | summarization | wrong_value | hook | 6 | 4 | 4 | 4/4 | 4/4 | deterministic 4 |

## By what decided

| decided by | runs | right step+type |
|---|---|---|
| deterministic | 70 | 100% |
| judge | 12 | 100% |

Deterministic checks decide most extraction faults for free; classification and some summary faults need the judge.

## Precision and recall by failure type

Rows include clean correct runs, whose right answer is `none`.

| type | precision | recall |
|---|---|---|
| hallucination | 100% (18/18) | 100% (18/18) |
| omission | 96% (24/25) | 100% (24/24) |
| wrong_value | 96% (22/23) | 100% (22/22) |
| misclassification | 86% (6/7) | 100% (6/6) |
| format_error | 100% (12/12) | 100% (12/12) |
| none | 100% (41/41) | 93% (41/44) |

## Confusion matrix: failure type

| planted ↓ / diagnosed → | hallucination | omission | wrong_value | misclassification | format_error | none |
|---|---|---|---|---|---|---|
| hallucination | **18** | · | · | · | · | · |
| omission | · | **24** | · | · | · | · |
| wrong_value | · | · | **22** | · | · | · |
| misclassification | · | · | · | **6** | · | · |
| format_error | · | · | · | · | **12** | · |
| none | · | 1 | 1 | 1 | · | **41** |

## Confusion matrix: step

| planted ↓ / diagnosed → | intake | extraction | classification | summarization |
|---|---|---|---|---|
| intake | **6** | · | · | · |
| extraction | · | **54** | · | · |
| classification | · | · | **6** | · |
| summarization | · | · | · | **16** |

## Is the analyzer's confidence meaningful?

| analyzer confidence | times it named a cause | right step+type |
|---|---|---|
| high | 70 | 100% |
| medium | 13 | 92% |
| low | 2 | 0% |

(False alarms count against the level the analyzer gave them.)

## Is the pipeline model's self-reported confidence meaningful?

The pipeline model rates its own output 1-5. Only outputs it really produced count (clean runs and prompt faults). If confidence were informative, low ratings would come with more wrong outputs.

**extraction**

| self-confidence | outputs | wrong |
|---|---|---|
| 4 | 4 | 0% (0) |
| 5 | 57 | 12% (7) |

**classification**

| self-confidence | outputs | wrong |
|---|---|---|
| 4 | 4 | 0% (0) |
| 5 | 57 | 0% (0) |

## Natural failures (no fault planted)

These are the pipeline's own mistakes. There's no label to score the diagnosis against; they are worth a human look.

| doc | what was wrong | analyzer says |
|---|---|---|
| oth_001 | "All Staff" was extracted but is not a party | no step-level failure |

## Misdiagnoses

None.

## False alarms

| doc | blamed | trace |
|---|---|---|
| inv_004 | extraction / omission (low) | `e227020c` |
| oth_003 | summarization / misclassification (medium) | `ae8ca815` |
| rct_004 | summarization / wrong_value (low) | `78cf233b` |

## Failure types

- **hallucination**: the output states something its input does not support: an invented name, date, amount or claim
- **omission**: the output leaves out something its input clearly contains
- **wrong_value**: the right fact, wrongly carried over: a date converted wrongly, an amount or currency changed, a value attached to the wrong label
- **misclassification**: the document was given the wrong type
- **format_error**: the output is malformed: unusable or cut-off output, a date not in YYYY-MM-DD, a currency that is not an ISO code

## Limitations

- The final-output score checks document type, parties, dates and amounts only. A summary that invents a claim in words ("payable under the contractor's standard terms") or garbles a sentence still scores as correct. So some "false alarms" may be real defects the scoring can't see, and a prompt fault can "not take" while still degrading the output. Check the false-alarm traces before trusting the rate.
- Hook faults edit outputs in clean, mechanical ways (a transposed digit, an invented name). Real model errors are often subtler, so deterministic-check accuracy here is likely an upper bound.
- Per-fault samples are small; read the intervals, not just the percentages.
- The judge is also an LLM, and its verdicts can vary between runs.
- Documents were generated from fact sheets: clean and well-formatted compared to real scans or emails.
