from __future__ import annotations

import pytest

from fakes import CLASSIFICATION, EXTRACTION, SUMMARY
from forensics.checks import run_checks
from forensics.faults.registry import FAULTS
from forensics.pipeline.models import (
    ClassificationOutput,
    ExtractionOutput,
    IntakeOutput,
    SummarizationInput,
    SummaryOutput,
)
from forensics.pipeline.prompts import PROMPTS
from test_analyzer import TRUTH
from test_checks import INTAKE_OUT, SOURCE

RAW = {"doc_id": "d", "text": SOURCE}
SUMMARY_IN = {"extraction": EXTRACTION, "classification": CLASSIFICATION}
OUTPUT = {
    "intake": IntakeOutput.model_validate(INTAKE_OUT),
    "extraction": ExtractionOutput.model_validate(EXTRACTION),
    "classification": ClassificationOutput.model_validate(CLASSIFICATION),
    "summarization": SummaryOutput.model_validate(SUMMARY),
}
STEP_INPUT = {"intake": RAW, "extraction": INTAKE_OUT, "summarization": SUMMARY_IN}

HOOK_FAULTS = [f for f in FAULTS.values() if not f.prompt_variant]
PROMPT_FAULTS = [f for f in FAULTS.values() if f.prompt_variant]


def plant(fault, seed=1):
    injection = fault.inject(seed)
    return injection, injection.hook(fault.step, OUTPUT[fault.step])


def test_every_fault_has_a_unique_name_and_a_valid_label():
    assert len(FAULTS) == 16
    assert {f.step for f in FAULTS.values()} == {"intake", "extraction", "classification", "summarization"}
    assert len({f.failure_type for f in FAULTS.values()}) == 5


@pytest.mark.parametrize("fault", HOOK_FAULTS, ids=lambda f: f.name)
def test_hook_faults_apply_and_only_touch_their_own_step(fault):
    injection, changed = plant(fault)
    assert injection.applied and changed != OUTPUT[fault.step] and injection.detail
    other = next(s for s in OUTPUT if s != fault.step)
    assert injection.hook(other, OUTPUT[other]) is OUTPUT[other]


@pytest.mark.parametrize("fault", [f for f in HOOK_FAULTS if f.step != "classification"
                                   and f.name != "summary_drops_amounts"], ids=lambda f: f.name)
def test_deterministic_checks_catch_hook_faults_with_the_right_type(fault):
    """Classification and dropped-summary faults have no deterministic signal; the judge
    must catch those, and the evaluation measures how often it does."""
    _, changed = plant(fault)
    findings = run_checks(fault.step, STEP_INPUT[fault.step], changed.model_dump(mode="json"))
    from forensics.analyzer.taxonomy import primary_type
    assert primary_type(findings) == fault.failure_type, [f.message for f in findings]


def test_faults_are_reproducible_from_their_seed():
    fault = FAULTS["transpose_amount"]
    assert plant(fault, seed=3)[1] == plant(fault, seed=3)[1]


def test_a_fault_with_nothing_to_change_does_not_apply():
    empty = OUTPUT["extraction"].model_copy(update={"amounts": []})
    injection = FAULTS["drop_amount"].inject(1)
    assert injection.hook("extraction", empty) is empty and not injection.applied


@pytest.mark.parametrize("fault", PROMPT_FAULTS, ids=lambda f: f.name)
def test_prompt_faults_select_an_existing_weaker_prompt(fault):
    injection = fault.inject(1)
    options = injection.options()
    assert options.prompt_variants == {fault.step: fault.prompt_variant} and not options.hooks
    assert fault.prompt_variant in PROMPTS[fault.step] and injection.applied


def test_applicability():
    no_money = TRUTH | {"amounts": []}
    assert not FAULTS["drop_amount"].applicable(SOURCE, no_money)
    assert FAULTS["invent_party"].applicable(SOURCE, no_money)
    # title-only can only fail where the title names a different type
    title_only = FAULTS["prompt_title_only"]
    assert not title_only.applicable("# INVOICE\nTotal: £5", TRUTH)
    assert not title_only.applicable("Acme Ltd\n# Delivery Note", TRUTH | {"doc_type": "other"})
    assert title_only.applicable("Acme Ltd\n# INVOICE\nPaid in full", TRUTH | {"doc_type": "receipt"})


def test_summary_input_is_valid_for_checks():
    SummarizationInput.model_validate(SUMMARY_IN)
