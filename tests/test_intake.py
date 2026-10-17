from __future__ import annotations

import pytest

from forensics.config import PipelineSettings
from forensics.pipeline import intake
from forensics.pipeline.intake import IntakeError
from forensics.pipeline.models import RawDocument


def run(text: str, max_chars: int = 20_000):
    return intake.run(RawDocument(doc_id="d1", text=text), PipelineSettings(max_document_chars=max_chars))


def test_normalizes_whitespace_and_line_endings():
    out = run("﻿# Invoice  \r\n\r\n\r\n\r\nTotal: £10   \r\n")
    assert out.text == "# Invoice\n\nTotal: £10"
    assert out.title == "Invoice"
    assert out.word_count == 4


def test_does_not_change_meaningful_characters():
    assert run("½ of £1,000 — due 3 März").text == "½ of £1,000 — due 3 März"


def test_hash_is_stable_across_line_ending_styles():
    assert run("a\r\nb").sha256 == run("a\nb").sha256


def test_empty_document_is_rejected():
    with pytest.raises(IntakeError, match="empty"):
        run("  \n\n  ")


def test_too_long_document_is_rejected_not_truncated():
    with pytest.raises(IntakeError, match="exceeds"):
        run("x" * 101, max_chars=100)
