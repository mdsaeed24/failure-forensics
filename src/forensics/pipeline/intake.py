"""Step 1: normalize the raw document text. No LLM."""

from __future__ import annotations

import hashlib
import re
import unicodedata

from forensics.config import PipelineSettings
from forensics.pipeline.models import IntakeOutput, RawDocument


class IntakeError(ValueError):
    """The document can't enter the pipeline (empty, too long, ...)."""


def normalize(text: str) -> str:
    """Line endings, Unicode form, trailing spaces, runs of blank lines.

    Only changes that can't alter meaning: NFC, not NFKC, so "½" stays "½".
    """
    text = text.lstrip("﻿")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _title(text: str) -> str | None:
    for line in text.split("\n"):
        if line.startswith("# "):
            return line[2:].strip() or None
    return None


def run(doc: RawDocument, settings: PipelineSettings) -> IntakeOutput:
    text = normalize(doc.text)
    if not text:
        raise IntakeError(f"{doc.doc_id}: document is empty")
    if len(text) > settings.max_document_chars:
        # Rejected rather than truncated: a silently cut document would be a failure
        # that looks like a later step's fault.
        raise IntakeError(
            f"{doc.doc_id}: {len(text)} characters exceeds max_document_chars "
            f"({settings.max_document_chars})"
        )
    return IntakeOutput(
        doc_id=doc.doc_id,
        text=text,
        title=_title(text),
        char_count=len(text),
        word_count=len(text.split()),
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
