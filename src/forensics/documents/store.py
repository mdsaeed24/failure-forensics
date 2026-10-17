"""Where generated documents, their state and the ground truth live.

    data/documents/<doc_id>.md      the document (the last attempt, even if it failed)
    data/documents/manifest.json    per document: hashes, verification, review decision
    data/ground_truth.json          the answer key: approved, up-to-date documents only

A document enters the ground truth only if a human approved it AND neither its
fact sheet nor the .md file changed since. Edit either one and it drops out
until it is regenerated or re-reviewed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from forensics.documents.fact_sheets import FactSheet
from forensics.documents.generator import GenerationOutcome
from forensics.documents.verifier import Verification

ReviewStatus = Literal["pending", "approved", "rejected"]


class Review(BaseModel):
    status: ReviewStatus = "pending"
    note: str | None = None
    at: dt.datetime | None = None


class ManifestEntry(BaseModel):
    doc_id: str
    status: Literal["verified", "failed"]
    sheet_sha256: str
    doc_sha256: str
    attempts: int
    rejected_attempts: list[list[str]] = []
    generator_model: str
    generated_at: dt.datetime
    cost_usd: float | None
    verification: Verification
    review: Review = Review()


class DocumentStore:
    def __init__(self, documents_dir: Path, ground_truth_path: Path) -> None:
        self.dir = documents_dir
        self.ground_truth_path = ground_truth_path
        self.manifest_path = documents_dir / "manifest.json"
        self.entries: dict[str, ManifestEntry] = {}
        if self.manifest_path.is_file():
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.entries = {k: ManifestEntry.model_validate(v) for k, v in raw.items()}

    def doc_path(self, doc_id: str) -> Path:
        return self.dir / f"{doc_id}.md"

    def read_document(self, doc_id: str) -> str | None:
        path = self.doc_path(doc_id)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def is_current(self, sheet: FactSheet) -> bool:
        """True if the stored document was generated from this exact sheet and is unedited."""
        entry = self.entries.get(sheet.doc_id)
        text = self.read_document(sheet.doc_id)
        return (
            entry is not None and text is not None
            and entry.sheet_sha256 == sheet.sha256() and entry.doc_sha256 == _sha256(text)
        )

    def record(self, sheet: FactSheet, outcome: GenerationOutcome, generator_model: str) -> None:
        """Saves a generation outcome. A new document always starts unreviewed."""
        if outcome.document is None or outcome.verification is None:
            return  # an API error: nothing was produced, keep whatever was there
        self.dir.mkdir(parents=True, exist_ok=True)
        self.doc_path(sheet.doc_id).write_text(outcome.document, encoding="utf-8")
        self.entries[sheet.doc_id] = ManifestEntry(
            doc_id=sheet.doc_id,
            status="verified" if outcome.status == "verified" else "failed",
            sheet_sha256=sheet.sha256(),
            doc_sha256=_sha256(outcome.document),
            attempts=outcome.attempts,
            rejected_attempts=outcome.rejected_attempts,
            generator_model=generator_model,
            generated_at=dt.datetime.now(dt.UTC),
            cost_usd=outcome.cost_usd,
            verification=outcome.verification,
        )
        self.save()

    def set_review(self, doc_id: str, status: ReviewStatus, note: str | None = None) -> None:
        if doc_id not in self.entries:
            raise KeyError(f"no generated document {doc_id!r}")
        self.entries[doc_id].review = Review(status=status, note=note, at=dt.datetime.now(dt.UTC))
        self.save()

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        data = {k: v.model_dump(mode="json") for k, v in sorted(self.entries.items())}
        _write_json(self.manifest_path, data)

    def build_ground_truth(self, sheets: list[FactSheet]) -> tuple[int, list[str]]:
        """Writes ground_truth.json. Returns (documents included, reasons others were left out)."""
        truth, skipped = {}, []
        for sheet in sheets:
            entry = self.entries.get(sheet.doc_id)
            if entry is None:
                skipped.append(f"{sheet.doc_id}: not generated yet")
            elif entry.review.status != "approved":
                skipped.append(f"{sheet.doc_id}: review {entry.review.status}")
            elif not self.is_current(sheet):
                skipped.append(f"{sheet.doc_id}: fact sheet or document changed since approval")
            else:
                truth[sheet.doc_id] = {
                    **sheet.model_dump(mode="json", exclude={"style"}),
                    "sheet_sha256": entry.sheet_sha256,
                    "doc_sha256": entry.doc_sha256,
                }
        _write_json(self.ground_truth_path, truth)
        return len(truth), skipped


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written file
