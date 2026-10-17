"""Typed settings, loaded from config.yaml plus secrets from .env."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_PLACEHOLDER_KEY = "sk-your-key-here"


class LLMSettings(BaseModel):
    base_url: str = "https://api.deepseek.com"
    pipeline_model: str
    judge_model: str
    max_output_tokens: int = Field(4000, gt=0)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    concurrency: int = Field(4, ge=1)
    max_retries: int = Field(2, ge=0)
    backoff_base_seconds: float = Field(1.0, ge=0.0)
    timeout_seconds: float = Field(180.0, gt=0.0)


class PipelineSettings(BaseModel):
    max_document_chars: int = Field(20_000, gt=0)


class ModelCallSettings(BaseModel):
    model: str
    thinking: bool = True
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(4000, gt=0)


class GenerationSettings(BaseModel):
    max_attempts: int = Field(3, ge=1)
    generator: ModelCallSettings
    verifier: ModelCallSettings


class ModelPrice(BaseModel):
    """USD per 1M tokens."""

    input_per_million: float = Field(ge=0.0)
    output_per_million: float = Field(ge=0.0)


class PathSettings(BaseModel):
    fact_sheets: Path = Path("data/fact_sheets.yaml")
    documents_dir: Path = Path("data/documents")
    ground_truth: Path = Path("data/ground_truth.json")
    samples_dir: Path = Path("data/samples")
    traces_dir: Path = Path("traces")
    reports_dir: Path = Path("reports")


class Settings(BaseModel):
    llm: LLMSettings
    pipeline: PipelineSettings = PipelineSettings()
    generation: GenerationSettings | None = None  # only the Phase 2 scripts need it
    prices: dict[str, ModelPrice] = {}
    paths: PathSettings = PathSettings()
    deepseek_api_key: SecretStr | None = None

    def resolve(self, path: Path) -> Path:
        """Relative paths in config.yaml are relative to the project root, not the cwd."""
        return path if path.is_absolute() else PROJECT_ROOT / path

    def require_api_key(self) -> str:
        key = self.deepseek_api_key.get_secret_value() if self.deepseek_api_key else ""
        if not key or key == _PLACEHOLDER_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and put your key in it."
            )
        return key


def load_settings(config_path: Path | None = None) -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")  # never overrides variables already set in the shell
    path = config_path or Path(os.environ.get("FORENSICS_CONFIG", PROJECT_ROOT / "config.yaml"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Settings(**data, deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY") or None)
