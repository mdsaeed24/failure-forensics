from __future__ import annotations

from typing import Any

import pytest

from fakes import FakeLLM
from forensics.config import LLMSettings, ModelPrice, Settings
from forensics.llm.client import LLMClient


@pytest.fixture(autouse=True)
def _no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests NEVER call the real API: building a real client fails loudly."""

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Tests must not call the real API; use the make_llm fixture.")

    monkeypatch.setattr("forensics.llm.client.AsyncOpenAI", refuse)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm=LLMSettings(
            pipeline_model="fake-flash",
            judge_model="fake-pro",
            concurrency=2,
            max_retries=2,
            backoff_base_seconds=0.0,  # no real waiting in tests
        ),
        prices={"fake-flash": ModelPrice(input_per_million=1.0, output_per_million=2.0)},
    )


@pytest.fixture
def make_llm(settings: Settings):
    """make_llm(reply, reply, ...) -> (LLMClient, FakeLLM) with replies scripted in order."""

    def make(*replies: Any) -> tuple[LLMClient, FakeLLM]:
        fake = FakeLLM(list(replies))
        return LLMClient(fake, settings), fake

    return make
