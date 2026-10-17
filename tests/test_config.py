from __future__ import annotations

import pytest

from forensics.config import PROJECT_ROOT, load_settings


def test_project_config_loads(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    settings = load_settings(PROJECT_ROOT / "config.yaml")
    assert settings.llm.pipeline_model != settings.llm.judge_model  # judge ≠ model under test
    assert settings.llm.pipeline_model in settings.prices
    assert settings.resolve(settings.paths.traces_dir) == PROJECT_ROOT / "traces"


def test_placeholder_key_is_rejected(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-your-key-here")
    settings = load_settings(PROJECT_ROOT / "config.yaml")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        settings.require_api_key()
