"""Check the model IDs in config.yaml against DeepSeek's /models endpoint.

    python scripts/check_models.py

Listing models is free: no tokens are used.
"""

from __future__ import annotations

import sys

from openai import OpenAI

from forensics.config import load_settings


def main() -> int:
    settings = load_settings()
    try:
        client = OpenAI(api_key=settings.require_api_key(), base_url=settings.llm.base_url)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    available = sorted(m.id for m in client.models.list())
    print("Models available to your key:")
    for model_id in available:
        print(f"  {model_id}")

    missing = 0
    print("\nConfigured in config.yaml:")
    for role, model_id in [("pipeline_model", settings.llm.pipeline_model),
                           ("judge_model", settings.llm.judge_model)]:
        found = model_id in available
        missing += not found
        print(f"  {role:<15} {model_id:<25} {'OK' if found else 'NOT FOUND'}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
