"""Tokens → dollars."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from forensics.config import ModelPrice

log = logging.getLogger(__name__)

_warned: set[str] = set()


def cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int, prices: Mapping[str, ModelPrice]
) -> float | None:
    """Returns None (not 0.0) for a model with no price, so missing prices can't hide as "free"."""
    price = prices.get(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            log.warning("No price configured for model %r; cost will be reported as unknown", model)
        return None
    return (
        prompt_tokens * price.input_per_million + completion_tokens * price.output_per_million
    ) / 1_000_000
