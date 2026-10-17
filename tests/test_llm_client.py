from __future__ import annotations

import asyncio

import openai
import pytest
from pydantic import BaseModel, SecretStr

from fakes import FakeReply, api_status_error, connection_error
from forensics.llm.client import LLMAPIError, LLMClient, LLMOutputError, record_calls


class Answer(BaseModel):
    value: int


def ask(llm: LLMClient):
    return asyncio.run(llm.complete_json(system="Reply in json.", user="?", schema=Answer))


def test_valid_reply_is_parsed_and_costed(make_llm):
    llm, fake = make_llm(FakeReply({"value": 7}, prompt_tokens=1000, completion_tokens=500))
    result = ask(llm)
    assert result.parsed == Answer(value=7)
    assert result.call.attempts == 1
    assert not result.call.repaired and not result.call.truncated
    assert result.call.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1e6)
    request = fake.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["max_tokens"] == 4000
    assert request["temperature"] == 0.0
    assert "extra_body" not in request  # thinking stays on unless turned off


def test_per_call_options_override_config(make_llm):
    llm, fake = make_llm({"value": 1})
    asyncio.run(llm.complete_json(system="json", user="?", schema=Answer, max_tokens=99,
                                  temperature=0.9, thinking=False))
    request = fake.requests[0]
    assert (request["max_tokens"], request["temperature"]) == (99, 0.9)
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_bad_json_gets_one_repair_retry_that_shows_the_error(make_llm):
    llm, fake = make_llm("{not json", {"value": 3})
    result = ask(llm)
    assert result.parsed.value == 3
    assert result.call.repaired and result.call.attempts == 2
    repair_messages = fake.requests[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": "{not json"}
    assert "could not be used" in repair_messages[-1]["content"]


def test_schema_mismatch_also_triggers_repair(make_llm):
    llm, _ = make_llm({"value": "seven"}, {"value": 7})
    assert ask(llm).parsed.value == 7


def test_second_invalid_reply_raises_with_raw_output(make_llm):
    llm, fake = make_llm("{not json", "still not json")
    with pytest.raises(LLMOutputError) as info:
        ask(llm)
    assert info.value.call.raw_output == "still not json"
    assert info.value.call.attempts == 2
    assert not fake.replies  # exactly one repair retry, no more


def test_empty_reply_is_invalid(make_llm):
    llm, _ = make_llm("", {"value": 1})
    assert ask(llm).call.repaired


def test_tokens_are_summed_across_attempts(make_llm):
    llm, _ = make_llm(FakeReply("bad", prompt_tokens=100, completion_tokens=10),
                      FakeReply({"value": 1}, prompt_tokens=150, completion_tokens=20))
    call = ask(llm).call
    assert (call.prompt_tokens, call.completion_tokens) == (250, 30)


def test_reasoning_tokens_and_text_are_recorded(make_llm):
    llm, _ = make_llm(FakeReply({"value": 1}, completion_tokens=300, reasoning_tokens=200,
                                reasoning="The answer is one."))
    call = ask(llm).call
    assert call.reasoning_tokens == 200 and call.completion_tokens == 300
    assert call.reasoning == "The answer is one."


def test_models_without_reasoning_record_zero(make_llm):
    call = ask(make_llm({"value": 1})[0]).call
    assert call.reasoning_tokens == 0 and call.reasoning == ""


def test_truncation_is_flagged(make_llm):
    llm, _ = make_llm(FakeReply({"value": 1}, finish_reason="length"))
    call = ask(llm).call
    assert call.truncated and call.finish_reason == "length"


@pytest.mark.parametrize("error", [
    lambda: api_status_error(openai.RateLimitError, 429),
    lambda: api_status_error(openai.InternalServerError, 503),
    connection_error,
])
def test_transient_errors_are_retried(make_llm, error):
    llm, _ = make_llm(error(), error(), {"value": 1})
    result = ask(llm)
    assert result.parsed.value == 1 and result.call.attempts == 3


def test_retries_give_up_after_max_retries(make_llm):
    llm, fake = make_llm(*(api_status_error(openai.RateLimitError, 429) for _ in range(4)))
    with pytest.raises(LLMAPIError, match="gave up after 3 attempts"):
        ask(llm)
    assert len(fake.replies) == 1  # max_retries=2 → 3 attempts, the 4th reply is never requested


def test_non_transient_errors_fail_immediately(make_llm):
    llm, fake = make_llm(api_status_error(openai.AuthenticationError, 401), {"value": 1})
    with pytest.raises(LLMAPIError):
        ask(llm)
    assert len(fake.requests) == 1


def test_concurrency_is_capped(make_llm):
    llm, fake = make_llm(*(FakeReply({"value": i}, delay=0.02) for i in range(8)))

    async def many():
        return await asyncio.gather(*(
            llm.complete_json(system="json", user="?", schema=Answer) for _ in range(8)
        ))

    asyncio.run(many())
    assert fake.max_in_flight == 2  # settings fixture: concurrency=2


def test_record_calls_captures_failed_calls_too(make_llm):
    llm, _ = make_llm({"value": 1}, "bad", "bad")

    async def two_calls():
        with record_calls() as calls:
            await llm.complete_json(system="json", user="?", schema=Answer)
            with pytest.raises(LLMOutputError):
                await llm.complete_json(system="json", user="?", schema=Answer)
        return calls

    calls = asyncio.run(two_calls())
    assert [c.repaired for c in calls] == [False, True]


def test_unknown_model_price_is_none_not_zero(make_llm):
    llm, _ = make_llm({"value": 1})
    result = asyncio.run(
        llm.complete_json(system="json", user="?", schema=Answer, model="unpriced-model")
    )
    assert result.call.cost_usd is None


def test_real_client_is_blocked_in_tests(settings):
    settings = settings.model_copy(update={"deepseek_api_key": SecretStr("sk-real-looking")})
    with pytest.raises(RuntimeError, match="must not call the real API"):
        LLMClient.from_settings(settings)
