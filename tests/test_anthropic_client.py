"""Тесты клиента Claude на поддельном клиенте SDK anthropic."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.config import LlmConfig
from assistant.journal import ActionJournal
from assistant.llm.agent import Agent, AgentStatus
from assistant.llm.anthropic_client import BETA_BINDING, BETA_FALLBACK, AnthropicClient
from assistant.llm.client import LlmCancelled, LlmError, LlmRefused, create_client
from assistant.logging_setup import SecretMasker
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.anthropic_fakes import (
    FakeAnthropic,
    FakeClaudeStream,
    factory_for,
    message,
    reply,
    status_error,
    text,
    thinking,
    tool_use,
)
from tests.conftest import RecordingSpeaker, wait_until

TOOL = {
    "type": "function",
    "function": {
        "name": "open_app",
        "description": "Открыть программу",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}


@pytest.fixture(autouse=True)
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-123456")


def call(call_id: str, name: str) -> dict[str, Any]:
    arguments = json.dumps({"name": name})
    return {"id": call_id, "type": "function", "function": {"name": "open_app", "arguments": arguments}}


def make(responses: list[Any], **profile: Any) -> tuple[AnthropicClient, FakeAnthropic]:
    fake = FakeAnthropic(responses)
    settings = LlmConfig.model_validate({"profile": "anthropic", "profiles": {"anthropic": profile}})
    return AnthropicClient(settings, factory=factory_for(fake)), fake


def test_request_follows_claude_rules() -> None:
    client, fake = make([reply(text("Готово."))])
    client.complete(
        [
            {"role": "system", "content": "Ты — Джарвис."},
            {"role": "user", "content": "открой блокнот"},
            {"role": "assistant", "content": None, "tool_calls": [call("c1", "A"), call("c2", "B")]},
            {"role": "tool", "tool_call_id": "c1", "content": "Открываю A"},
            {"role": "tool", "tool_call_id": "c2", "content": "Ошибка: нет B", "_is_error": True},
        ],  # fmt: skip
        tools=[TOOL],
    )
    request = fake.requests[0]
    assert request["model"] == "claude-opus-5-5" and request["max_tokens"] == 16000
    assert request["system"] == "Ты — Джарвис."
    assert request["output_config"] == {"effort": "low"}
    assert "temperature" not in request  # модели Claude 5 её не принимают
    binding = {"prefix_mismatch_behavior": "drop_block"}
    assert request["thinking"] == {"type": "adaptive", "block_binding": binding}
    assert request["betas"] == [BETA_BINDING, BETA_FALLBACK] and request["fallbacks"] == "default"
    expected_tool = {"name": "open_app", "description": "Открыть программу"}
    assert request["tools"] == [{**expected_tool, "input_schema": TOOL["function"]["parameters"]}]
    assistant, results = request["messages"][1:]
    assert [block["type"] for block in assistant["content"]] == ["tool_use", "tool_use"]
    assert assistant["content"][0]["input"] == {"name": "A"}
    # Результаты одного хода — одним сообщением; неудача помечена is_error.
    assert results == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "Открываю A"},
        {"type": "tool_result", "tool_use_id": "c2", "content": "Ошибка: нет B", "is_error": True},
    ]}  # fmt: skip
    assert fake.client_kwargs["api_key"] == "sk-ant-test-key-123456"


def test_text_streams_and_tool_calls_are_collected() -> None:
    heard: list[str] = []
    blocks = (thinking(), text("Открываю."), tool_use("open_app", {"name": "Блокнот"}))
    client, _ = make([reply(*blocks, pieces=("Откры", "ваю."))])
    asked = [{"role": "user", "content": "открой блокнот"}]
    result = client.complete(asked, tools=[TOOL], on_text=heard.append)
    assert heard == ["Откры", "ваю."] and result.text == "Открываю."
    assert [(c.id, c.name, json.loads(c.arguments)) for c in result.tool_calls] == [
        ("toolu_1", "open_app", {"name": "Блокнот"})
    ]
    message_ = result.assistant_message()
    assert [block.type for block in message_["_native"]] == ["thinking", "text", "tool_use"]
    assert result.first_token_ms is not None


def test_thinking_is_replayed_only_inside_current_task() -> None:
    # Рассуждения прошлого обмена убираются (удаление с начала разрешено),
    # а в текущей задаче возвращаются без изменений.
    old = message(thinking("old"), text("Первый ответ.")).content
    current = message(thinking("new"), tool_use("open_app", {"name": "A"})).content
    client, fake = make([reply(text("Готово."))])
    client.complete([
        {"role": "user", "content": "раз"},
        {"role": "assistant", "content": "Первый ответ.", "_native": old},
        {"role": "user", "content": "два"},
        {"role": "assistant", "content": None, "_native": current},
        {"role": "tool", "tool_call_id": "toolu_1", "content": "ok"},
    ])  # fmt: skip
    messages = fake.requests[0]["messages"]
    assert [block.type for block in messages[1]["content"]] == ["text"]
    assert [block.type for block in messages[3]["content"]] == ["thinking", "tool_use"]
    assert messages[3]["content"][0].signature == "new"


def test_refusal_is_explained() -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": "-"}
    refused = reply(stop_reason="refusal", stop_details=details)
    client, _ = make([refused])
    with pytest.raises(LlmRefused, match="cyber") as info:
        client.complete([{"role": "user", "content": "?"}])
    assert info.value.spoken == "С этим я помочь не могу"


def test_truncated_tool_call_is_not_run() -> None:
    client, _ = make([reply(tool_use("open_app", {"name": "Бл"}), stop_reason="max_tokens")])
    with pytest.raises(LlmError, match="max_tokens"):
        client.complete([{"role": "user", "content": "открой"}], tools=[TOOL])


def test_betas_are_dropped_if_organization_rejects_them() -> None:
    rejected = status_error(
        anthropic.BadRequestError, 400, "Unexpected value(s) `x` for the `anthropic-beta` header."
    )
    client, fake = make([rejected, reply(text("Да.")), reply(text("Ещё."))])
    assert client.complete([{"role": "user", "content": "?"}]).text == "Да."
    client.complete([{"role": "user", "content": "?"}])
    first, second, third = fake.requests
    assert first["betas"] and "betas" not in second and "betas" not in third
    assert "thinking" not in second and "fallbacks" not in second


def test_fallbacks_only_for_models_that_support_them() -> None:
    client, fake = make([reply(text("Да."))], model="claude-haiku-5-5")
    client.complete([{"role": "user", "content": "?"}])
    assert fake.requests[0]["betas"] == [BETA_BINDING] and "fallbacks" not in fake.requests[0]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            status_error(anthropic.AuthenticationError, 401, "invalid x-api-key"),
            "api.anthropic.com не принял ключ API",
        ),
        (status_error(anthropic.NotFoundError, 404, "model: x"), "claude-opus-5-5 не найдена"),
        (status_error(anthropic.RateLimitError, 429, "slow down"), "лимит запросов"),
        (
            status_error(anthropic.BadRequestError, 400, "Your credit balance is too low to access the API"),
            "закончились деньги",
        ),
        (status_error(anthropic.InternalServerError, 529, "Overloaded"), "перегружен"),
        (
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")),
            "нет связи с api.anthropic.com",
        ),
    ],
)
def test_errors_are_explained(error: Exception, expected: str) -> None:
    client, _ = make([error])
    with pytest.raises(LlmError, match=expected):
        client.complete([{"role": "user", "content": "?"}])


def test_windows_variables_do_not_redirect_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    # Так их ставят, чтобы пустить Claude Code через другой сервис: ключ Claude там не подойдёт.
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://proxy.example/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "proxy-token-1234567890")
    settings = LlmConfig.model_validate({"profile": "anthropic"})
    client = AnthropicClient(settings, api_key="sk-ant-api03-typed-key-123456")
    sdk = client._client
    assert str(sdk.base_url).rstrip("/") == "https://api.anthropic.com"
    assert sdk.auth_headers == {"X-Api-Key": "sk-ant-api03-typed-key-123456"}


def test_cancel_works_while_claude_is_silent() -> None:
    silent = reply(text("Не успел."))
    silent.release = threading.Event()
    client, _ = make([silent])
    cancel = threading.Event()
    threading.Timer(0.2, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(LlmCancelled):
        client.complete([{"role": "user", "content": "думай"}], cancel=cancel)
    assert time.monotonic() - started < 1.0
    silent.release.set()
    assert wait_until(lambda: silent.closed)


def test_list_models_and_create_client() -> None:
    client, _ = make([])
    assert client.list_models() == ["claude-opus-5-5", "claude-haiku-5-5"]
    settings = LlmConfig.model_validate({"profile": "anthropic"})
    assert isinstance(create_client(settings), AnthropicClient)


def test_agent_tool_loop_with_claude(tmp_path: Path) -> None:
    """Агент на Claude: вызов инструмента, его результат и ответ; рассуждения возвращаются."""
    opened: list[str] = []
    registry = ActionRegistry()

    @registry.action("open_app", "Открыть программу")
    def open_app(name: str) -> str:
        opened.append(name)
        return f"Открываю {name}"

    fake = FakeAnthropic(
        [reply(thinking(), tool_use("open_app", {"name": "Блокнот"})), reply(text("Блокнот открыт."))]
    )
    settings = LlmConfig.model_validate({"profile": "anthropic"})
    executor = ActionExecutor(
        registry, SafetyPolicy(), StaticConfirmer(True), ActionJournal(tmp_path / "j.jsonl", SecretMasker())
    )
    speaker = RecordingSpeaker()
    claude = AnthropicClient(settings, factory_for(fake))
    agent = Agent(settings, executor, speaker, client_factory=lambda: claude)
    result = agent.run("открой блокнот")
    assert result.status is AgentStatus.DONE and opened == ["Блокнот"]
    assert speaker.said == ["Блокнот открыт."]
    second = fake.requests[1]["messages"]
    assert [block.type for block in second[1]["content"]] == ["thinking", "tool_use"]
    assert second[2]["content"][0]["content"] == "Открываю Блокнот"
    assert isinstance(fake.responses, list) and not fake.responses


def test_stream_is_closed_when_reading_stops() -> None:
    stream: FakeClaudeStream = reply(text("Раз. Два."), pieces=("Раз. ", "Два."))
    client, _ = make([stream])
    cancel = threading.Event()
    with pytest.raises(LlmCancelled):
        client.complete([{"role": "user", "content": "?"}], on_text=lambda piece: cancel.set(), cancel=cancel)
    assert wait_until(lambda: stream.closed)
