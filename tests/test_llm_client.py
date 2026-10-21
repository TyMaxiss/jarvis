"""Тесты клиента нейросети на поддельном OpenAI-клиенте."""

from __future__ import annotations

import json
import threading

import openai
import pytest

from assistant.config import LlmConfig
from assistant.llm.client import LlmCancelled, LlmClient, LlmError
from tests.llm_fakes import (
    FakeOpenAI,
    chunk,
    connection_error,
    factory_for,
    status_error,
    text_reply,
    tool_reply,
)


@pytest.fixture(autouse=True)
def keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key-123456")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-test-key-123456")


def make(
    responses: list[object], profile: str = "deepseek", **overrides: object
) -> tuple[LlmClient, FakeOpenAI]:
    fake = FakeOpenAI(responses)
    settings = LlmConfig.model_validate({"profile": profile, **overrides})
    return LlmClient(settings, factory=factory_for(fake)), fake


def test_deepseek_profile_request() -> None:
    client, fake = make([text_reply("Готово.")])
    reply = client.complete([{"role": "user", "content": "привет"}], tools=[{"type": "function"}])
    assert reply.text == "Готово." and reply.tool_calls == []
    assert fake.client_kwargs["base_url"] == "https://api.deepseek.com"
    assert fake.client_kwargs["api_key"] == "sk-test-deepseek-key-123456"
    assert fake.client_kwargs["max_retries"] == 2  # ТЗ: два повтора при ошибках сети, таймаутах, 429
    request = fake.requests[0]
    assert request["model"] == "deepseek-flash" and request["stream"] is True
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}  # быстрый режим без рассуждений
    assert request["tools"] == [{"type": "function"}]
    assert fake.streams[0].closed


def test_ollama_needs_no_key_and_no_extra_body() -> None:
    client, fake = make([text_reply("Да.")], profile="ollama")
    client.complete([{"role": "user", "content": "привет"}])
    assert fake.client_kwargs["api_key"]  # SDK требует непустой ключ
    assert "extra_body" not in fake.requests[0] and "tools" not in fake.requests[0]


def test_missing_key_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(LlmError, match="DEEPSEEK_API_KEY"):
        make([], profile="deepseek")


def test_text_is_streamed_in_pieces() -> None:
    client, _ = make([text_reply("Сейчас ", "двенадцать ", "часов.")])
    pieces: list[str] = []
    reply = client.complete([{"role": "user", "content": "который час"}], on_text=pieces.append)
    assert pieces == ["Сейчас ", "двенадцать ", "часов."]
    assert reply.text == "Сейчас двенадцать часов."
    assert reply.first_token_ms is not None and reply.total_ms >= reply.first_token_ms


def test_tool_calls_are_assembled_from_pieces() -> None:
    client, _ = make([tool_reply(("open_app", {"name": "Telegram"}), ("volume_set", {"percent": 30}))])
    reply = client.complete([{"role": "user", "content": "рабочий режим"}])
    assert [(c.id, c.name, json.loads(c.arguments)) for c in reply.tool_calls] == [
        ("call_0", "open_app", {"name": "Telegram"}),
        ("call_1", "volume_set", {"percent": 30}),
    ]
    message = reply.assistant_message()
    assert message["tool_calls"][1]["function"]["name"] == "volume_set"


def test_reasoning_is_kept_for_next_request() -> None:
    # DeepSeek в режиме рассуждений требует вернуть reasoning_content в следующих запросах.
    client, _ = make([[chunk(reasoning_content="Нужно узнать время."), chunk("Сейчас полдень.")]])
    reply = client.complete([{"role": "user", "content": "который час"}])
    assert reply.assistant_message()["reasoning_content"] == "Нужно узнать время."


def test_cancel_stops_reading() -> None:
    cancel = threading.Event()
    client, fake = make([text_reply("раз ", "два ", "три")])

    def on_text(piece: str) -> None:
        cancel.set()

    with pytest.raises(LlmCancelled):
        client.complete([{"role": "user", "content": "считай"}], on_text=on_text, cancel=cancel)
    assert fake.streams[0].closed


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.AuthenticationError, 401, "Invalid key"), "ключ API не подошёл"),
        (
            status_error(openai.PermissionDeniedError, 403, "FreeTierError: only from within OpenCode"),
            "только в приложении OpenCode",
        ),
        (status_error(openai.RateLimitError, 429, "slow down"), "лимит запросов"),
        (status_error(openai.NotFoundError, 404, "no such model"), "модель deepseek-flash не найдена"),
        (status_error(openai.APIStatusError, 402, "Insufficient Balance"), "закончились деньги"),
        (status_error(openai.InternalServerError, 503, "busy"), "временно недоступен"),
        (connection_error(), "нет связи с https://api.deepseek.com"),
    ],
)
def test_errors_are_explained(error: Exception, expected: str) -> None:
    client, _ = make([error])
    with pytest.raises(LlmError, match=expected):
        client.complete([{"role": "user", "content": "привет"}])


def test_error_text_has_no_key() -> None:
    client, _ = make([status_error(openai.AuthenticationError, 401, "bad key sk-test-deepseek-key-123456")])
    with pytest.raises(LlmError) as info:
        client.complete([{"role": "user", "content": "привет"}])
    assert "sk-test" not in str(info.value)
