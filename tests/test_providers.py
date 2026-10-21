"""Провайдеры для настроек: профиль из выбора в интерфейсе, проверка связи, список моделей."""

from __future__ import annotations

from typing import Any

import openai
import pytest
from pydantic import ValidationError

from assistant.config import DEFAULT_LLM_PROFILES, LlmConfig
from assistant.llm import client as client_module
from assistant.llm.providers import (
    CUSTOM_KEY_ENV,
    CUSTOM_PROFILE,
    PROVIDERS,
    build_profile,
    check_connection,
    list_models,
    provider_info,
    with_profile,
)
from tests.llm_fakes import FakeOpenAI, factory_for, status_error, text_reply, tool_reply


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeOpenAI:
    """Подменяет openai.OpenAI в клиенте: запросы уходят в подделку."""
    fake = FakeOpenAI([])
    real = client_module.LlmClient

    def make_client(settings: LlmConfig, factory: Any = None, api_key: str | None = None) -> Any:
        return real(settings, factory=factory_for(fake), api_key=api_key)

    monkeypatch.setattr(client_module, "LlmClient", make_client)
    return fake


def test_every_default_profile_has_provider_info() -> None:
    for name in DEFAULT_LLM_PROFILES:
        assert name in PROVIDERS, name
    assert PROVIDERS["openai"].models[0] == "gpt-6-luna"
    # Модели, которым для инструментов нужен Responses API, в список не попадают.
    assert not {"gpt-6-astra", "gpt-6.1-sol"} & set(PROVIDERS["openai"].models)
    assert PROVIDERS["anthropic"].models[0] == DEFAULT_LLM_PROFILES["anthropic"]["model"]


def test_profile_from_config_gets_generic_info() -> None:
    groq = {"base_url": "https://x.example/v1", "model": "m"}
    settings = LlmConfig.model_validate({"profiles": {"groq": groq}})
    info = provider_info("groq", settings.profiles["groq"])
    assert info.base_url_editable and not info.models


def test_build_profile_changes_only_model_of_fixed_provider() -> None:
    settings = LlmConfig()
    profile = build_profile(settings, "deepseek", "deepseek-v4-pro", base_url="https://evil.example")
    assert profile.model == "deepseek-v4-pro"
    assert profile.base_url == "https://api.deepseek.com"  # адрес DeepSeek из интерфейса не меняется
    assert profile.extra_body == {"thinking": {"type": "disabled"}}


def test_build_profile_for_ollama_and_custom() -> None:
    settings = LlmConfig()
    ollama = build_profile(settings, "ollama", "qwen3:4b", base_url="http://192.168.1.5:11434/v1")
    assert ollama.base_url == "http://192.168.1.5:11434/v1" and ollama.model == "qwen3:4b"
    custom = build_profile(settings, CUSTOM_PROFILE, "local-model", base_url="http://localhost:1234/v1")
    assert custom.provider == "openai" and custom.api_key_env == ""
    url = "http://localhost:1234/v1"
    keyed = build_profile(settings, CUSTOM_PROFILE, "m", base_url=url, key_env=CUSTOM_KEY_ENV)
    assert keyed.api_key_env == CUSTOM_KEY_ENV
    with pytest.raises(ValidationError):
        build_profile(settings, CUSTOM_PROFILE, "m")  # без адреса нельзя
    with pytest.raises(KeyError):
        build_profile(settings, "nope", "m")


def test_with_profile_selects_profile_without_touching_original() -> None:
    settings = LlmConfig()
    changed = with_profile(settings, "openai", build_profile(settings, "openai", "gpt-5.4-mini"))
    assert changed.profile == "openai" and changed.active.model == "gpt-5.4-mini"
    assert settings.profile == "deepseek" and settings.profiles["openai"].model == "gpt-6-luna"


def test_check_connection_sends_tool_and_reports_time(fake: FakeOpenAI) -> None:
    fake.responses = [text_reply("Готово.")]
    settings = with_profile(LlmConfig(), "openai", build_profile(LlmConfig(), "openai", "gpt-6-luna"))
    result = check_connection(settings, api_key="sk-typed-key-1234567890")
    assert result.ok and result.text == "Готово." and result.ms >= 0
    assert fake.client_kwargs["api_key"] == "sk-typed-key-1234567890"
    request = fake.requests[0]
    assert request["tools"][0]["function"]["name"] == "ping"
    assert request["reasoning_effort"] == "none"


def test_check_connection_accepts_tool_call_as_answer(fake: FakeOpenAI) -> None:
    fake.responses = [tool_reply(("ping", {}))]
    result = check_connection(LlmConfig(), api_key="sk-typed-key-1234567890")
    assert result.ok and result.text == "вызвала инструмент"


def test_check_connection_explains_errors(fake: FakeOpenAI, monkeypatch: pytest.MonkeyPatch) -> None:
    fake.responses = [status_error(openai.AuthenticationError, 401, "Invalid key")]
    result = check_connection(LlmConfig(), api_key="sk-bad-key-1234567890")
    assert not result.ok and "ключ API не подошёл" in result.error
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    result = check_connection(LlmConfig())  # ключа нет ни в поле, ни в .env
    assert not result.ok and "DEEPSEEK_API_KEY" in result.error


def test_list_models_is_sorted(fake: FakeOpenAI) -> None:
    class Model:
        def __init__(self, id: str) -> None:
            self.id = id

    fake.models = type("Models", (), {"list": staticmethod(lambda: [Model("b"), Model("a")])})()
    assert list_models(LlmConfig(), api_key="sk-typed-key-1234567890") == ["a", "b"]
