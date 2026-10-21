"""Тесты сборки в main.py: маскировка ключей и сообщения о нейросети при старте."""

from __future__ import annotations

import pytest

from assistant.config import AppConfig
from assistant.logging_setup import SecretMasker
from assistant.main import build_masker, describe_llm


def test_keys_of_all_profiles_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    # Переменная не похожа на ключ по имени (*_KEY, *_TOKEN), но указана в профиле.
    monkeypatch.setenv("MY_LLM_CREDENTIAL", "very-secret-value-2026")
    config = AppConfig.model_validate(
        {"llm": {"profiles": {"my": {"base_url": "https://llm.example/v1", "model": "m",
                                     "api_key_env": "MY_LLM_CREDENTIAL"}}}}
    )  # fmt: skip
    line = "ответ: very-secret-value-2026"
    assert SecretMasker.from_environ().mask(line) == line  # по имени переменной не узнать
    assert "very-secret" not in build_masker(config).mask(line)


def test_start_message_explains_missing_key_and_zen_free_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    text = describe_llm(AppConfig.model_validate({"llm": {"profile": "opencode_zen"}}))
    assert "профиль opencode_zen, модель deepseek-v4-flash-free" in text
    assert "укажите его в настройках (или OPENCODE_API_KEY в .env)" in text
    assert "403 FreeTierError" in text


def test_start_message_for_ready_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key-123456")
    text = describe_llm(AppConfig.model_validate({"llm": {"profile": "deepseek"}}))
    assert text == "Нейросеть: профиль deepseek, модель deepseek-flash.\n"
    disabled = describe_llm(AppConfig.model_validate({"llm": {"enabled": False}}))
    assert disabled.startswith("Нейросеть выключена")
