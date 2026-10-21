"""Ключи API: запись ключа и подсказки, почему сервис его не принял."""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from assistant.llm.keys import (
    ANTHROPIC_API,
    clean_key,
    format_problem,
    key_owner,
    rejected_key_hint,
    rejected_key_message,
    server_message,
)

CLAUDE_KEY = "sk-ant-api03-" + "A1b2_C3d4-" * 9 + "QAA"
OPENAI_KEY = "sk-proj-" + "x7Yz" * 30
DEEPSEEK_KEY = "sk-" + "0123456789abcdef" * 2


def test_invisible_characters_and_edges_are_removed() -> None:
    assert clean_key(f"﻿  {CLAUDE_KEY}​\n") == CLAUDE_KEY


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("sk-ant-api03-R2D...7QAA", "сокращённая запись"),
        ("sk-ant-api03-R2D…7QAA", "сокращённая запись"),
        ("sk-ant-api03-••••7QAA", "сокращённая запись"),
        ("sk-ant-api03 R2D2", "пробел или перенос строки (13-й символ)"),
        ("sk-ant-аpi03-R2D2", "символ «а» (8-й)"),  # русская «а»
        ("sk-ant", "слишком короткий"),
    ],
)
def test_format_problems_are_named(key: str, expected: str) -> None:
    assert expected in format_problem(key)


def test_real_looking_keys_have_no_format_problem() -> None:
    assert [format_problem(key) for key in (CLAUDE_KEY, OPENAI_KEY, DEEPSEEK_KEY)] == ["", "", ""]


def test_key_owner_by_prefix() -> None:
    assert key_owner(CLAUDE_KEY).tile == "Claude"  # type: ignore[union-attr]
    assert key_owner(OPENAI_KEY).tile == "ChatGPT"  # type: ignore[union-attr]
    assert key_owner(DEEPSEEK_KEY).tile == "DeepSeek"  # type: ignore[union-attr]
    assert key_owner("sk-or-v1-abcdef0123456789").title == "OpenRouter"  # type: ignore[union-attr]
    assert key_owner("my-local-token-123") is None


@pytest.mark.parametrize(
    ("key", "base_url", "expected"),
    [
        ("sk-ant-oat01-abcdefgh", ANTHROPIC_API, "токен входа по подписке Claude"),
        ("sk-ant-admin01-abcdefgh", ANTHROPIC_API, "ключ администратора Anthropic"),
        (OPENAI_KEY, ANTHROPIC_API, "ключ OpenAI — вставьте его в плитку «ChatGPT»"),
        (CLAUDE_KEY, "https://api.deepseek.com", "ключ Claude (Anthropic) — вставьте его в плитку «Claude»"),
        (
            "sk-or-v1-abcdef0123456789",
            ANTHROPIC_API,
            "в «Другой сервис» с адресом https://openrouter.ai/api/v1",
        ),
        (CLAUDE_KEY, ANTHROPIC_API, "Ключ Claude (Anthropic) начинается так: sk-ant-api03-…"),
        ("some-other-key-123", "https://api.openai.com/v1", "Ключ OpenAI начинается так: sk-proj-…"),
    ],
)
def test_rejected_key_hints(key: str, base_url: str, expected: str) -> None:
    assert expected in rejected_key_hint(key, base_url)


def test_no_hint_for_unknown_service() -> None:
    assert rejected_key_hint("token-1234567890", "http://localhost:1234/v1") == ""


def anthropic_401(message: str) -> anthropic.AuthenticationError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": "authentication_error", "message": message}}
    return anthropic.AuthenticationError(
        f"Error code: 401 - {body}", response=httpx2.Response(401, request=request), body=body
    )


def test_message_has_server_answer_and_hint_but_not_the_key() -> None:
    error = anthropic_401(f"invalid x-api-key {OPENAI_KEY}")
    assert OPENAI_KEY not in server_message(error, OPENAI_KEY)
    text = rejected_key_message(anthropic_401("invalid x-api-key"), OPENAI_KEY, ANTHROPIC_API)
    assert text.startswith("api.anthropic.com не принял ключ API (invalid x-api-key). ")
    assert "ключ OpenAI" in text and OPENAI_KEY not in text


def test_message_without_hint_points_to_settings() -> None:
    text = rejected_key_message(anthropic_401("unauthorized"), "token-1234567890", "http://localhost:1234/v1")
    assert text == "localhost не принял ключ API (unauthorized) — проверьте ключ в настройках"
