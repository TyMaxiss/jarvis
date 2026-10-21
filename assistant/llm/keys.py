"""Ключи API: как они выглядят у разных сервисов и почему сервис мог ключ не принять.

Принять ключ или нет, решает сам сервис. Здесь только подсказки, чтобы отказ было
легко понять: скопирована сокращённая запись ключа, вставлен ключ другого сервиса
или вовсе не ключ API.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from assistant.logging_setup import SecretMasker

ANTHROPIC_API = "https://api.anthropic.com"
# Невидимые символы, которые прилипают к ключу при копировании из браузера или мессенджера.
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"))
# Так ключ показывают в списке ключей: целиком он виден только при создании.
_MASKS = ("...", "…", "•", "***")
_DEEPSEEK_KEY = re.compile(r"sk-[0-9a-f]{32}")
_MAX_DETAIL = 160


@dataclass(frozen=True)
class Service:
    """Сервис, чьи ключи узнаются по началу."""

    title: str  # название в подсказке
    base_url: str  # адрес API
    example: str  # как начинается ключ
    tile: str = ""  # плитка в настройках ('' — подключается как «Другой сервис»)

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or ""


ANTHROPIC = Service("Claude (Anthropic)", ANTHROPIC_API, "sk-ant-api03-…", "Claude")
OPENAI = Service("OpenAI", "https://api.openai.com/v1", "sk-proj-…", "ChatGPT")
DEEPSEEK = Service("DeepSeek", "https://api.deepseek.com", "sk-…", "DeepSeek")
OPENROUTER = Service("OpenRouter", "https://openrouter.ai/api/v1", "sk-or-v1-…")
SERVICES = (ANTHROPIC, OPENAI, DEEPSEEK, OPENROUTER)

# Похожи на ключи API, но с ними к моделям не пустят.
_NOT_API_KEYS = (
    (
        "sk-ant-oat",
        "токен входа по подписке Claude (например, из Claude Code), а не ключ API: подписка "
        "Claude Pro или Max не даёт доступа к API",
    ),
    ("sk-ant-admin", "ключ администратора Anthropic (Admin API): с ним нельзя обращаться к моделям"),
    ("sk-admin-", "ключ администратора OpenAI: с ним нельзя обращаться к моделям"),
)


def clean_key(value: str) -> str:
    """Ключ без пробелов по краям и невидимых символов."""
    return value.translate(_INVISIBLE).strip()


def format_problem(key: str) -> str:
    """Что не так с записью ключа ('' — похоже на ключ)."""
    if any(mask in key for mask in _MASKS):
        return (
            "это сокращённая запись ключа (с «…» или точками). Целиком ключ показывается только "
            "один раз, при создании: создайте новый ключ и сразу скопируйте его"
        )
    for position, char in enumerate(key, 1):
        if char.isspace():
            return f"в ключе пробел или перенос строки ({position}-й символ)"
        if not "!" <= char <= "~":
            return f"в ключе символ «{char}» ({position}-й) — ключ пишется латиницей и цифрами"
    if len(key) < 8:
        return "ключ слишком короткий — скопируйте его целиком"
    return ""


def key_owner(key: str) -> Service | None:
    """Чей ключ — по его началу (None — не узнать)."""
    if key.startswith("sk-ant-api"):
        return ANTHROPIC
    if key.startswith(("sk-proj-", "sk-svcacct-")):
        return OPENAI
    if key.startswith("sk-or-"):
        return OPENROUTER
    if _DEEPSEEK_KEY.fullmatch(key):
        return DEEPSEEK
    return None


def service_at(base_url: str) -> Service | None:
    """Сервис по адресу API (None — свой или незнакомый)."""
    host = urlparse(base_url).hostname or ""
    return next((service for service in SERVICES if service.host == host), None)


def rejected_key_hint(key: str, base_url: str) -> str:
    """Почему сервис по адресу base_url мог не принять ключ ('' — подсказать нечего)."""
    for prefix, what in _NOT_API_KEYS:
        if key.startswith(prefix):
            return f"Это {what}."
    target = service_at(base_url)
    owner = key_owner(key)
    if owner is not None and owner != target:
        where = f"в плитку «{owner.tile}»" if owner.tile else f"в «Другой сервис» с адресом {owner.base_url}"
        return f"Похоже, это ключ {owner.title} — вставьте его {where}."
    if target is not None:
        return (
            f"Ключ {target.title} начинается так: {target.example} Проверьте, что он скопирован "
            "целиком и не удалён на сайте сервиса."
        )
    return ""


def server_message(exc: Exception, key: str = "") -> str:
    """Что ответил сервис (коротко и без ключа: сервисы иногда повторяют его в ответе)."""
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None
    message = error.get("message") if isinstance(error, dict) else None
    text = message if isinstance(message, str) else str(getattr(exc, "message", "") or exc)
    text = SecretMasker([key] if key else []).mask(" ".join(text.split()))
    return text if len(text) <= _MAX_DETAIL else text[: _MAX_DETAIL - 1] + "…"


def rejected_key_message(exc: Exception, key: str, base_url: str) -> str:
    """Ошибка «ключ не принят» с ответом сервиса и подсказкой, что проверить."""
    host = urlparse(base_url).hostname or base_url
    detail = server_message(exc, key)
    text = f"{host} не принял ключ API" + (f" ({detail})" if detail else "")
    hint = rejected_key_hint(key, base_url)
    return f"{text}. {hint}" if hint else f"{text} — проверьте ключ в настройках"
