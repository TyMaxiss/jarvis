"""Браузер: открыть сайт, поиск в интернете и на YouTube."""

from __future__ import annotations

import re
import webbrowser
from typing import Annotated
from urllib.parse import quote_plus, urlsplit

from pydantic import Field

from assistant.actions.registry import ActionError, ActionRegistry
from assistant.config import BrowserConfig

_DOMAIN_RE = re.compile(r"^[\w.-]+\.[a-zа-я]{2,}(/.*)?$", re.IGNORECASE)


def normalize_url(url: str) -> str:
    """Проверяет адрес и добавляет https:// к голому домену («youtube.com»).

    Разрешены только http и https: file:// и другие схемы могут запустить программу.

    Raises:
        ActionError: адрес не похож на веб-страницу.
    """
    url = url.strip()
    if _DOMAIN_RE.match(url):
        url = "https://" + url
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ActionError(f"Это не адрес веб-страницы: {url}")
    return url


def open_in_browser(url: str) -> None:
    """Открывает адрес в браузере по умолчанию."""
    if not webbrowser.open(url):
        raise ActionError("Не удалось открыть браузер")


def register(registry: ActionRegistry, config: BrowserConfig) -> None:
    """Регистрирует действия браузера."""

    @registry.action("open_url", "Открыть сайт в браузере по умолчанию")
    def open_url(
        url: Annotated[str, Field(description="Адрес, например https://youtube.com или vk.com")],
    ) -> str:
        normalized = normalize_url(url)
        open_in_browser(normalized)
        return f"Открываю {urlsplit(normalized).netloc}"

    @registry.action("web_search", "Найти в интернете (открывает страницу поиска в браузере)")
    def web_search(
        query: Annotated[str, Field(min_length=1, description="Поисковый запрос")],
    ) -> str:
        open_in_browser(config.search_url.format(query=quote_plus(query)))
        return f"Ищу «{query}»"

    @registry.action("youtube_search", "Найти видео на YouTube (открывает результаты поиска)")
    def youtube_search(
        query: Annotated[str, Field(min_length=1, description="Что искать на YouTube")],
    ) -> str:
        open_in_browser(config.youtube_search_url.format(query=quote_plus(query)))
        return f"Ищу на YouTube «{query}»"
