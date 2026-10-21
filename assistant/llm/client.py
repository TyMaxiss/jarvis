"""Клиенты нейросети: общий интерфейс и OpenAI-совместимый клиент (официальный SDK openai).

Профиль с provider: openai — любой OpenAI-совместимый API (DeepSeek, ChatGPT, Ollama, Zen),
provider: anthropic — Claude через официальный SDK anthropic (llm/anthropic_client.py).
Клиента по профилю выбирает create_client.

Ответ читается потоком: текст отдаётся по кусочкам (озвучка начинается с первого
предложения), вызовы инструментов собираются из кусочков в конце. Повторы при
ошибках сети, таймаутах, 429 и 5xx делает сам SDK (max_retries, с нарастающей паузой).
Ошибки превращаются в LlmError с понятным текстом; ключи в текст не попадают.

Запрос идёт в отдельном потоке, поэтому отмена срабатывает сразу, даже пока нейросеть
ещё молчит: иначе её заметили бы только с приходом следующего кусочка.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

from assistant.config import LlmConfig, LlmProfile
from assistant.llm.keys import rejected_key_message

log = logging.getLogger(__name__)

# OpenAI SDK требует непустой ключ даже там, где он не нужен (Ollama).
_NO_KEY = "not-needed"
# Как часто проверять отмену, пока нейросеть молчит, секунды.
_CANCEL_POLL = 0.1
_END = object()  # поток ответа закончился
_NOTHING = object()  # за время ожидания ничего не пришло


class LlmError(Exception):
    """Нейросеть не ответила. Текст объясняет причину пользователю (без секретов)."""

    spoken = "Нейросеть недоступна"  # что сказать вслух


class LlmRefused(LlmError):
    """Модель отказалась отвечать на запрос (фильтр безопасности провайдера)."""

    spoken = "С этим я помочь не могу"


class LlmCancelled(Exception):
    """Запрос прерван пользователем."""


@dataclass(frozen=True)
class ToolCall:
    """Вызов инструмента, который запросила модель."""

    id: str
    name: str
    arguments: str  # JSON-строка, как её прислала модель


@dataclass
class LlmReply:
    """Ответ модели на один запрос."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning: str = ""  # reasoning_content: DeepSeek в режиме рассуждений требует вернуть его
    native: Any = None  # ответ в формате провайдера (Claude: блоки, которые возвращаются как есть)
    first_token_ms: float | None = None
    total_ms: float = 0.0

    def assistant_message(self) -> dict[str, Any]:
        """Сообщение ассистента для продолжения разговора.

        Поля с «_» служебные: их видит только клиент нейросети, провайдеру они не уходят.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        if self.native is not None:
            message["_native"] = self.native
        return message


class ChatClient(Protocol):
    """Клиент выбранного профиля нейросети."""

    profile_name: str
    profile: LlmProfile

    def describe(self) -> str: ...

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
    ) -> LlmReply: ...

    def list_models(self) -> list[str]: ...


def create_client(settings: LlmConfig, api_key: str | None = None) -> ChatClient:
    """Клиент для выбранного профиля.

    Args:
        settings: раздел llm конфига.
        api_key: ключ вместо сохранённого в .env (проверка ещё не сохранённого ключа).

    Raises:
        LlmError: ключ нужен, но не задан.
    """
    if settings.active.provider == "anthropic":
        from assistant.llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings, api_key=api_key)
    return LlmClient(settings, api_key=api_key)


def describe_error(exc: Exception, profile: LlmProfile, key: str = "") -> str:
    """Понятное объяснение ошибки OpenAI-совместимого провайдера (key — для подсказки о ключе)."""
    import openai

    detail = str(getattr(exc, "message", "") or exc)
    if isinstance(exc, openai.AuthenticationError):
        return rejected_key_message(exc, key, profile.base_url)
    if isinstance(exc, openai.PermissionDeniedError):
        if "FreeTierError" in detail or "free tier" in detail.lower():
            return (
                "бесплатные модели OpenCode Zen работают только в приложении OpenCode (403 FreeTierError). "
                "Выберите в настройках другого провайдера или платную модель Zen"
            )
        return f"доступ запрещён (403): {detail}"
    if isinstance(exc, openai.RateLimitError):
        return "превышен лимит запросов (429), повторы не помогли — попробуйте позже"
    if isinstance(exc, openai.NotFoundError):
        return f"модель {profile.model} не найдена у провайдера {profile.base_url}"
    if isinstance(exc, openai.APITimeoutError):
        return "нейросеть не ответила вовремя"
    if isinstance(exc, openai.APIConnectionError):
        hint = " — запущена ли Ollama?" if "11434" in profile.base_url else ""
        return f"нет связи с {profile.base_url}{hint}"
    if isinstance(exc, openai.BadRequestError) and "/v1/responses" in detail:
        return (
            f"модель {profile.model} вызывает действия только через Responses API — "
            "выберите в настройках другую модель, например gpt-6-luna"
        )
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 402:
            return "на счёте провайдера закончились деньги (402)"
        if exc.status_code >= 500:
            return f"сервис нейросети временно недоступен ({exc.status_code})"
        return f"запрос отклонён ({exc.status_code}): {detail}"
    return detail or type(exc).__name__


def resolve_api_key(profile_name: str, profile: LlmProfile) -> str:
    """Ключ из переменной окружения (.env загружается вместе с конфигом).

    Raises:
        LlmError: ключ нужен, но не задан.
    """
    if not profile.api_key_env:
        return _NO_KEY
    key = os.environ.get(profile.api_key_env, "").strip()
    if not key:
        raise LlmError(
            f"нет ключа API для профиля {profile_name}: укажите его в настройках "
            f"(или {profile.api_key_env} в .env)"
        )
    return key


def public_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Сообщения без служебных полей (ключи с «_»): их видит только наш код."""
    return [{key: value for key, value in m.items() if not key.startswith("_")} for m in messages]


@dataclass(frozen=True)
class _Failure:
    error: Exception


def stream_in_background(
    produce: Callable[[Callable[[Any], None], threading.Event], None],
    cancel: threading.Event | None,
) -> Generator[Any, None, None]:
    """Выполняет produce(put, abandoned) в отдельном потоке и отдаёт всё, что он положил.

    Пока провайдер молчит, отмена проверяется каждые 0,1 с. Ошибка внутри produce
    поднимается здесь. Когда чтение прекращено (отмена, ошибка у вызывающего),
    выставляется abandoned: produce должен перестать читать и закрыть соединение —
    брошенный запрос закроется, когда придёт его следующий кусочек или истечёт таймаут.

    Raises:
        LlmCancelled: пользователь прервал запрос.
    """
    items: queue.Queue[Any] = queue.Queue()
    abandoned = threading.Event()

    def run() -> None:
        try:
            produce(items.put, abandoned)
        except Exception as exc:  # ошибку разбирает вызывающий поток
            items.put(_Failure(exc))
        else:
            items.put(_END)

    threading.Thread(target=run, name="llm-request", daemon=True).start()
    try:
        while True:
            try:
                item = items.get(timeout=_CANCEL_POLL)
            except queue.Empty:
                item = _NOTHING
            if cancel is not None and cancel.is_set():
                raise LlmCancelled
            if item is _NOTHING:
                continue
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item
    finally:
        abandoned.set()


def uses_completion_tokens(profile: LlmProfile) -> bool:
    """Предел длины ответа у OpenAI — max_completion_tokens: max_tokens там устарел и не
    принимается моделями с рассуждениями; остальные OpenAI-совместимые API знают max_tokens."""
    return urlparse(profile.base_url).hostname == "api.openai.com"


class LlmClient:
    """Запросы к OpenAI-совместимому профилю (DeepSeek, ChatGPT, Ollama, Zen).

    Args:
        settings: раздел llm конфига.
        factory: конструктор клиента (по умолчанию openai.OpenAI) — подменяется в тестах.
        api_key: ключ вместо сохранённого в .env.
    """

    def __init__(
        self, settings: LlmConfig, factory: Callable[..., Any] | None = None, api_key: str | None = None
    ) -> None:
        self._settings = settings
        self.profile_name = settings.profile
        self.profile = settings.active
        if factory is None:
            from openai import OpenAI

            factory = OpenAI
        self._key = api_key or resolve_api_key(self.profile_name, self.profile)
        self._client = factory(
            base_url=self.profile.base_url,
            api_key=self._key,
            max_retries=settings.retries,
            timeout=settings.request_timeout,
        )

    def describe(self) -> str:
        """Профиль и модель для сообщений («deepseek: deepseek-flash»)."""
        return f"{self.profile_name}: {self.profile.model}"

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
    ) -> LlmReply:
        """Отправляет разговор и читает ответ потоком.

        Args:
            messages: сообщения в формате chat completions.
            tools: описания инструментов.
            on_text: вызывается с каждым кусочком текста ответа.
            cancel: если установлено — чтение прерывается.
            timeout: лимит на этот запрос, секунды.

        Raises:
            LlmError: провайдер не ответил или отказал.
            LlmCancelled: пользователь прервал запрос.
        """
        import openai

        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.profile.model,
            "messages": public_messages(messages),
            "stream": True,
        }
        if self.profile.temperature is not None:
            request["temperature"] = self.profile.temperature
        if self.profile.effort is not None:
            request["reasoning_effort"] = self.profile.effort
        if self.profile.max_tokens is not None:
            limit = "max_completion_tokens" if uses_completion_tokens(self.profile) else "max_tokens"
            request[limit] = self.profile.max_tokens
        if tools:
            request["tools"] = list(tools)
        if self.profile.extra_body:
            request["extra_body"] = self.profile.extra_body
        if timeout is not None:
            request["timeout"] = timeout
        try:
            reply = self._read(self._chunks(request, cancel), on_text, started)
        except (LlmCancelled, LlmError):
            raise
        except openai.OpenAIError as exc:
            log.info("Нейросеть %s: %s", self.describe(), exc)
            raise LlmError(describe_error(exc, self.profile, self._key)) from exc
        log.info(
            "Нейросеть %s: первый токен %s мс, всего %.0f мс, вызовов инструментов %d",
            self.describe(),
            f"{reply.first_token_ms:.0f}" if reply.first_token_ms is not None else "—",
            reply.total_ms,
            len(reply.tool_calls),
        )
        return reply

    def list_models(self) -> list[str]:
        """Модели, доступные по ключу (GET /models).

        Raises:
            LlmError: провайдер не ответил или отказал.
        """
        import openai

        try:
            return sorted(model.id for model in self._client.models.list())
        except openai.OpenAIError as exc:
            raise LlmError(describe_error(exc, self.profile, self._key)) from exc

    def _chunks(self, request: dict[str, Any], cancel: threading.Event | None) -> Generator[Any, None, None]:
        """Кусочки ответа: запрос выполняется в отдельном потоке."""

        def produce(put: Callable[[Any], None], abandoned: threading.Event) -> None:
            stream = self._client.chat.completions.create(**request)
            try:
                for chunk in stream:
                    if abandoned.is_set():
                        break
                    put(chunk)
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()

        return stream_in_background(produce, cancel)

    @staticmethod
    def _read(
        chunks: Generator[Any, None, None], on_text: Callable[[str], None] | None, started: float
    ) -> LlmReply:
        """Собирает ответ из кусочков потока."""
        reply = LlmReply()
        text: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        try:
            for chunk in chunks:
                if not chunk.choices:  # служебный кусок (например, usage)
                    continue
                delta = chunk.choices[0].delta
                extra = delta.model_extra or {}
                if extra.get("reasoning_content"):
                    reasoning.append(str(extra["reasoning_content"]))
                if delta.content:
                    if reply.first_token_ms is None:
                        reply.first_token_ms = (time.perf_counter() - started) * 1000
                    text.append(delta.content)
                    if on_text is not None:
                        on_text(delta.content)
                for part in delta.tool_calls or []:
                    if reply.first_token_ms is None:
                        reply.first_token_ms = (time.perf_counter() - started) * 1000
                    slot = calls.setdefault(part.index, {"id": "", "name": "", "arguments": ""})
                    if part.id:
                        slot["id"] = part.id
                    if part.function is not None:
                        slot["name"] += part.function.name or ""
                        slot["arguments"] += part.function.arguments or ""
        finally:
            chunks.close()  # если ответ не дочитан, запрос будет брошен
        reply.text = "".join(text)
        reply.reasoning = "".join(reasoning)
        reply.tool_calls = [
            ToolCall(slot["id"] or f"call_{index}", slot["name"], slot["arguments"] or "{}")
            for index, slot in sorted(calls.items())
            if slot["name"]
        ]
        reply.total_ms = (time.perf_counter() - started) * 1000
        return reply
