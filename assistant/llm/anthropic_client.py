"""Claude через официальный SDK anthropic — с тем же интерфейсом, что и LlmClient.

Агент хранит разговор в формате chat completions; здесь он переводится в формат
Messages API: системный промпт — отдельным полем, вызовы инструментов — блоками tool_use,
результаты — блоками tool_result в одном сообщении пользователя.

Особенности моделей Claude 5 (по документации Anthropic):
* рассуждения у Claude Opus 5.5 не выключаются, скорость задаёт output_config.effort
  (low — быстрее всего); temperature эти модели не принимают;
* блоки рассуждений (thinking) возвращаются без изменений внутри текущей задачи. Они
  привязаны к началу разговора, поэтому из прошлых обменов они убираются — удалять
  блоки разрешено только с начала. Если начало всё же разойдётся, API отбрасывает
  блоки сам (prefix_mismatch_behavior: drop_block), а не отвечает ошибкой 400;
* при отказе модели по соображениям безопасности (stop_reason: refusal) запрос
  повторяется на рекомендованной запасной модели (fallbacks: "default");
* eager_input_streaming у инструментов не включается: аргументы нужны агенту только
  целиком, а без него сервер сам проверяет, что они — правильный JSON.
Бета-функции включаются заголовками; если организация их не принимает (400 с упоминанием
anthropic-beta), запрос повторяется без них.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Generator, Sequence
from typing import Any

from assistant.config import LlmConfig, LlmProfile
from assistant.llm.client import (
    LlmCancelled,
    LlmError,
    LlmRefused,
    LlmReply,
    ToolCall,
    resolve_api_key,
    stream_in_background,
)

log = logging.getLogger(__name__)

BETA_BINDING = "thinking-binding-controls-2026-08-01"
BETA_FALLBACK = "server-side-fallback-2026-07-01"
# Модели, для которых Anthropic рекомендует серверную запасную модель при отказе.
FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})
DEFAULT_MAX_TOKENS = 16000
_THINKING_BLOCKS = ("thinking", "redacted_thinking")


def describe_error(exc: Exception, profile_name: str, profile: LlmProfile) -> str:
    """Понятное объяснение ошибки Anthropic API."""
    import anthropic

    detail = str(getattr(exc, "message", "") or exc)
    if isinstance(exc, anthropic.AuthenticationError):
        return f"ключ API не подошёл (профиль {profile_name}) — проверьте ключ в настройках"
    if isinstance(exc, anthropic.PermissionDeniedError):
        return f"доступ запрещён (403): {detail}"
    if isinstance(exc, anthropic.NotFoundError):
        return f"модель {profile.model} не найдена или недоступна вашей организации"
    if isinstance(exc, anthropic.RateLimitError):
        return "превышен лимит запросов (429), повторы не помогли — попробуйте позже"
    if isinstance(exc, anthropic.APITimeoutError):
        return "нейросеть не ответила вовремя"
    if isinstance(exc, anthropic.APIConnectionError):
        return f"нет связи с {profile.base_url or 'api.anthropic.com'}"
    if isinstance(exc, anthropic.APIStatusError):
        if exc.status_code == 402 or "credit balance" in detail.lower():
            return "на счёте Anthropic закончились деньги — пополните баланс в Claude Console"
        if exc.status_code >= 500:
            return f"сервис Claude перегружен или недоступен ({exc.status_code})"
        return f"запрос отклонён ({exc.status_code}): {detail}"
    return detail or type(exc).__name__


def to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Описание инструмента chat completions → Messages API."""
    function = tool["function"]
    return {
        "name": function["name"],
        "description": function.get("description", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }


def to_anthropic_messages(messages: Sequence[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Разговор chat completions → системный промпт и сообщения Messages API.

    Текущая задача начинается с последнего сообщения пользователя: ответы Claude в ней
    возвращаются как есть, с блоками рассуждений, а в прошлых обменах блоки рассуждений
    убираются (это удаление «с начала», оно не ломает привязку остальных блоков).
    """
    system_parts: list[str] = []
    result: list[dict[str, Any]] = []
    users = [i for i, m in enumerate(messages) if m["role"] == "user"]
    current = users[-1] if users else 0
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "system":
            system_parts.append(str(message["content"]))
        elif role == "user":
            result.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            content = _assistant_content(message, keep_thinking=index > current)
            if content:
                result.append({"role": "assistant", "content": content})
        elif role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": str(message.get("content") or ""),
            }
            if message.get("_is_error"):
                block["is_error"] = True
            # Все результаты одного хода — в одном сообщении пользователя.
            if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                result[-1]["content"].append(block)
            else:
                result.append({"role": "user", "content": [block]})
    return "\n\n".join(system_parts), result


def _assistant_content(message: dict[str, Any], keep_thinking: bool) -> list[Any]:
    """Блоки ответа ассистента: свои, как прислал Claude, или собранные из chat completions."""
    native = message.get("_native")
    if native is not None:
        return [block for block in native if keep_thinking or _block_type(block) not in _THINKING_BLOCKS]
    blocks: list[dict[str, Any]] = []
    if message.get("content"):
        blocks.append({"type": "text", "text": str(message["content"])})
    for call in message.get("tool_calls") or []:
        function = call["function"]
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}
        blocks.append({"type": "tool_use", "id": call["id"], "name": function["name"], "input": arguments})
    return blocks


def _block_type(block: Any) -> str:
    return str(block.get("type") if isinstance(block, dict) else getattr(block, "type", ""))


class AnthropicClient:
    """Запросы к Claude.

    Args:
        settings: раздел llm конфига (активный профиль с provider: anthropic).
        factory: конструктор клиента (по умолчанию anthropic.Anthropic) — подменяется в тестах.
    """

    def __init__(self, settings: LlmConfig, factory: Callable[..., Any] | None = None) -> None:
        self._settings = settings
        self.profile_name = settings.profile
        self.profile = settings.active
        if factory is None:
            from anthropic import Anthropic

            factory = Anthropic
        options: dict[str, Any] = {
            "api_key": resolve_api_key(self.profile_name, self.profile),
            "max_retries": settings.retries,
            "timeout": settings.request_timeout,
        }
        if self.profile.base_url:
            options["base_url"] = self.profile.base_url
        self._client = factory(**options)
        self._use_betas = True
        self._lock = threading.Lock()

    def describe(self) -> str:
        """Профиль и модель для сообщений («anthropic: claude-opus-5-5»)."""
        return f"{self.profile_name}: {self.profile.model}"

    def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] = (),
        on_text: Callable[[str], None] | None = None,
        cancel: threading.Event | None = None,
        timeout: float | None = None,
    ) -> LlmReply:
        """Отправляет разговор и читает ответ потоком (интерфейс как у LlmClient.complete).

        Raises:
            LlmError: Claude не ответил или отказал (LlmRefused — отказ по соображениям безопасности).
            LlmCancelled: пользователь прервал запрос.
        """
        import anthropic

        started = time.perf_counter()
        system, converted = to_anthropic_messages(messages)
        while True:
            request = self._request(system, converted, tools, timeout)
            try:
                reply = self._read(self._events(request, cancel), on_text, started)
            except (LlmCancelled, LlmError):
                raise
            except anthropic.BadRequestError as exc:
                if request.get("betas") and "anthropic-beta" in str(getattr(exc, "message", exc)):
                    log.warning("Claude не принял бета-функции (%s) — дальше без них", exc)
                    with self._lock:
                        self._use_betas = False
                    continue
                log.info("Нейросеть %s: %s", self.describe(), exc)
                raise LlmError(describe_error(exc, self.profile_name, self.profile)) from exc
            except anthropic.AnthropicError as exc:
                log.info("Нейросеть %s: %s", self.describe(), exc)
                raise LlmError(describe_error(exc, self.profile_name, self.profile)) from exc
            except ValueError as exc:  # SDK не смог разобрать поток ответа
                log.info("Нейросеть %s: испорченный ответ: %s", self.describe(), exc)
                raise LlmError("ответ Claude пришёл испорченным — повторите просьбу") from exc
            break
        log.info(
            "Нейросеть %s: первый токен %s мс, всего %.0f мс, вызовов инструментов %d",
            self.describe(),
            f"{reply.first_token_ms:.0f}" if reply.first_token_ms is not None else "—",
            reply.total_ms,
            len(reply.tool_calls),
        )
        return reply

    def list_models(self) -> list[str]:
        """Модели, доступные по ключу.

        Raises:
            LlmError: Claude не ответил или отказал.
        """
        import anthropic

        try:
            return [model.id for model in self._client.models.list()]
        except anthropic.AnthropicError as exc:
            raise LlmError(describe_error(exc, self.profile_name, self.profile)) from exc

    def _request(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        timeout: float | None,
    ) -> dict[str, Any]:
        profile = self.profile
        request: dict[str, Any] = {
            "model": profile.model,
            "max_tokens": profile.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [to_anthropic_tool(tool) for tool in tools]
        if profile.effort:
            request["output_config"] = {"effort": profile.effort}
        if profile.extra_body:
            request["extra_body"] = profile.extra_body
        if timeout is not None:
            request["timeout"] = timeout
        with self._lock:
            use_betas = self._use_betas
        if use_betas:
            betas = [BETA_BINDING]
            request["thinking"] = {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
            if profile.fallbacks and profile.model in FALLBACK_MODELS:
                betas.append(BETA_FALLBACK)
                request["fallbacks"] = "default"
            request["betas"] = betas
        return request

    def _events(self, request: dict[str, Any], cancel: threading.Event | None) -> Generator[Any, None, None]:
        """Кусочки текста и в конце окончательное сообщение; запрос — в отдельном потоке."""

        def produce(put: Callable[[Any], None], abandoned: threading.Event) -> None:
            with self._client.beta.messages.stream(**request) as stream:
                for event in stream:
                    if abandoned.is_set():
                        return
                    if event.type == "text":
                        put(("text", event.text))
                put(("final", stream.get_final_message()))

        return stream_in_background(produce, cancel)

    def _read(
        self, events: Generator[Any, None, None], on_text: Callable[[str], None] | None, started: float
    ) -> LlmReply:
        reply = LlmReply()
        final: Any = None
        try:
            for kind, value in events:
                if kind == "text":
                    if reply.first_token_ms is None:
                        reply.first_token_ms = (time.perf_counter() - started) * 1000
                    if on_text is not None:
                        on_text(value)
                else:
                    final = value
        finally:
            events.close()
        if final is None:
            raise LlmError("ответ Claude оборвался")
        if final.stop_reason == "refusal":
            details = getattr(final, "stop_details", None)
            category = getattr(details, "category", None) if details is not None else None
            raise LlmRefused(f"Claude отказался отвечать{f' (категория {category})' if category else ''}")
        calls = [block for block in final.content if block.type == "tool_use"]
        if calls and final.stop_reason == "max_tokens":
            raise LlmError(
                f"ответ оборвался: не хватило max_tokens — увеличьте его в профиле {self.profile_name}"
            )
        reply.text = "".join(block.text for block in final.content if block.type == "text")
        reply.tool_calls = [
            ToolCall(block.id, block.name, json.dumps(block.input, ensure_ascii=False)) for block in calls
        ]
        reply.native = list(final.content)
        reply.total_ms = (time.perf_counter() - started) * 1000
        if reply.first_token_ms is None:
            reply.first_token_ms = reply.total_ms
        return reply
