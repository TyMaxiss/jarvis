"""Агент: нейросеть вызывает инструменты из реестра, пока не ответит текстом.

Один запуск — одна просьба пользователя: не больше max_steps обращений к нейросети
и не дольше timeout секунд. Инструменты выполняются через ActionExecutor, поэтому
действуют та же политика безопасности, подтверждения, dry-run и журнал (source="llm").
Запрещённое действие не выполняется, а модель получает отказ текстом.

Ответ озвучивается по мере получения, начиная с первого законченного предложения.
Разговор помнится: последние history обменов; после reset_after секунд тишины он забывается.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from assistant.actions.executor import ActionExecutor, PlannedStep, StepStatus
from assistant.actions.registry import ArgsError, UnknownActionError
from assistant.actions.winapi import WindowInfo
from assistant.config import LlmConfig
from assistant.llm.client import LlmCancelled, LlmClient, LlmError
from assistant.llm.prompts import build_system_prompt
from assistant.safety.confirm import Confirmer
from assistant.tts.base import Speaker

log = logging.getLogger(__name__)

UNAVAILABLE = "Нейросеть недоступна"
_TIMEOUT_REPLY = "Не успел выполнить задачу вовремя."
_STEP_LIMIT_REPLY = "Задача оказалась слишком длинной, я остановился."
_CANCELLED_TOOL = "Отменено пользователем"
# Ответ модели и так озвучивается: отдельный инструмент «сказать» ей не нужен.
_HIDDEN_TOOLS = ("say",)
_SENTENCE_END = re.compile(r"[.!?…]+[»\")]*\s+")


class AgentStatus(StrEnum):
    """Чем закончилась задача."""

    DONE = "done"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    STEP_LIMIT = "step_limit"
    ERROR = "error"


@dataclass(frozen=True)
class AgentStep:
    """Вызов инструмента и его результат."""

    number: int
    action: str
    args: dict[str, Any]
    status: str
    text: str

    def describe(self) -> str:
        """Запись вида open_app(name='Telegram') → Открываю Telegram."""
        values = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.action}({values}) → {self.text}"


@dataclass
class AgentResult:
    """Что сделал агент."""

    text: str = ""
    status: AgentStatus = AgentStatus.DONE
    steps: list[AgentStep] = field(default_factory=list)
    error: str = ""  # почему нейросеть недоступна
    requests: int = 0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def ok(self) -> bool:
        """Нейросеть ответила."""
        return self.status is AgentStatus.DONE


class SentenceStreamer:
    """Собирает кусочки текста и отдаёт законченные предложения.

    Args:
        emit: что сделать с готовым предложением (произнести).
    """

    def __init__(self, emit: Callable[[str], None]) -> None:
        self._emit = emit
        self._buffer = ""

    def feed(self, piece: str) -> None:
        """Добавляет кусочек; каждое законченное предложение сразу отдаётся."""
        self._buffer += piece
        while match := _SENTENCE_END.search(self._buffer):
            sentence, self._buffer = self._buffer[: match.end()].strip(), self._buffer[match.end() :]
            if sentence:
                self._emit(sentence)

    def flush(self) -> None:
        """Отдаёт остаток (последнее предложение без пробела после точки)."""
        rest, self._buffer = self._buffer.strip(), ""
        if rest:
            self._emit(rest)


def confirm_question(step: PlannedStep) -> str:
    """Вопрос перед рискованным действием: что и с чем будет сделано."""
    shown = "; ".join(str(v) for v in step.args_dict().values() if v not in (None, "", False))
    return f"{step.action.description}{f' ({shown})' if shown else ''}?"


class Agent:
    """Выполняет просьбы, которых нет среди локальных команд.

    Args:
        settings: раздел llm конфига.
        executor: исполнитель действий (политика, подтверждение, журнал, dry-run).
        speaker: кто произносит ответ.
        name: имя ассистента (для промпта).
        client_factory: создаёт клиента нейросети; по умолчанию LlmClient(settings).
        window_provider: активное окно (для промпта).
        clock: монотонные часы (подменяются в тестах).
        now: текущие дата и время (для промпта).
    """

    def __init__(
        self,
        settings: LlmConfig,
        executor: ActionExecutor,
        speaker: Speaker,
        *,
        name: str = "джарвис",
        client_factory: Callable[[], LlmClient] | None = None,
        window_provider: Callable[[], WindowInfo] | None = None,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._settings = settings
        self._executor = executor
        self._speaker = speaker
        self._name = name
        self._client_factory = client_factory or (lambda: LlmClient(settings))
        self._client: LlmClient | None = None
        self._window_provider = window_provider
        self._clock = clock
        self._now = now
        self._history: list[list[dict[str, Any]]] = []
        self._last_activity: float | None = None

    def describe(self) -> str:
        """Профиль и модель («deepseek: deepseek-flash»)."""
        return f"{self._settings.profile}: {self._settings.active.model}"

    def forget(self) -> None:
        """Забывает разговор."""
        self._history.clear()

    def run(
        self,
        text: str,
        *,
        confirmer: Confirmer | None = None,
        cancel: threading.Event | None = None,
        on_step: Callable[[AgentStep], None] | None = None,
    ) -> AgentResult:
        """Выполняет просьбу пользователя.

        Args:
            text: фраза пользователя.
            confirmer: как подтверждать рискованные действия.
            cancel: если установлено — задача прерывается.
            on_step: вызывается после каждого выполненного инструмента.
        """
        cancel = cancel or threading.Event()
        started = self._clock()
        deadline = started + self._settings.timeout
        if self._last_activity is not None and started - self._last_activity > self._settings.reset_after:
            self.forget()
        result = AgentResult()

        def speak(sentence: str) -> None:
            if not cancel.is_set():  # после «стоп» новых фраз не начинаем
                self._say(result, sentence)

        streamer = SentenceStreamer(speak)
        exchange: list[dict[str, Any]] = [{"role": "user", "content": text}]
        try:
            client = self._get_client()
        except LlmError as exc:
            return self._unavailable(result, str(exc), started)
        tools = self._executor.registry.tools(exclude=_HIDDEN_TOOLS)

        for _ in range(self._settings.max_steps):
            remaining = deadline - self._clock()
            if cancel.is_set():
                result.status = AgentStatus.CANCELLED
                break
            if remaining <= 0:
                self._finish(result, AgentStatus.TIMEOUT, _TIMEOUT_REPLY, exchange)
                break
            messages = [{"role": "system", "content": self._system_prompt()}, *self._past(), *exchange]
            try:
                reply = client.complete(
                    messages,
                    tools,
                    on_text=streamer.feed,
                    cancel=cancel,
                    timeout=min(self._settings.request_timeout, remaining),
                )
            except LlmCancelled:
                result.status = AgentStatus.CANCELLED
                break
            except LlmError as exc:
                if result.requests == 0:
                    return self._unavailable(result, str(exc), started)
                result.error = str(exc)
                self._finish(result, AgentStatus.ERROR, UNAVAILABLE, exchange)
                break
            result.requests += 1
            result.llm_ms += reply.total_ms
            streamer.flush()
            exchange.append(reply.assistant_message())
            if not reply.tool_calls:
                result.status = AgentStatus.DONE
                break
            for call in reply.tool_calls:
                if cancel.is_set():
                    exchange.append({"role": "tool", "tool_call_id": call.id, "content": _CANCELLED_TOOL})
                    continue
                step = self._run_tool(len(result.steps) + 1, call.name, call.arguments, text, confirmer)
                result.steps.append(step)
                if on_step is not None:
                    on_step(step)
                exchange.append({"role": "tool", "tool_call_id": call.id, "content": step.text})
        else:
            self._finish(result, AgentStatus.STEP_LIMIT, _STEP_LIMIT_REPLY, exchange)

        self._remember(exchange)
        result.total_ms = (self._clock() - started) * 1000
        log.info(
            "Агент: %s, запросов %d (нейросеть %.0f мс), инструментов %d, всего %.0f мс",
            result.status, result.requests, result.llm_ms, len(result.steps), result.total_ms,
        )  # fmt: skip
        return result

    # --- Внутреннее ------------------------------------------------------------------

    def _get_client(self) -> LlmClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _system_prompt(self) -> str:
        window = None
        if self._window_provider is not None:
            try:
                window = self._window_provider()
            except Exception:
                log.debug("Не удалось узнать активное окно для промпта", exc_info=True)
        return build_system_prompt(self._name, self._now(), window)

    def _past(self) -> list[dict[str, Any]]:
        return [message for exchange in self._history for message in exchange]

    def _remember(self, exchange: list[dict[str, Any]]) -> None:
        """Сохраняет обмен, если нейросеть успела ответить."""
        if len(exchange) > 1 and self._settings.history > 0:
            self._history.append(exchange)
            del self._history[: -self._settings.history]
        self._last_activity = self._clock()

    def _say(self, result: AgentResult, sentence: str) -> None:
        result.text = f"{result.text} {sentence}".strip()
        self._speaker.say(sentence)

    def _finish(
        self, result: AgentResult, status: AgentStatus, reply: str, exchange: list[dict[str, Any]]
    ) -> None:
        """Задача не завершена моделью: говорим почему и закрываем обмен ответом ассистента."""
        result.status = status
        self._say(result, reply)
        exchange.append({"role": "assistant", "content": reply})

    def _unavailable(self, result: AgentResult, reason: str, started: float) -> AgentResult:
        log.info("Нейросеть недоступна: %s", reason)
        result.status = AgentStatus.ERROR
        result.error = reason
        self._say(result, UNAVAILABLE)
        result.total_ms = (self._clock() - started) * 1000
        return result

    def _run_tool(
        self, number: int, name: str, raw_args: str, phrase: str, confirmer: Confirmer | None
    ) -> AgentStep:
        """Выполняет вызов инструмента; любой исход становится текстом для модели."""
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return AgentStep(number, name, {}, "error", "Ошибка: аргументы не в формате JSON")
        if not isinstance(args, dict):
            return AgentStep(number, name, {}, "error", "Ошибка: аргументы должны быть объектом JSON")
        try:
            planned = self._executor.plan([(name, args)])
        except UnknownActionError:
            return AgentStep(number, name, args, "error", f"Ошибка: инструмента {name} нет")
        except ArgsError as exc:
            return AgentStep(number, name, args, "error", f"Ошибка в аргументах: {exc}")
        run = self._executor.run(
            planned, source="llm", phrase=phrase, question=confirm_question(planned[0]), confirmer=confirmer
        )
        if run.status is StepStatus.CANCELLED:
            text = "Пользователь не подтвердил действие, оно не выполнено"
        elif run.status is StepStatus.FORBIDDEN:
            text = f"Отказано политикой безопасности. {run.last_text}"
        else:
            text = run.last_text or "Готово"
        return AgentStep(number, name, planned[0].args_dict(), run.status.value, text)
