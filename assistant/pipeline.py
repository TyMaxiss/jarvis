"""Конвейер обработки фразы: локальный матчер → выполнение команды → ответ.

Фразы, которые не нашлись локально, на этапе 3 будут уходить в LLM-агента.
Пока для них возвращается маршрут "llm" и ответ «Не знаю такой команды».
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from assistant.actions.executor import ActionExecutor, RunResult, StepStatus
from assistant.actions.registry import ArgsError, UnknownActionError
from assistant.actions.winapi import WindowInfo
from assistant.commands.loader import render_reply, substitute
from assistant.nlu.matcher import MatchedCommand, Matcher, MatchResult
from assistant.safety.confirm import Confirmer
from assistant.tts.base import Speaker

log = logging.getLogger(__name__)

Route = Literal["local", "llm"]


@dataclass
class CommandOutcome:
    """Результат выполнения одной найденной команды."""

    matched: MatchedCommand
    run: RunResult
    reply: str | None = None


@dataclass
class PipelineResult:
    """Что произошло с фразой."""

    text: str
    route: Route
    match: MatchResult
    outcomes: list[CommandOutcome] = field(default_factory=list)
    reply: str = ""
    match_ms: float = 0.0
    total_ms: float = 0.0


class Pipeline:
    """Обрабатывает фразы пользователя.

    Args:
        matcher: локальный матчер команд.
        executor: исполнитель действий.
        speaker: кто произносит ответы.
        window_provider: функция, возвращающая активное окно (для условий when).
    """

    def __init__(
        self,
        matcher: Matcher,
        executor: ActionExecutor,
        speaker: Speaker,
        window_provider: Callable[[], WindowInfo],
    ) -> None:
        self._matcher = matcher
        self._executor = executor
        self._speaker = speaker
        self._window_provider = window_provider
        # Голос и консоль могут прислать фразы одновременно — выполняем по одной.
        self._lock = threading.Lock()

    def handle_text(
        self, text: str, source: str = "local", confirmer: Confirmer | None = None
    ) -> PipelineResult:
        """Обрабатывает фразу: ищет команду, выполняет, произносит ответ.

        Args:
            text: фраза пользователя.
            source: откуда пришла: local (текст), voice (голос).
            confirmer: как подтверждать рискованные действия для этой фразы.
        """
        with self._lock:
            return self._handle(text, source, confirmer)

    def _handle(self, text: str, source: str, confirmer: Confirmer | None) -> PipelineResult:
        started = time.perf_counter()
        match = self._matcher.match(text, self._window_provider())
        match_ms = (time.perf_counter() - started) * 1000

        if not match.matched:
            # Этап 3: здесь фраза уйдёт в LLM-агента.
            log.info("«%s» не найдена локально (%s) — маршрут LLM", text, match.reason or match.status)
            reply = "" if match.status == "empty" else "Не знаю такой команды"
            if reply:
                self._speaker.say(reply)
            return PipelineResult(text, "llm", match, reply=reply, match_ms=match_ms,
                                  total_ms=(time.perf_counter() - started) * 1000)  # fmt: skip

        outcomes: list[CommandOutcome] = []
        for matched in match.matches:
            outcome = self._run_command(matched, text, source, confirmer)
            outcomes.append(outcome)
            if not outcome.run.ok:
                break
        reply = " ".join(o.reply for o in outcomes if o.reply)
        total_ms = (time.perf_counter() - started) * 1000
        log.info("Задержки: матчинг %.0f мс, текст → результат %.0f мс", match_ms, total_ms)
        if reply:
            self._speaker.say(reply)
        return PipelineResult(text, "local", match, outcomes, reply, match_ms, total_ms)

    def _run_command(
        self, matched: MatchedCommand, phrase: str, source: str, confirmer: Confirmer | None
    ) -> CommandOutcome:
        """Подставляет слоты в шаги команды и выполняет их."""
        command = matched.command
        try:
            planned = self._executor.plan(
                (step.action, substitute(step.args, matched.slots)) for step in command.steps
            )
        except (ArgsError, UnknownActionError) as exc:
            log.warning("Команда %s: %s", command.id, exc)
            return CommandOutcome(matched, RunResult([], StepStatus.FAILED), f"Не могу выполнить: {exc}")

        run = self._executor.run(
            planned,
            source=source,
            command_id=command.id,
            phrase=phrase,
            require_confirm=command.confirm,
            question=f"Выполнить «{matched.text}»?",
            confirmer=confirmer,
        )
        if run.status is StepStatus.CANCELLED:
            reply: str | None = "Отменено"
        elif not run.ok:
            reply = run.last_text or "Не получилось"
        elif command.reply:
            reply = render_reply(command.reply, matched.slots, run.last_text)
        else:
            reply = None
        return CommandOutcome(matched, run, reply)
