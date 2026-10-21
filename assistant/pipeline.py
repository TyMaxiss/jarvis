"""Конвейер обработки фразы: локальный матчер → выполнение команды → ответ.

Фразы, которые не нашлись локально (или нашлись неоднозначно), уходят агенту
нейросети. Если нейросеть выключена в конфиге, ответ — «Не знаю такой команды».
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from assistant.actions.executor import ActionExecutor, RunResult, StepResult, StepStatus
from assistant.actions.registry import ArgsError, UnknownActionError
from assistant.actions.winapi import WindowInfo
from assistant.commands.loader import Command, render_reply, substitute
from assistant.llm.agent import Agent, AgentResult, AgentStatus, AgentStep
from assistant.nlu.matcher import MatchedCommand, Matcher, MatchResult
from assistant.safety.confirm import Confirmer
from assistant.tts.base import Speaker

log = logging.getLogger(__name__)

Route = Literal["local", "llm"]

# Почему задача нейросети не выполнена — для Истории.
_AGENT_PROBLEMS = {
    AgentStatus.CANCELLED: "прервано",
    AgentStatus.TIMEOUT: "нейросеть не уложилась во время",
    AgentStatus.STEP_LIMIT: "слишком много шагов",
}


def new_request() -> str:
    """Id фразы в журнале: им связаны фраза и её действия."""
    return uuid.uuid4().hex[:12]


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
    agent: AgentResult | None = None
    request: str = ""  # id фразы в журнале

    @property
    def ok(self) -> bool:
        """Всё получилось: команды выполнены или нейросеть справилась."""
        if self.route == "local":
            return bool(self.outcomes) and all(outcome.run.ok for outcome in self.outcomes)
        return self.agent is not None and self.agent.ok

    @property
    def problem(self) -> str:
        """Что пошло не так ('' — всё в порядке)."""
        if self.ok:
            return ""
        if self.route == "local":
            return next((o.reply or "не получилось" for o in self.outcomes if not o.run.ok), "")
        if self.agent is None:
            return "нейросеть выключена"
        return self.agent.error or _AGENT_PROBLEMS.get(self.agent.status, self.agent.status.value)

    @property
    def handled(self) -> bool:
        """Фраза обработана: найдена команда или ответила нейросеть."""
        return self.route == "local" or (self.agent is not None and self.agent.ok)


class Pipeline:
    """Обрабатывает фразы пользователя.

    Args:
        matcher: локальный матчер команд.
        executor: исполнитель действий.
        speaker: кто произносит ответы.
        window_provider: функция, возвращающая активное окно (для условий when).
        agent: агент нейросети для фраз, которых нет среди команд; None — нейросеть выключена.
    """

    def __init__(
        self,
        matcher: Matcher,
        executor: ActionExecutor,
        speaker: Speaker,
        window_provider: Callable[[], WindowInfo],
        agent: Agent | None = None,
    ) -> None:
        self._matcher = matcher
        self._executor = executor
        self._speaker = speaker
        self._window_provider = window_provider
        self._agent = agent
        self._cancel = threading.Event()
        # Голос и консоль могут прислать фразы одновременно — выполняем по одной.
        self._lock = threading.Lock()
        self.on_agent_step: Callable[[AgentStep], None] | None = None
        # Шаг локальной команды: номер, число шагов, итог (для оверлея «шаг 2 из 5»).
        self.on_command_step: Callable[[int, int, StepResult], None] | None = None

    @property
    def agent(self) -> Agent | None:
        """Агент нейросети; None — нейросеть выключена."""
        return self._agent

    def set_agent(self, agent: Agent | None) -> None:
        """Включает или выключает нейросеть без перезапуска (настройки в интерфейсе).

        Выполняющаяся задача доделывается прежним агентом: фраза читает агента один раз.
        """
        self._agent = agent

    def reload_commands(self, commands: list[Command]) -> None:
        """Подменяет команды матчера, когда фраза не выполняется (файлы команд изменились)."""
        with self._lock:
            self._matcher.set_commands(commands)

    def cancel(self) -> None:
        """Прерывает задачу нейросети (горячая клавиша или «стоп»). Можно звать из любого потока."""
        self._cancel.set()

    def forget(self) -> None:
        """Забывает разговор с нейросетью."""
        if self._agent is not None:
            self._agent.forget()

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
            result = self._handle(text, source, confirmer, new_request())
        self._journal_phrase(result, source)
        return result

    def run_command(
        self, command: Command, source: str = "local", confirmer: Confirmer | None = None
    ) -> PipelineResult:
        """Выполняет команду напрямую, без поиска по фразе: сценарий из интерфейса или по горячей клавише.

        Слотов у такой команды нет: значения взять неоткуда.
        """
        phrase = command.phrases[0] if command.phrases else command.id
        matched = MatchedCommand(command, {}, 100.0, phrase, phrase)
        request = new_request()
        with self._lock:
            started = time.perf_counter()
            outcome = self._run_command(matched, phrase, source, confirmer, request)
            total_ms = (time.perf_counter() - started) * 1000
        if outcome.reply:
            self._speaker.say(outcome.reply)
        match = MatchResult("matched", phrase, [matched])
        result = PipelineResult(
            phrase, "local", match, [outcome], outcome.reply or "", 0.0, total_ms, request=request
        )
        self._journal_phrase(result, source)
        return result

    def _journal_phrase(self, result: PipelineResult, source: str) -> None:
        """Запись о фразе для Истории; её действия уже в журнале с тем же request."""
        if result.match.status == "empty":
            return
        self._executor.journal.write_phrase(
            request=result.request,
            phrase=result.text,
            source=source,
            route=result.route,
            commands=[outcome.matched.command.id for outcome in result.outcomes],
            reply=result.reply,
            ok=result.ok,
            duration_ms=result.total_ms,
            error=result.problem,
        )

    def _handle(self, text: str, source: str, confirmer: Confirmer | None, request: str) -> PipelineResult:
        started = time.perf_counter()
        match = self._matcher.match(text, self._window_provider())
        match_ms = (time.perf_counter() - started) * 1000

        if not match.matched:
            log.info("«%s» не найдена локально (%s) — маршрут LLM", text, match.reason or match.status)
            if match.status == "empty":
                return PipelineResult(text, "llm", match, match_ms=match_ms, request=request)
            llm_agent = self._agent
            agent = None
            if llm_agent is None:
                reply = "Не знаю такой команды"
                self._speaker.say(reply)
            else:
                self._cancel.clear()
                agent = llm_agent.run(
                    text,
                    confirmer=confirmer,
                    cancel=self._cancel,
                    on_step=self.on_agent_step,
                    request=request,
                )
                reply = agent.text
            total_ms = (time.perf_counter() - started) * 1000
            return PipelineResult(
                text,
                "llm",
                match,
                reply=reply,
                match_ms=match_ms,
                total_ms=total_ms,
                agent=agent,
                request=request,
            )

        outcomes: list[CommandOutcome] = []
        for matched in match.matches:
            outcome = self._run_command(matched, text, source, confirmer, request)
            outcomes.append(outcome)
            if not outcome.run.ok:
                break
        reply = " ".join(o.reply for o in outcomes if o.reply)
        total_ms = (time.perf_counter() - started) * 1000
        log.info("Задержки: матчинг %.0f мс, текст → результат %.0f мс", match_ms, total_ms)
        if reply:
            self._speaker.say(reply)
        return PipelineResult(text, "local", match, outcomes, reply, match_ms, total_ms, request=request)

    def _run_command(
        self, matched: MatchedCommand, phrase: str, source: str, confirmer: Confirmer | None, request: str
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
            on_step=self.on_command_step,
            request=request,
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
