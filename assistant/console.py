"""Консоль: ввод фраз текстом (для отладки) и вывод того, что происходит с голосом."""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from assistant.actions import winapi
from assistant.actions.apps import AppIndex
from assistant.commands.loader import CommandStore
from assistant.llm.agent import AgentStatus, AgentStep
from assistant.nlu.matcher import Matcher, MatchResult
from assistant.pipeline import Pipeline, PipelineResult

log = logging.getLogger(__name__)

HELP = """\
Введите фразу так, как сказали бы голосом, например «сделай громче».
Служебные команды:
  :check <фраза>  какая команда сработает и с какой оценкой (без выполнения)
  :delay <сек>    пауза перед выполнением, чтобы успеть переключиться в нужное окно
  :reload         перечитать YAML-команды
  :commands       список команд
  :apps [текст]   найденные приложения (можно с фильтром)
  :windows        открытые окна
  :listen         начать слушать голосовую команду (как горячая клавиша)
  :forget         забыть разговор с нейросетью
  :help           эта справка
Задачу нейросети можно прервать: Ctrl+C в консоли, Ctrl+Alt+Q или голосом «стоп»
  :q              выход"""

_AGENT_STATUS = {
    AgentStatus.DONE: "ответила",
    AgentStatus.CANCELLED: "прервано",
    AgentStatus.TIMEOUT: "не успела за отведённое время",
    AgentStatus.STEP_LIMIT: "превышен лимит шагов",
    AgentStatus.ERROR: "ошибка",
}

_VOICE_MESSAGES = {
    "idle": "Жду обращения",
    "listening": "Слушаю…",
    "holding": "Слушаю, пока клавиши зажаты…",
    "recognizing": "Распознаю…",
}


class ConsoleSpeaker:
    """Выводит ответы ассистента в консоль.

    Во время обработки введённой фразы ответы можно придержать (deferred), чтобы
    напечатать их после подробностей о шагах. Придерживаются только ответы того потока,
    который вызвал deferred(): голосовой поток печатает сразу.
    """

    def __init__(self, name: str, out: TextIO = sys.stdout) -> None:
        self._name = name.capitalize()
        self._out = out
        self._local = threading.local()

    def say(self, text: str) -> None:
        line = f"  {self._name}: {text}"
        held: list[str] | None = getattr(self._local, "held", None)
        if held is not None:
            held.append(line)
            return
        self._out.write(line + "\n")
        self._out.flush()

    @contextmanager
    def deferred(self) -> Iterator[None]:
        """Копит ответы внутри блока и печатает их при выходе."""
        self._local.held = []
        try:
            yield
        finally:
            held, self._local.held = self._local.held, None
            for line in held:
                self._out.write(line + "\n")
            self._out.flush()


class ConsoleApp:
    """Цикл чтения фраз из консоли.

    Args:
        pipeline: конвейер обработки фраз.
        speaker: вывод ответов (придерживается, пока печатаются шаги).
        matcher: матчер (для :check и перезагрузки).
        store: хранилище команд (для :reload и :commands).
        index: индекс приложений (для :apps).
        window_provider: активное окно (для :check).
        ask: функция чтения строки (по умолчанию input).
        out: куда писать.
        on_listen: что сделать по команде :listen (запустить прослушивание), если голос включён.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        speaker: ConsoleSpeaker,
        matcher: Matcher,
        store: CommandStore,
        index: AppIndex,
        window_provider: Callable[[], winapi.WindowInfo],
        ask: Callable[[str], str] = input,
        out: TextIO = sys.stdout,
        on_listen: Callable[[], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._speaker = speaker
        self._matcher = matcher
        self._store = store
        self._index = index
        self._window_provider = window_provider
        self._ask = ask
        self._out = out
        self.on_listen = on_listen
        self._delay = 0.0
        pipeline.on_agent_step = self._print_agent_step

    def run(self) -> None:
        """Читает фразы до :q или Ctrl+C."""
        self._write("Консольный режим. Введите фразу или :help для справки.")
        while True:
            try:
                line = self._ask("> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._write("")
                break
            if not line:
                continue
            if line.startswith(":"):
                if not self._service_command(line):
                    break
                continue
            if self._delay:
                self._write(f"  Выполню через {self._delay:g} с — переключитесь в нужное окно…")
                time.sleep(self._delay)
            with self._speaker.deferred():
                try:
                    result = self._pipeline.handle_text(line, source="local")
                except KeyboardInterrupt:
                    self._write("  Прервано")
                    continue
                except Exception:
                    log.exception("Ошибка при обработке фразы «%s»", line)
                    self._write("  Внутренняя ошибка, подробности в логе")
                    continue
                self._print_result(result)

    def _service_command(self, line: str) -> bool:
        """Выполняет служебную команду. Возвращает False для выхода."""
        name, _, arg = line[1:].partition(" ")
        name, arg = name.lower(), arg.strip()
        if name in ("q", "quit", "exit", "выход"):
            return False
        if name == "help":
            self._write(HELP)
        elif name == "check":
            self._print_match(self._matcher.match(arg, self._window_provider()))
        elif name == "delay":
            try:
                self._delay = max(0.0, float(arg.replace(",", ".") or 0))
            except ValueError:
                self._write("  Укажите число секунд, например :delay 3")
            else:
                self._write(f"  Пауза перед выполнением: {self._delay:g} с")
        elif name == "reload":
            result = self._store.reload()
            self._matcher.set_commands(self._store.commands)
            self._write(f"  Загружено команд: {len(result.commands)}")
            for issue in result.issues:
                self._write(f"  ! {issue}")
        elif name == "commands":
            for command in self._store.commands:
                self._write(f"  {command.id:20} {command.source:14} {', '.join(command.phrases[:3])}")
        elif name == "apps":
            entries = [e for e in self._index.entries() if arg.lower() in e.name.lower()]
            for entry in entries[:100]:
                self._write(f"  {entry.name}  [{entry.kind}]")
            self._write(f"  Всего: {len(entries)}")
        elif name == "forget":
            self._pipeline.forget()
            self._write("  Разговор забыт")
        elif name == "listen":
            if self.on_listen is None:
                self._write("  Голос выключен (запуск с --text или нет моделей)")
            else:
                self.on_listen()
        elif name == "windows":
            if not winapi.IS_WINDOWS:
                self._write("  Доступно только в Windows")
            else:
                for window in winapi.list_windows():
                    self._write(f"  {window.title}  [{window.process}]")
        else:
            self._write(f"  Неизвестная служебная команда :{name}. Список — :help")
        return True

    # --- Сообщения голосового цикла ---------------------------------------------------

    def voice_status(self, state: str, detail: str = "") -> None:
        """Показывает состояние голоса: «Слушаю…», распознанный текст."""
        if state == "recognized":
            self._write(f"  [голос] Распознано: «{detail}»")
        elif state == "latency":
            self._write(f"  [голос] {detail}")
        elif state in _VOICE_MESSAGES:
            self._write(f"  [голос] {_VOICE_MESSAGES[state]}")

    def voice_result(self, result: Any) -> None:
        """Показывает, что выполнено по голосовой команде."""
        self._print_result(result)

    def _print_match(self, match: MatchResult) -> None:
        self._write(f"  Нормализовано: «{match.normalized}»")
        if match.matched:
            for item in match.matches:
                slots = f" {item.slots}" if item.slots else ""
                self._write(f"  Сработает: {item.command.id} ({item.score:.0f}, «{item.phrase}»){slots}")
        else:
            self._write(f"  Локально не найдено ({match.reason or match.status}) — уйдёт в нейросеть")
        for candidate in match.candidates[:3]:
            self._write(f"    кандидат: {candidate.describe()}")

    def _print_agent_step(self, step: AgentStep) -> None:
        """Шаг агента — сразу, пока задача ещё выполняется."""
        mark = "✓" if step.status in ("done", "dry_run") else "✗"
        self._write(f"  [нейросеть] шаг {step.number}: {mark} {step.describe()}")

    def _print_result(self, result: PipelineResult) -> None:
        if result.route == "llm":
            self._print_agent_result(result)
            return
        for outcome in result.outcomes:
            matched = outcome.matched
            slots = f" {matched.slots}" if matched.slots else ""
            self._write(f"  [локально] {matched.command.id} ({matched.score:.0f}){slots}")
            for step in outcome.run.steps:
                mark = "✓" if step.ok else "✗"
                self._write(f"    {mark} {step.action}: {step.text} ({step.duration_ms:.0f} мс)")
            if outcome.run.status.value == "cancelled":
                self._write("    отменено")
        self._write(f"  ({result.total_ms:.0f} мс)")

    def _print_agent_result(self, result: PipelineResult) -> None:
        """Итог задачи нейросети: статус, число запросов и время."""
        agent = result.agent
        if agent is None:
            if result.match.status != "empty":
                self._write(f"  [→ нейросеть] {result.match.reason or result.match.status}, но она выключена")
            return
        if agent.error:
            self._write(f"  [нейросеть] недоступна: {agent.error}")
            return
        self._write(
            f"  [нейросеть] {_AGENT_STATUS[agent.status]}: запросов {agent.requests}, "
            f"нейросеть {agent.llm_ms:.0f} мс, всего {agent.total_ms:.0f} мс"
        )

    def _write(self, text: str) -> None:
        self._out.write(text + "\n")
        self._out.flush()
