"""Тесты конвейера: фраза → команда → действия → ответ."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.commands.loader import Command, CommandStore
from assistant.config import MatcherConfig
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.conftest import NO_WINDOW, RecordingSpeaker

PACK = """
- id: volume_set
  phrases: ["громкость на {percent}"]
  slots: {percent: int}
  reply: "Громкость {percent}"
  steps:
    - set_volume: {percent: "{percent}"}

- id: time
  phrases: ["который час"]
  reply: "{result}"
  steps:
    - get_time: {}

- id: open
  phrases: ["открой {name}"]
  steps:
    - open_app: {name: "{name}"}

- id: shutdown
  phrases: ["выключи компьютер"]
  confirm: true
  steps:
    - get_time: {}
"""


def build(tmp_path: Path, confirm_answer: bool = True) -> tuple[Pipeline, RecordingSpeaker, list[str]]:
    calls: list[str] = []
    reg = ActionRegistry()

    @reg.action("set_volume", "Громкость")
    def set_volume(percent: Annotated[int, Field(ge=0, le=100)]) -> str:
        calls.append(f"volume:{percent}")
        return f"{percent}%"

    @reg.action("get_time", "Время")
    def get_time() -> str:
        calls.append("time")
        return "Сейчас 12:00"

    @reg.action("open_app", "Открыть")
    def open_app(name: str) -> str:
        raise ActionError(f"Не нашёл приложение «{name}»")

    (tmp_path / "pack.yaml").write_text(PACK, encoding="utf-8")
    store = CommandStore(tmp_path, "my.yaml", reg)
    assert store.reload().issues == []
    matcher = Matcher(Normalizer(["джарвис"]), MatcherConfig())
    matcher.set_commands(store.commands)
    journal = ActionJournal(tmp_path / "journal.jsonl", SecretMasker())
    executor = ActionExecutor(reg, SafetyPolicy(), StaticConfirmer(confirm_answer), journal)
    speaker = RecordingSpeaker()
    return Pipeline(matcher, executor, speaker, lambda: NO_WINDOW), speaker, calls


def test_local_command_with_reply(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    result = pipeline.handle_text("Джарвис, громкость на сорок")
    assert result.route == "local" and result.handled
    assert calls == ["volume:40"]
    assert speaker.said == ["Громкость 40"]
    assert result.total_ms >= result.match_ms >= 0


def test_reply_with_result(tmp_path: Path) -> None:
    pipeline, speaker, _ = build(tmp_path)
    pipeline.handle_text("который час")
    assert speaker.said == ["Сейчас 12:00"]


def test_composite_runs_all_parts(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    pipeline.handle_text("громкость на 10 и который час")
    assert calls == ["volume:10", "time"]
    assert speaker.said == ["Громкость 10 Сейчас 12:00"]


def test_failed_action_is_spoken(tmp_path: Path) -> None:
    pipeline, speaker, _ = build(tmp_path)
    result = pipeline.handle_text("открой фотошоп")
    assert not result.outcomes[0].run.ok
    assert speaker.said == ["Не нашёл приложение «фотошоп»"]


def test_invalid_slot_value_is_reported(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    # Шаблон пропускает любое число, но схема действия ограничивает громкость 0–100.
    pipeline.handle_text("громкость на сто пятьдесят")
    assert calls == []
    assert speaker.said == ["Не могу выполнить: set_volume: percent — должно быть не больше 100"]


def test_confirmation_declined(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path, confirm_answer=False)
    pipeline.handle_text("выключи компьютер")
    assert calls == []
    assert speaker.said == ["Отменено"]


def test_unknown_phrase_goes_to_llm_route(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    result = pipeline.handle_text("расскажи сказку")
    assert result.route == "llm" and not result.handled
    assert calls == []
    assert speaker.said == ["Не знаю такой команды"]


def test_empty_phrase_is_silent(tmp_path: Path) -> None:
    pipeline, speaker, _ = build(tmp_path)
    result = pipeline.handle_text("Джарвис")
    assert result.match.status == "empty"
    assert speaker.said == []


class StubAgent:
    """Агент нейросети, который отвечает заранее заданным результатом."""

    def __init__(self, status: str = "done") -> None:
        from assistant.llm.agent import AgentResult, AgentStatus

        self.result = AgentResult(text="Ответ нейросети", status=AgentStatus(status))
        self.calls: list[tuple[str, object]] = []
        self.forgotten = False

    def run(self, text: str, *, confirmer: object, cancel: object, on_step: object, request: str) -> object:
        self.calls.append((text, cancel))
        return self.result

    def forget(self) -> None:
        self.forgotten = True


def test_unknown_phrase_goes_to_agent(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    agent = StubAgent()
    pipeline._agent = agent  # type: ignore[assignment]
    result = pipeline.handle_text("расскажи сказку")
    assert result.route == "llm" and result.handled and result.reply == "Ответ нейросети"
    assert [text for text, _ in agent.calls] == ["расскажи сказку"]
    assert calls == [] and speaker.said == []  # ответ озвучивает сам агент
    pipeline.forget()
    assert agent.forgotten


def test_failed_agent_answer_is_not_handled(tmp_path: Path) -> None:
    pipeline, _, _ = build(tmp_path)
    pipeline._agent = StubAgent(status="error")  # type: ignore[assignment]
    assert not pipeline.handle_text("расскажи сказку").handled


def test_cancel_reaches_agent_and_is_reset(tmp_path: Path) -> None:
    pipeline, _, _ = build(tmp_path)
    agent = StubAgent()
    pipeline._agent = agent  # type: ignore[assignment]
    pipeline.cancel()  # нажали «стоп» до задачи — новая задача не должна сразу прерваться
    pipeline.handle_text("расскажи сказку")
    cancel = agent.calls[0][1]
    assert not cancel.is_set()  # type: ignore[attr-defined]
    pipeline.cancel()
    assert cancel.is_set()  # type: ignore[attr-defined]


def test_run_command_without_phrase(tmp_path: Path) -> None:
    pipeline, speaker, calls = build(tmp_path)
    steps: list[tuple[int, int]] = []
    pipeline.on_command_step = lambda number, total, step: steps.append((number, total))
    command = next(c for c in pipeline_commands(tmp_path) if c.id == "time")
    result = pipeline.run_command(command, source="hotkey")
    assert result.route == "local" and result.handled
    assert calls == ["time"] and speaker.said == ["Сейчас 12:00"]
    assert steps == [(1, 1)]


def test_run_command_asks_confirmation(tmp_path: Path) -> None:
    pipeline, _, calls = build(tmp_path, confirm_answer=False)
    command = next(c for c in pipeline_commands(tmp_path) if c.id == "shutdown")
    result = pipeline.run_command(command)
    assert calls == [] and result.reply == "Отменено"


def test_reload_commands_and_toggle_agent(tmp_path: Path) -> None:
    pipeline, speaker, _ = build(tmp_path)
    pipeline.reload_commands([])
    assert pipeline.handle_text("который час").route == "llm"
    assert speaker.said == ["Не знаю такой команды"]
    assert pipeline.agent is None


def pipeline_commands(tmp_path: Path) -> list[Command]:
    """Команды из того же пака, что и в build (он уже записан в tmp_path)."""
    reg = ActionRegistry()

    @reg.action("set_volume", "Громкость")
    def set_volume(percent: int) -> str:
        return ""

    @reg.action("get_time", "Время")
    def get_time() -> str:
        return ""

    @reg.action("open_app", "Открыть")
    def open_app(name: str) -> str:
        return ""

    store = CommandStore(tmp_path, "my.yaml", reg)
    store.reload()
    return store.commands


# --- «Стоп» ------------------------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["стоп", "Джарвис, стоп!", "хватит", "Отмена.", "стоп, хватит"])
def test_stop_phrase_stops_instead_of_command(tmp_path: Path, phrase: str) -> None:
    pipeline, speaker, calls = build(tmp_path)
    stopped: list[str] = []
    pipeline.on_stop.append(lambda: stopped.append("речь"))
    result = pipeline.handle_text(phrase, "text")
    assert result.stopped and result.ok and result.reply == "Остановлено"
    assert result.outcomes == [] and result.agent is None
    assert stopped == ["речь"] and calls == [] and speaker.said == []
    phrase_line = json.loads((tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert phrase_line["phrase"] == phrase and phrase_line["reply"] == "Остановлено" and phrase_line["ok"]


def test_command_with_stop_word_is_still_a_command(tmp_path: Path) -> None:
    pipeline, _, _ = build(tmp_path)
    assert not pipeline.is_stop("отмени таймер") and not pipeline.is_stop("стоп музыка")
    assert pipeline.is_stop("Джарвис, отмени")


def test_stop_does_not_wait_for_running_task(tmp_path: Path) -> None:
    # Задача нейросети идёт и держит конвейер; «стоп» из другого потока не ждёт её, а прерывает.
    pipeline, _, _ = build(tmp_path)
    started = threading.Event()

    class SlowAgent(StubAgent):
        def run(
            self, text: str, *, confirmer: object, cancel: threading.Event, on_step: object, request: str
        ) -> object:  # type: ignore[override]
            from assistant.llm.agent import AgentResult, AgentStatus

            started.set()
            cancel.wait(timeout=5)
            return AgentResult(text="", status=AgentStatus.CANCELLED if cancel.is_set() else AgentStatus.DONE)

    pipeline._agent = SlowAgent()  # type: ignore[assignment]
    results: list[object] = []
    worker = threading.Thread(target=lambda: results.append(pipeline.handle_text("расскажи сказку")))
    worker.start()
    assert started.wait(2) and pipeline.busy
    assert pipeline.handle_text("стоп").stopped
    worker.join(2)
    assert not worker.is_alive() and not pipeline.busy
    task = results[0]
    assert task.reply == "Остановлено" and not task.ok  # type: ignore[attr-defined]


def test_stop_hook_errors_do_not_break_others(tmp_path: Path) -> None:
    pipeline, _, _ = build(tmp_path)
    done: list[int] = []

    def broken() -> None:
        raise RuntimeError("окно уже закрыто")

    pipeline.on_stop += [broken, lambda: done.append(1)]
    pipeline.stop()
    assert done == [1]


def test_stop_cancels_phrases_waiting_in_queue(tmp_path: Path) -> None:
    # Задача из текста идёт, за ней в очереди голосовая фраза и сценарий; «стоп» отменяет все три.
    pipeline, speaker, calls = build(tmp_path)
    started = threading.Event()

    class SlowAgent(StubAgent):
        def run(
            self, text: str, *, confirmer: object, cancel: threading.Event, on_step: object, request: str
        ) -> object:  # type: ignore[override]
            from assistant.llm.agent import AgentResult, AgentStatus

            started.set()
            cancel.wait(timeout=5)
            return AgentResult(text="", status=AgentStatus.CANCELLED)

    pipeline._agent = SlowAgent()  # type: ignore[assignment]
    results: dict[str, object] = {}
    command = next(c for c in pipeline_commands(tmp_path) if c.id == "time")
    workers = [
        threading.Thread(target=lambda: results.setdefault("task", pipeline.handle_text("расскажи сказку"))),
        threading.Thread(
            target=lambda: results.setdefault("queued", pipeline.handle_text("который час", "voice"))
        ),
        threading.Thread(target=lambda: results.setdefault("scenario", pipeline.run_command(command))),
    ]
    workers[0].start()
    assert started.wait(2)
    for worker in workers[1:]:
        worker.start()
    time.sleep(0.1)  # фраза и сценарий ждут очереди
    pipeline.handle_text("стоп")
    for worker in workers:
        worker.join(2)
    assert all(result.stopped for key, result in results.items() if key != "task")  # type: ignore[attr-defined]
    assert calls == [] and speaker.said == []  # «который час» и сценарий не выполнены
    assert pipeline.handle_text("который час").reply == "Сейчас 12:00"  # следующие фразы — как обычно
