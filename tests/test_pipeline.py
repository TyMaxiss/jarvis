"""Тесты конвейера: фраза → команда → действия → ответ."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

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

    def run(self, text: str, *, confirmer: object, cancel: object, on_step: object) -> object:
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
