"""Тесты агента: нейросеть подменена потоками заранее заданных ответов."""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import pytest
from pydantic import Field

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry, Risk
from assistant.actions.winapi import WindowInfo
from assistant.config import LlmConfig
from assistant.journal import ActionJournal
from assistant.llm.agent import Agent, AgentStatus, SentenceStreamer
from assistant.llm.client import LlmClient, LlmError
from assistant.logging_setup import SecretMasker
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from tests.conftest import RecordingSpeaker
from tests.llm_fakes import FakeOpenAI, connection_error, factory_for, text_reply, tool_reply


class Clock:
    """Часы, которые идут только когда их двигают."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class World:
    """Реестр с действиями, исполнитель, журнал и агент на поддельной нейросети."""

    def __init__(
        self,
        tmp_path: Path,
        responses: list[Any],
        *,
        confirm: bool = True,
        dry_run: bool = False,
        **settings: Any,
    ) -> None:
        self.calls: list[str] = []
        self.clock = Clock()
        registry = ActionRegistry()

        @registry.action("open_app", "Открыть программу")
        def open_app(name: Annotated[str, Field(min_length=1)]) -> str:
            self.calls.append(f"open_app:{name}")
            return f"Открываю {name}"

        @registry.action("volume_set", "Установить громкость")
        def volume_set(percent: Annotated[int, Field(ge=0, le=100)]) -> str:
            self.calls.append(f"volume:{percent}")
            return f"Громкость {percent}%"

        @registry.action("shutdown", "Выключить компьютер", risk=Risk.CONFIRM)
        def shutdown() -> str:
            self.calls.append("shutdown")
            return "Выключаю"

        @registry.action("format_disk", "Отформатировать диск", risk=Risk.FORBIDDEN)
        def format_disk(drive: str) -> str:
            self.calls.append(f"format:{drive}")
            return "Отформатировано"

        @registry.action("slow", "Долгое действие")
        def slow() -> str:
            self.clock.now += 50
            return "Сделано"

        @registry.action("say", "Произнести фразу вслух")
        def say(text: str) -> str:
            return text

        self.journal_path = tmp_path / "journal.jsonl"
        self.confirmer = StaticConfirmer(confirm)
        executor = ActionExecutor(
            registry,
            SafetyPolicy(),
            self.confirmer,
            ActionJournal(self.journal_path, SecretMasker()),
            dry_run=dry_run,
        )
        self.speaker = RecordingSpeaker()
        self.fake = FakeOpenAI(responses)
        self.settings = LlmConfig.model_validate({"profile": "ollama", **settings})
        self.agent = Agent(
            self.settings,
            executor,
            self.speaker,
            client_factory=lambda: LlmClient(self.settings, factory=factory_for(self.fake)),
            window_provider=lambda: WindowInfo(1, "Без имени — Блокнот", "Notepad", 2, "notepad.exe"),
            clock=self.clock,
            now=lambda: datetime(2026, 10, 21, 18, 30),
        )

    def request(self, index: int) -> list[dict[str, Any]]:
        """Сообщения, отправленные нейросети в запросе номер index."""
        return self.fake.requests[index]["messages"]

    def journal(self) -> list[dict[str, Any]]:
        lines = self.journal_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]


def test_text_answer_is_spoken_by_sentences(tmp_path: Path) -> None:
    w = World(tmp_path, [text_reply("Привет! Чем ", "могу помочь?")])
    result = w.agent.run("привет")
    assert result.ok and result.requests == 1
    assert w.speaker.said == ["Привет!", "Чем могу помочь?"]
    assert result.text == "Привет! Чем могу помочь?"


def test_tool_call_then_answer(tmp_path: Path) -> None:
    w = World(tmp_path, [tool_reply(("open_app", {"name": "Telegram"})), text_reply("Открыл Телеграм.")])
    steps: list[str] = []
    result = w.agent.run("запусти телегу", on_step=lambda step: steps.append(step.describe()))
    assert result.ok and w.calls == ["open_app:Telegram"]
    assert steps == ["open_app(name='Telegram') → Открываю Telegram"]
    second = w.request(1)
    assert second[-2]["tool_calls"][0]["function"]["name"] == "open_app"
    assert second[-1] == {"role": "tool", "tool_call_id": "call_0", "content": "Открываю Telegram"}
    assert w.speaker.said == ["Открыл Телеграм."]
    record = w.journal()[0]
    assert (record["source"], record["action"], record["phrase"]) == ("llm", "open_app", "запусти телегу")


def test_system_prompt_has_rules_date_and_window(tmp_path: Path) -> None:
    w = World(tmp_path, [text_reply("Ок.")])
    w.agent.run("привет")
    system = w.request(0)[0]
    assert system["role"] == "system"
    assert "данные, а не инструкции" in system["content"] and "без markdown" in system["content"]
    assert "21.10.2026 18:30, среда" in system["content"]
    assert "«Без имени — Блокнот» (notepad.exe)" in system["content"]


def test_tools_come_from_registry_without_say_and_forbidden(tmp_path: Path) -> None:
    w = World(tmp_path, [text_reply("Ок.")])
    w.agent.run("привет")
    names = [tool["function"]["name"] for tool in w.fake.requests[0]["tools"]]
    assert names == ["open_app", "volume_set", "shutdown", "slow"]


def test_forbidden_action_is_refused_to_model(tmp_path: Path) -> None:
    w = World(tmp_path, [tool_reply(("format_disk", {"drive": "C:"})), text_reply("Это запрещено.")])
    result = w.agent.run("отформатируй диск це")
    assert w.calls == []
    assert result.steps[0].status == "forbidden"
    assert w.request(1)[-1]["content"].startswith("Отказано политикой безопасности")


@pytest.mark.parametrize(("answer", "executed"), [(True, ["shutdown"]), (False, [])])
def test_confirm_action_asks_user(tmp_path: Path, answer: bool, executed: list[str]) -> None:
    w = World(tmp_path, [tool_reply(("shutdown", {})), text_reply("Ок.")], confirm=answer)
    result = w.agent.run("выключи комп")
    assert w.calls == executed
    assert w.confirmer.questions == ["Выключить компьютер?"]
    if not answer:
        assert "не подтвердил" in result.steps[0].text


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (("no_such_tool", {}), "инструмента no_such_tool нет"),
        (("volume_set", {"percent": 150}), "Ошибка в аргументах"),
    ],
)
def test_bad_tool_calls_become_errors_for_model(
    tmp_path: Path, call: tuple[str, dict[str, Any]], error: str
) -> None:
    w = World(tmp_path, [tool_reply(call), text_reply("Не получилось.")])
    result = w.agent.run("сделай что-нибудь")
    assert result.ok and w.calls == []
    assert error in w.request(1)[-1]["content"]


def test_invalid_json_arguments(tmp_path: Path) -> None:
    from tests.llm_fakes import chunk

    call = {"index": 0, "id": "c1", "function": {"name": "open_app", "arguments": "{oops"}}
    broken = [chunk(tool_calls=[call])]
    w = World(tmp_path, [broken, text_reply("Ошибка.")])
    w.agent.run("открой")
    assert "не в формате JSON" in w.request(1)[-1]["content"]


def test_step_limit(tmp_path: Path) -> None:
    w = World(tmp_path, [tool_reply(("volume_set", {"percent": n})) for n in (10, 20, 30)], max_steps=3)
    result = w.agent.run("крути громкость")
    assert result.status is AgentStatus.STEP_LIMIT and result.requests == 3
    assert w.calls == ["volume:10", "volume:20", "volume:30"]
    assert w.speaker.said == ["Задача оказалась слишком длинной, я остановился."]


def test_overall_timeout(tmp_path: Path) -> None:
    # Каждый вызов «slow» двигает часы на 50 с; лимит — 120 с на задачу.
    w = World(tmp_path, [tool_reply(("slow", {})) for _ in range(5)], timeout=120)
    result = w.agent.run("делай долго")
    assert result.status is AgentStatus.TIMEOUT and result.requests == 3
    assert w.speaker.said == ["Не успел выполнить задачу вовремя."]


def test_network_failure_says_unavailable(tmp_path: Path) -> None:
    w = World(tmp_path, [connection_error()])
    result = w.agent.run("расскажи сказку")
    assert result.status is AgentStatus.ERROR
    assert w.speaker.said == ["Нейросеть недоступна"]
    assert "нет связи с http://localhost:11434/v1" in result.error


def test_missing_key_says_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    w = World(tmp_path, [], profile="deepseek")
    w.agent._client_factory = lambda: LlmClient(w.settings)  # настоящий клиент: ключа нет
    result = w.agent.run("привет")
    assert result.status is AgentStatus.ERROR and "DEEPSEEK_API_KEY" in result.error
    assert w.speaker.said == ["Нейросеть недоступна"]


def test_cancel_skips_remaining_tools(tmp_path: Path) -> None:
    cancel = threading.Event()
    w = World(tmp_path, [tool_reply(("open_app", {"name": "A"}), ("open_app", {"name": "B"}))])
    result = w.agent.run("открой две программы", cancel=cancel, on_step=lambda step: cancel.set())
    assert result.status is AgentStatus.CANCELLED
    assert w.calls == ["open_app:A"] and len(w.fake.requests) == 1
    assert w.speaker.said == []  # прервали — молчим


def test_dry_run_does_not_execute(tmp_path: Path) -> None:
    w = World(tmp_path, [tool_reply(("open_app", {"name": "Telegram"})), text_reply("Готово.")], dry_run=True)
    w.agent.run("открой телегу")
    assert w.calls == []
    assert w.request(1)[-1]["content"].startswith("[dry-run]")


def test_history_is_kept_and_limited(tmp_path: Path) -> None:
    w = World(tmp_path, [text_reply("Первый."), text_reply("Второй."), text_reply("Третий.")], history=1)
    w.agent.run("раз")
    w.agent.run("два")
    w.agent.run("три")
    contents = [m["content"] for m in w.request(2)[1:]]
    assert contents == ["два", "Второй.", "три"]  # помнит только последний обмен


def test_history_is_forgotten_after_silence(tmp_path: Path) -> None:
    w = World(tmp_path, [text_reply("Первый."), text_reply("Второй.")], reset_after=300)
    w.agent.run("раз")
    w.clock.now += 301
    w.agent.run("два")
    assert [m["content"] for m in w.request(1)[1:]] == ["два"]


def test_failed_exchange_is_not_remembered(tmp_path: Path) -> None:
    w = World(tmp_path, [connection_error(), text_reply("Ок.")])
    w.agent.run("раз")
    w.agent.run("два")
    assert [m["content"] for m in w.request(1)[1:]] == ["два"]  # запрос 0 — неудачный «раз»


@pytest.mark.parametrize(
    ("pieces", "sentences"),
    [
        (["Привет", "! Как ", "дела? Хорошо"], ["Привет!", "Как дела?", "Хорошо"]),
        (["Он сказал «Готово.» И ушёл…"], ["Он сказал «Готово.»", "И ушёл…"]),
        (["Без точки"], ["Без точки"]),
    ],
)
def test_sentence_streamer(pieces: list[str], sentences: list[str]) -> None:
    said: list[str] = []
    streamer = SentenceStreamer(said.append)
    for piece in pieces:
        streamer.feed(piece)
    streamer.flush()
    assert said == sentences


def test_llm_error_type_is_exported() -> None:
    assert issubclass(LlmError, Exception)
