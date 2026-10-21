"""Таймеры и напоминания: срабатывание, файл, пропущенное, действия и фразы."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.apps import AppIndex
from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.actions.timers import parse_moment
from assistant.actions.timers import register as register_timers
from assistant.commands.loader import CommandStore
from assistant.config import AppConfig
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.main import build_registry
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.timers import Alarm, Alarms, duration_text, moment_text
from tests.conftest import NO_WINDOW, ROOT, RecordingSpeaker

TZ = timezone(timedelta(hours=3))
START = datetime(2026, 10, 21, 17, 0, tzinfo=TZ)


class Clock:
    """Часы, которые двигает тест."""

    def __init__(self, now: datetime = START) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


def fired_into(alarms: Alarms) -> list[tuple[Alarm, bool]]:
    fired: list[tuple[Alarm, bool]] = []
    alarms.on_fire = lambda alarm, missed: fired.append((alarm, missed))
    return fired


def test_durations_and_moments_in_russian() -> None:
    assert duration_text(60) == "1 минуту"
    assert duration_text(150) == "2 минуты 30 секунд"
    assert duration_text(300) == "5 минут"
    assert duration_text(3600 * 2 + 60 * 21) == "2 часа 21 минуту"
    assert duration_text(45) == "45 секунд"
    assert moment_text(START.replace(hour=18, minute=5), START) == "в 18:05"
    assert moment_text(START + timedelta(days=1), START) == "завтра в 17:00"
    assert moment_text(START + timedelta(days=3), START) == "24.10 в 17:00"


def test_timer_fires_when_due(tmp_path: Path) -> None:
    clock = Clock()
    alarms = Alarms(tmp_path / "timers.json", now=clock)
    fired = fired_into(alarms)
    changes: list[bool] = []
    alarms.on_change = lambda: changes.append(True)
    timer = alarms.add_timer(300, "чай")
    assert timer.title() == "таймер на 5 минут «чай»" and changes == [True]
    clock.advance(minutes=4)
    assert alarms.tick() == [] and fired == []
    clock.advance(minutes=1)
    assert alarms.tick() == [timer]
    assert fired == [(timer, False)] and alarms.active() == []
    assert timer.message() == "Таймер на 5 минут «чай»: время вышло"


def test_list_survives_restart_and_missed_is_announced(tmp_path: Path) -> None:
    path = tmp_path / "timers.json"
    clock = Clock()
    alarms = Alarms(path, now=clock)
    soon = alarms.add_reminder("позвонить маме", START + timedelta(minutes=30))
    later = alarms.add_reminder("вынести мусор", START + timedelta(hours=5))
    assert [item["text"] for item in json.loads(path.read_text(encoding="utf-8"))["alarms"]] == [
        "позвонить маме",
        "вынести мусор",
    ]
    clock.advance(hours=1)  # программа была закрыта, пока подошло первое
    restarted = Alarms(path, now=clock)
    fired = fired_into(restarted)
    restarted.start()
    try:
        assert [(alarm.text, missed) for alarm, missed in fired] == [("позвонить маме", True)]
        assert fired[0][0].message(missed=True) == (
            "Пока программа была закрыта, в 17:30 сработало: напоминаю: позвонить маме"
        )
        assert [alarm.id for alarm in restarted.active()] == [later.id] and soon.id != later.id
    finally:
        restarted.stop()


def test_cancel_by_id_text_nearest_and_all() -> None:
    clock = Clock()
    alarms = Alarms(None, now=clock)
    tea = alarms.add_timer(300, "чай")
    pasta = alarms.add_timer(600, "макароны")
    mom = alarms.add_reminder("позвонить маме", START + timedelta(hours=2))
    assert alarms.cancel(alarm_id=pasta.id) == [pasta]
    assert alarms.cancel("МАМ") == [mom]
    assert alarms.cancel() == [tea]  # без условия — ближайший таймер
    alarms.add_timer(60)
    alarms.add_reminder("x", START + timedelta(hours=1))
    assert len(alarms.cancel(everything=True)) == 2 and alarms.active() == []


def test_broken_file_starts_empty(tmp_path: Path) -> None:
    path = tmp_path / "timers.json"
    path.write_text("{не json", encoding="utf-8")
    alarms = Alarms(path, now=Clock())
    alarms.start()
    try:
        assert alarms.active() == []
    finally:
        alarms.stop()


def test_thread_fires_real_timer(tmp_path: Path) -> None:
    alarms = Alarms(None)
    done = threading.Event()
    alarms.on_fire = lambda alarm, missed: done.set()
    alarms.start()
    try:
        alarms.add_timer(0.2)
        assert done.wait(3), "таймер не сработал"
    finally:
        alarms.stop()


@pytest.fixture
def timer_actions() -> tuple[ActionRegistry, Alarms, Clock]:
    clock = Clock()
    alarms = Alarms(None, now=clock)
    registry = ActionRegistry()
    register_timers(registry, alarms, now=clock)
    return registry, alarms, clock


def run(registry: ActionRegistry, name: str, **args: Any) -> str:
    action = registry.get(name)
    return action.run(action.parse_args(args))


def test_timer_actions(timer_actions: tuple[ActionRegistry, Alarms, Clock]) -> None:
    registry, alarms, clock = timer_actions
    assert run(registry, "set_timer", minutes=5) == "Поставил таймер на 5 минут"
    assert (
        run(registry, "set_timer", hours=1, minutes=30, label="пирог")
        == "Поставил таймер на 1 час 30 минут «пирог»"
    )
    with pytest.raises(ActionError, match="На сколько"):
        run(registry, "set_timer")
    clock.advance(minutes=2)
    assert run(registry, "list_timers") == (
        "таймер на 5 минут, осталось 3 минуты; таймер на 1 час 30 минут «пирог», осталось 1 час 28 минут"
    )
    assert run(registry, "cancel_timer", what="пирог") == "Отменил таймер на 1 час 30 минут «пирог»"
    assert run(registry, "cancel_timer", what="суп") == "Не нашёл таймер или напоминание «суп»"
    assert run(registry, "cancel_timer", everything=True) == "Отменил таймер на 5 минут"
    assert run(registry, "list_timers") == "Таймеров и напоминаний нет"
    assert alarms.active() == []


def test_reminder_actions(timer_actions: tuple[ActionRegistry, Alarms, Clock]) -> None:
    registry, alarms, _ = timer_actions
    assert (
        run(registry, "set_reminder", text="полить цветы", minutes=10)
        == "Напомню через 10 минут: полить цветы"
    )
    assert run(registry, "set_reminder", text="ужин", at="19:30") == "Напомню в 19:30: ужин"
    assert run(registry, "set_reminder", text="зарядка", at="7.00") == "Напомню завтра в 7:00: зарядка"
    assert run(registry, "set_reminder", text="врач", at="2026-10-24 09:00") == "Напомню 24.10 в 9:00: врач"
    with pytest.raises(ActionError, match="Когда напомнить"):
        run(registry, "set_reminder", text="что-то")
    with pytest.raises(ActionError, match="уже прошло"):
        run(registry, "set_reminder", text="вчера", at="2026-10-20 09:00")
    assert [a.text for a in alarms.active()] == ["полить цветы", "ужин", "зарядка", "врач"]


def test_parse_moment_rejects_garbage() -> None:
    with pytest.raises(ActionError):
        parse_moment("25:00", START)
    with pytest.raises(ActionError):
        parse_moment("в обед", START)


@pytest.mark.parametrize(
    ("phrase", "seconds"),
    [
        ("джарвис поставь таймер на пять минут", 300),
        ("засеки две минуты", 120),
        ("таймер на минуту", 60),
        ("поставь таймер на тридцать секунд", 30),
        ("таймер на два часа", 7200),
        ("таймер на полчаса", 1800),
    ],
)
def test_phrases_set_timers(phrase: str, seconds: int, tmp_path: Path, config: AppConfig) -> None:
    alarms = Alarms(None)
    registry = build_registry(config, AppIndex(config.apps, scanner=list), RecordingSpeaker(), alarms)
    store = CommandStore(ROOT / "commands", "my.yaml", registry)
    assert store.reload().issues == []
    matcher = Matcher(Normalizer(config.wake_words, config.matcher.filler_words), config.matcher)
    matcher.set_commands(store.commands)
    executor = ActionExecutor(
        registry, SafetyPolicy(), StaticConfirmer(True), ActionJournal(tmp_path / "j.jsonl", SecretMasker())
    )
    speaker = RecordingSpeaker()
    result = Pipeline(matcher, executor, speaker, lambda: NO_WINDOW).handle_text(phrase)
    assert result.route == "local" and [a.seconds for a in alarms.active()] == [seconds]
    assert speaker.said == [f"Поставил таймер на {duration_text(seconds)}"]
