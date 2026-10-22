"""Части интерфейса без окон: события, ход команды, подтверждение, оверлей, наблюдение за командами."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.registry import ActionError
from assistant.commands.watcher import CommandWatcher
from assistant.ui.activity import Activity, UiSpeaker, VoiceStatus
from assistant.ui.confirm import UiConfirmer
from assistant.ui.events import EventBus, event_script
from assistant.ui.overlay import OverlayController
from assistant.ui.tray import tray_image
from assistant.ui.win32 import MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT, bottom_center, hotkey_modifiers
from tests.conftest import wait_until
from tests.ui_fakes import RecordingBus

# --- События -------------------------------------------------------------------------


def test_event_script_is_valid_js_call() -> None:
    script = event_script("reply", {"text": "Готово «Блокнот»\n</script>"})
    assert script.startswith("window.__assistantEvent && window.__assistantEvent(")
    payload = json.loads(script[script.index("(") + 1 : -1])
    assert payload == {"type": "reply", "payload": {"text": "Готово «Блокнот»\n</script>"}}


def test_bus_sends_to_attached_windows_in_order() -> None:
    bus = EventBus()
    main: list[str] = []
    overlay: list[str] = []
    bus.emit("state", {"lost": True})  # окон ещё нет — событие никуда не уходит
    bus.attach("main", main.append)
    bus.attach("overlay", overlay.append)
    for number in range(20):
        bus.emit("step", {"number": number})
    bus.emit("level", {"value": 0.5}, to=["overlay"])
    assert wait_until(lambda: len(main) == 20 and len(overlay) == 21)
    assert [json.loads(s[s.index("(") + 1 : -1])["payload"]["number"] for s in main] == list(range(20))
    assert '"level"' in overlay[-1] and bus.attached("main")
    bus.close()


def test_slow_window_does_not_block_others() -> None:
    bus = EventBus(limit=5)
    release = threading.Event()
    fast: list[str] = []
    bus.attach("slow", lambda script: release.wait(2))
    bus.attach("fast", fast.append)
    started = time.monotonic()
    for _ in range(50):
        bus.emit("level", {"value": 0.1})
    assert time.monotonic() - started < 0.5  # emit не ждёт окна
    assert wait_until(lambda: len(fast) == 5)  # у быстрого окна свои 5 мест в очереди
    release.set()
    bus.close()


# --- Ход команды -----------------------------------------------------------------------


class Result:
    """Как PipelineResult."""

    def __init__(self, text: str, route: str, ok: bool, reply: str = "", status: str = "matched") -> None:
        self.text, self.route, self.reply, self.total_ms = text, route, reply, 512.3
        self.match = type("Match", (), {"status": status})()
        run = type("Run", (), {"ok": ok})()
        self.outcomes = [type("Outcome", (), {"run": run})()]
        self.agent = type("Agent", (), {"ok": ok})() if route == "llm" else None


def test_activity_tracks_command_and_recent() -> None:
    bus = RecordingBus()
    activity = Activity(bus, now=lambda: datetime(2026, 10, 21, 18, 42), limit=2)
    activity.begin("открой блокнот", "voice")
    activity.step(1, 2, "open_app", "Открываю Блокнот", True)
    UiSpeaker(activity).say("Открываю.")
    UiSpeaker(activity).say("Готово.")
    item = activity.finish(Result("открой блокнот", "local", True), "voice")
    assert bus.of("recognized") == [{"text": "открой блокнот", "source": "voice"}]
    assert bus.of("reply")[-1] == {"text": "Открываю. Готово."}
    assert item == {
        "id": 1, "time": "18:42", "phrase": "открой блокнот", "route": "local", "reply": "Открываю. Готово.",
        "ok": True, "ms": 512, "source": "voice",
    }  # fmt: skip
    activity.begin("что-то", "text")
    activity.finish(Result("что-то", "llm", False, reply="Нейросеть недоступна"), "text")
    activity.failed("третье", "hotkey", "Ошибка")
    assert [r["phrase"] for r in activity.recent] == ["третье", "что-то"]  # помнится только limit
    assert activity.recent[1]["ok"] is False and activity.recent[1]["source"] == "text"


def test_empty_phrase_is_not_recorded() -> None:
    activity = Activity(RecordingBus())
    assert activity.finish(Result("", "llm", False, status="empty"), "voice") is None
    assert activity.recent == []


def test_voice_status_routes_messages() -> None:
    bus = RecordingBus()
    activity = Activity(bus)
    states: list[str] = []
    status = VoiceStatus(activity, states.append)
    for state in ("listening", "recognizing", "processing", "paused", "idle"):
        status.voice_status(state)
    status.voice_status("recognized", "сделай громче")
    status.voice_status("latency", "500 мс")
    status.voice_status("stopped", "стоп")
    assert states == ["listening", "recognizing", "processing", "paused", "idle"]
    assert bus.of("recognized") == [{"text": "сделай громче", "source": "voice"}]


# --- Подтверждение -----------------------------------------------------------------------


class FakeVoiceConfirmer:
    """Голосовое подтверждение: отвечает answer через delay секунд (или пока не отменят)."""

    def __init__(self, answer: bool | None, delay: float = 0.05) -> None:
        self.answer_value = answer
        self.delay = delay
        self.cancelled = False

    def answer(self, question: str, cancel: threading.Event) -> bool | None:
        if cancel.wait(self.delay):
            self.cancelled = True
            return None
        return self.answer_value


def ask_later(confirmer: UiConfirmer, use_voice: bool) -> tuple[threading.Thread, list[bool]]:
    result: list[bool] = []
    thread = threading.Thread(target=lambda: result.append(confirmer.ask("Выключить компьютер?", use_voice)))
    thread.start()
    return thread, result


def test_button_answers_question() -> None:
    bus = RecordingBus()
    confirmer = UiConfirmer(bus, timeout=5)
    thread, result = ask_later(confirmer, use_voice=False)
    assert wait_until(lambda: bus.of("confirm"))
    question = bus.of("confirm")[0]
    assert question["question"] == "Выключить компьютер?"
    assert not confirmer.answer("другой-вопрос", True)
    assert confirmer.answer(question["id"], True)
    thread.join(2)
    assert result == [True] and bus.of("confirm_done") == [{"id": question["id"]}]
    assert not confirmer.answer(question["id"], True)  # вопрос уже снят


def test_silence_means_no() -> None:
    confirmer = UiConfirmer(RecordingBus(), timeout=0.2)
    assert confirmer.for_source("text").confirm("Точно?") is False


def test_voice_answer_and_overlay_hooks() -> None:
    changes: list[bool] = []
    confirmer = UiConfirmer(RecordingBus(), voice=FakeVoiceConfirmer(True), on_change=changes.append)
    assert confirmer.for_source("voice").confirm("Выключить компьютер?") is True
    assert changes == [True, False]
    # Текстовую команду голосом не подтверждают и оверлей не показывают.
    confirmer.voice = FakeVoiceConfirmer(True)
    confirmer._timeout = 0.2
    assert confirmer.for_source("text").confirm("Точно?") is False
    assert changes == [True, False]


def test_button_stops_voice_listening() -> None:
    bus = RecordingBus()
    voice = FakeVoiceConfirmer(True, delay=5)
    confirmer = UiConfirmer(bus, voice=voice, timeout=10)
    thread, result = ask_later(confirmer, use_voice=True)
    assert wait_until(lambda: bus.of("confirm"))
    confirmer.answer(bus.of("confirm")[0]["id"], False)
    thread.join(2)
    assert result == [False] and wait_until(lambda: voice.cancelled)


def test_stop_takes_back_questions() -> None:
    # Ассистента остановили (кнопка, Ctrl+Alt+Q, «стоп»): вопрос «да/нет» снимается ответом «нет».
    bus = RecordingBus()
    voice = FakeVoiceConfirmer(True, delay=5)
    confirmer = UiConfirmer(bus, voice=voice, timeout=10)
    thread, result = ask_later(confirmer, use_voice=True)
    assert wait_until(lambda: bus.of("confirm"))
    confirmer.cancel_all()
    thread.join(2)
    assert result == [False] and wait_until(lambda: voice.cancelled)
    confirmer.cancel_all()  # вопросов нет — ничего не происходит


def test_button_still_works_shortly_after_voice_silence() -> None:
    bus = RecordingBus()
    confirmer = UiConfirmer(bus, voice=FakeVoiceConfirmer(None, delay=0.05), timeout=10, grace=1.0)
    thread, result = ask_later(confirmer, use_voice=True)
    assert wait_until(lambda: bus.of("confirm"))
    time.sleep(0.3)  # голосом не ответили
    confirmer.answer(bus.of("confirm")[0]["id"], True)
    thread.join(2)
    assert result == [True]
    confirmer = UiConfirmer(RecordingBus(), voice=FakeVoiceConfirmer(None, delay=0.05), timeout=10, grace=0.2)
    started = time.monotonic()
    assert confirmer.ask("Точно?", use_voice=True) is False
    assert time.monotonic() - started < 2  # после молчания ждём только grace


# --- Оверлей -----------------------------------------------------------------------------


class FakeTimer:
    def __init__(self, seconds: float, action: Any) -> None:
        self.seconds, self.action = seconds, action
        self.started = self.cancelled = False
        self.daemon = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True


class OverlayWorld:
    def __init__(self, enabled: bool = True) -> None:
        self.calls: list[str] = []
        self.timers: list[FakeTimer] = []
        self.enabled = enabled

        def timer(seconds: float, action: Any) -> FakeTimer:
            self.timers.append(FakeTimer(seconds, action))
            return self.timers[-1]

        self.overlay = OverlayController(
            lambda: self.calls.append("show"),
            lambda: self.calls.append("hide"),
            lambda: self.enabled,
            timer=timer,
        )


def test_overlay_shows_while_voice_is_active_and_hides_after_3s() -> None:
    world = OverlayWorld()
    overlay = world.overlay
    for state in ("listening", "capturing", "recognizing", "processing"):
        overlay.voice_state(state)
    assert world.calls == ["show"] and overlay.listening
    overlay.voice_state("idle")
    assert world.timers[-1].seconds == 3.0 and world.timers[-1].started
    overlay.voice_state("listening")  # продолжение без имени — оверлей остаётся
    assert world.timers[-1].cancelled and world.calls == ["show"]
    overlay.voice_state("idle")
    world.timers[-1].action()
    assert world.calls == ["show", "hide"] and not overlay.visible


def test_overlay_waits_for_confirmation() -> None:
    world = OverlayWorld()
    world.overlay.confirming(True)
    world.overlay.voice_state("idle")
    assert world.calls == ["show"] and not world.timers
    world.overlay.confirming(False)
    world.timers[-1].action()
    assert world.calls == ["show", "hide"]


def test_disabled_overlay_never_shows() -> None:
    world = OverlayWorld(enabled=False)
    world.overlay.voice_state("listening")
    assert world.calls == []
    world.enabled = True
    world.overlay.voice_state("capturing")
    world.overlay.hide_now()
    assert world.calls == ["show", "hide"]


def test_overlay_errors_do_not_break_voice() -> None:
    def broken() -> None:
        raise RuntimeError("окна нет")

    overlay = OverlayController(broken, broken, lambda: True)
    overlay.voice_state("listening")  # исключение в логе, а не в голосовом цикле
    assert overlay.visible


# --- Наблюдение за командами ------------------------------------------------------------------


def test_watcher_reloads_after_files_settle(tmp_path: Path) -> None:
    pack = tmp_path / "my.yaml"
    pack.write_text("[]\n", encoding="utf-8")
    calls: list[int] = []
    watcher = CommandWatcher(tmp_path, lambda: calls.append(1))
    assert not watcher.check()
    pack.write_text("- id: x\n", encoding="utf-8")
    assert not watcher.check()  # первое изменение — ждём, не пишется ли файл дальше
    assert watcher.check() and calls == [1]
    assert not watcher.check()
    (tmp_path / "new.yaml").write_text("[]\n", encoding="utf-8")
    watcher.check()
    assert watcher.check() and calls == [1, 1]
    (tmp_path / "notes.txt").write_text("не команды", encoding="utf-8")
    assert not watcher.check() and not watcher.check()


def test_watcher_survives_errors(tmp_path: Path) -> None:
    def broken() -> None:
        raise ValueError("битый YAML")

    watcher = CommandWatcher(tmp_path, broken)
    (tmp_path / "my.yaml").write_text("x", encoding="utf-8")
    watcher.check()
    assert watcher.check()


# --- WinAPI-помощники и значок -----------------------------------------------------------------


def test_overlay_position_is_bottom_center() -> None:
    assert bottom_center((480, 64), (0, 0, 1920, 1040), 24) == (720, 952)
    assert bottom_center((720, 96), (-1920, 0, 0, 1080), 36) == (-1320, 948)  # монитор слева от основного


def test_hotkey_modifiers_for_register_hotkey() -> None:
    assert hotkey_modifiers("ctrl+alt+enter") == (MOD_CONTROL | MOD_ALT | MOD_NOREPEAT, 0x0D)
    assert hotkey_modifiers("shift+f9") == (MOD_SHIFT | MOD_NOREPEAT, 0x78)
    with pytest.raises(ActionError):
        hotkey_modifiers("ctrl+alt")
    with pytest.raises(ActionError):
        hotkey_modifiers("ctrl+a+b")


def test_tray_icon_is_drawn() -> None:
    image = tray_image("#7c6cff", 64)
    assert image.size == (64, 64)
    assert image.getpixel((32, 4))[:3] == (0x7C, 0x6C, 0xFF)  # круг акцентного цвета
    assert image.getpixel((32, 32)) == (255, 255, 255, 255)  # белая «волна» в центре
    assert image.getpixel((0, 0))[3] == 0  # углы прозрачные


def test_window_urls_carry_view_and_theme() -> None:
    from assistant.ui.windows import UI_DIST, ui_urls

    urls = ui_urls("light", "#22b8cf")
    assert urls["overlay"] == f"{UI_DIST / 'index.html'}?view=overlay&theme=light&accent=%2322b8cf"
    dev = ui_urls("dark", "#7c6cff", "http://localhost:5173")
    assert dev["chat"] == "http://localhost:5173/?view=chat&theme=dark&accent=%237c6cff"
    assert set(urls) == {"main", "overlay", "chat"}
