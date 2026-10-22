"""Окна интерфейса на настоящей Windows: pywebview + WebView2, оверлей без фокуса, быстрый чат.

Запускаются только при JARVIS_DESKTOP_TESTS=1 (так в CI): создают окна на рабочем столе.
Нужен собранный интерфейс (ui/dist) и Microsoft Edge WebView2 Runtime.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.executor import ActionExecutor
from assistant.actions.registry import ActionRegistry
from assistant.commands.loader import CommandStore
from assistant.journal import ActionJournal
from assistant.logging_setup import SecretMasker
from assistant.nlu.matcher import Matcher
from assistant.pipeline import Pipeline
from assistant.safety.confirm import StaticConfirmer
from assistant.safety.policy import SafetyPolicy
from assistant.ui.activity import Activity, UiSpeaker
from assistant.ui.api import Api
from assistant.ui.confirm import UiConfirmer
from assistant.ui.controller import WINDOWS, UiController
from assistant.ui.events import EventBus
from assistant.ui.settings import SettingsStore
from tests.conftest import NO_WINDOW, ROOT
from tests.ui_fakes import FakeAgent, FakeVolume

pytestmark = [
    pytest.mark.windows,
    pytest.mark.desktop,
    pytest.mark.skipif(sys.platform != "win32", reason="нужна Windows"),
    pytest.mark.skipif(
        os.environ.get("JARVIS_DESKTOP_TESTS") != "1", reason="включаются через JARVIS_DESKTOP_TESTS=1"
    ),
]


def wait(condition: Any, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def follow(hwnd: int, seconds: float) -> list[tuple[int, int]]:
    """Путь видимого окна: верх окна и тип его региона (0 — региона нет, 1 — пустой, 2–3 — обрезан)."""
    from assistant.actions import winapi
    from assistant.ui import win32

    user32 = ctypes.WinDLL("user32")
    user32.GetWindowRgn.restype = ctypes.c_int
    user32.GetWindowRgn.argtypes = (wintypes.HWND, wintypes.HRGN)
    api = win32._api()
    assert api is not None
    path: list[tuple[int, int]] = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if winapi.api().IsWindowVisible(hwnd):
            region = api.CreateRectRgn(0, 0, 0, 0)
            sample = (win32.window_rect(hwnd)[1], user32.GetWindowRgn(hwnd, region))
            api.DeleteObject(region)
            if not path or path[-1] != sample:
                path.append(sample)
        time.sleep(0.002)
    return path


def test_windows_load_interface_and_talk_to_python(
    tmp_path: Path, registry: ActionRegistry, store: CommandStore, matcher: Matcher
) -> None:
    import webview

    from assistant.actions import winapi
    from assistant.ui import win32
    from assistant.ui.windows import UI_DIST, Windows, ui_urls

    if not (UI_DIST / "index.html").exists():
        pytest.fail("интерфейс не собран: нет ui/dist/index.html")
    version = win32.webview2_version()
    assert version, "нет Microsoft Edge WebView2 Runtime"

    config_path = tmp_path / "config.yaml"
    shutil.copyfile(ROOT / "config.example.yaml", config_path)
    settings = SettingsStore(config_path, tmp_path / ".env")
    bus = EventBus()
    activity = Activity(bus)
    executor = ActionExecutor(
        registry,
        SafetyPolicy(),
        StaticConfirmer(True),
        ActionJournal(tmp_path / "j.jsonl", SecretMasker()),
        dry_run=True,
    )
    agent: Any = FakeAgent(settings.load().llm)
    pipeline = Pipeline(matcher, executor, UiSpeaker(activity), lambda: NO_WINDOW, agent=agent)
    controller = UiController(
        config=settings.load(), store=settings, pipeline=pipeline, executor=executor, commands=store,
        make_agent=FakeAgent, speech=None, volume=FakeVolume(), bus=bus, activity=activity,
        confirmer=UiConfirmer(bus, timeout=1), log_file=tmp_path / "a.log", journal_file=tmp_path / "j.jsonl",
        force_dry_run=True,
    )  # fmt: skip
    webview.settings["DRAG_REGION_DIRECT_TARGET_ONLY"] = True
    apis = {name: Api(controller, name) for name in WINDOWS}
    # Оверлей двигается, даже если на сервере CI анимации Windows выключены.
    windows = Windows(
        webview, ui_urls("dark", "#7c6cff"), apis, "dark", start_hidden=False, animate=lambda: True
    )
    controller.host = windows
    results: dict[str, Any] = {}

    def scenario() -> None:
        try:
            # Все три страницы загрузились и спросили состояние у Python.
            results["ready"] = wait(lambda: all(bus.attached(name) for name in WINDOWS), 60)
            main = windows._windows["main"]
            results["heading"] = main.evaluate_js("document.querySelector('h1')?.textContent")
            results["renderer"] = getattr(webview, "renderer", None)

            # Оверлей выезжает снизу из-за края рабочей области и не забирает фокус.
            before = winapi.api().GetForegroundWindow()
            controller.overlay.voice_state("listening")
            overlay = windows._handle("overlay")
            results["overlay_rise"] = follow(overlay, 0.6)
            results["overlay_visible"] = bool(winapi.api().IsWindowVisible(overlay))
            results["focus_kept"] = winapi.api().GetForegroundWindow() == before
            style = win32._api().GetWindowLongPtrW(overlay, win32.GWL_EXSTYLE)
            results["overlay_no_activate"] = bool(style & win32.WS_EX_NOACTIVATE)
            x, y, width, height = win32.window_rect(overlay)
            left, top, right, bottom = win32.work_area()
            results["overlay_bottom"] = (
                abs(left + (right - left - width) // 2 - x) <= 2 and bottom - (y + height) > 0
            )
            results["overlay_place"] = (y, bottom)
            controller.overlay.hide_now()
            results["overlay_fall"] = follow(overlay, 0.6)
            results["overlay_hidden"] = wait(lambda: not winapi.api().IsWindowVisible(overlay), 5)

            # Быстрый чат: окно показывается, команда из него проходит через js_api в конвейер.
            controller.toggle_chat()
            chat = windows._handle("chat")
            results["chat_visible"] = wait(lambda: bool(winapi.api().IsWindowVisible(chat)), 5)
            results["chat_foreground"] = winapi.api().GetForegroundWindow() == chat
            windows._windows["chat"].evaluate_js("window.pywebview.api.send_text('сделай громче')")
            results["chat_command"] = wait(lambda: bool(activity.recent), 10) and activity.recent[0]["phrase"]
            controller.toggle_chat()
            results["chat_hidden"] = wait(lambda: not winapi.api().IsWindowVisible(chat), 5)

            # Размер главного окна меняется уголком, «развернуть» — на рабочую область.
            controller.window_resize(1000, 700, 1)
            hwnd = windows._handle("main")
            results["resized"] = (
                wait(lambda: win32.window_rect(hwnd)[2] > 0, 2) and win32.window_rect(hwnd)[2:]
            )
            results["scale"] = win32.dpi_scale(hwnd)
            controller.window_action("main", "maximize")
            results["maximized"] = win32.window_rect(hwnd)[2:] == (right - left, bottom - top)
            controller.window_action("main", "maximize")
        except Exception as exc:
            results["error"] = repr(exc)
        finally:
            controller.quit()

    started = time.monotonic()
    finished = threading.Event()

    def watchdog() -> None:
        # Если окна так и не закрылись (WebView2 не запустился), тест не должен висеть до конца CI.
        if not finished.wait(180):
            print(f"Окна не закрылись за 180 с; результаты: {results}", flush=True)
            os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    webview.start(scenario, gui="edgechromium")
    finished.set()
    bus.close()
    print(f"WebView2 {version}; результаты: {results}; {time.monotonic() - started:.1f} с")
    assert "error" not in results, results
    assert results["ready"], "страницы окон не загрузились"
    assert results["renderer"] in (None, "edgechromium")
    assert results["heading"] == "Голос выключен"
    assert results["overlay_visible"] and results["overlay_no_activate"] and results["overlay_bottom"]
    assert results["focus_kept"], "оверлей забрал фокус"
    assert results["overlay_hidden"]
    # Появление: только вверх, сначала низ обрезан краем, на месте региона нет.
    place, edge = results["overlay_place"]
    rise = [y for y, _ in results["overlay_rise"]]
    assert len(set(rise)) >= 3 and rise == sorted(rise, reverse=True), results["overlay_rise"]
    assert rise[0] > place and rise[-1] == place, results["overlay_rise"]
    assert any(kind for _, kind in results["overlay_rise"]), "низ оверлея не обрезался"
    assert results["overlay_rise"][-1][1] == 0, "на месте оверлей должен быть виден целиком"
    # Уход: только вниз до края рабочей области, низ обрезан.
    fall = [y for y, _ in results["overlay_fall"]]
    assert len(set(fall)) >= 3 and fall == sorted(fall), results["overlay_fall"]
    assert place <= fall[0] and fall[-1] <= edge, results["overlay_fall"]
    assert any(kind for _, kind in results["overlay_fall"]), "уходя, оверлей не обрезался"
    assert results["chat_visible"] and results["chat_hidden"]
    assert results["chat_command"] == "сделай громче"
    width, height = results["resized"]
    assert (
        abs(width - round(1000 * results["scale"])) <= 2 and abs(height - round(700 * results["scale"])) <= 2
    )
    assert results["maximized"]


def test_system_hotkey_fires(tmp_path: Path) -> None:
    from assistant.actions import winapi
    from assistant.actions.input import parse_hotkey
    from assistant.ui.win32 import SystemHotkey

    pressed = threading.Event()
    hotkey = SystemHotkey("ctrl+alt+f11", pressed.set)
    hotkey.start()
    try:
        codes = parse_hotkey("ctrl+alt+f11")
        winapi.send_inputs([winapi.key_event(vk, up=False) for vk in codes])
        winapi.send_inputs([winapi.key_event(vk, up=True) for vk in reversed(codes)])
        assert pressed.wait(3), "RegisterHotKey не получил нажатие"
    finally:
        hotkey.close()
    # Сочетание снято — его можно зарегистрировать снова.
    again = SystemHotkey("ctrl+alt+f11", lambda: None)
    again.start()
    again.close()


def test_tray_icon_starts_and_stops() -> None:
    from assistant.ui.tray import Tray, TrayActions

    noop = lambda: None  # noqa: E731
    actions = TrayActions(noop, noop, noop, lambda: False, lambda: True, noop, noop, noop, noop)
    tray = Tray("Джарвис: проверка", "#7c6cff", actions)
    tray.start()
    time.sleep(1)
    tray.update("Джарвис: на паузе")
    tray.stop()
