"""Проверки на настоящей Windows (в CI — на раннере GitHub Actions).

Тесты с меткой desktop создают собственные окна, печатают в них через SendInput и
закрывают Блокнот. Чтобы не мешать работе за компьютером, они запускаются только
при JARVIS_DESKTOP_TESTS=1 (так сделано в CI). Если интерактивного рабочего стола нет
(окно не получает фокус), они пропускаются.
"""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from ctypes import wintypes
from typing import Any

import pytest

pytestmark = [
    pytest.mark.windows,
    pytest.mark.skipif(sys.platform != "win32", reason="нужна Windows"),
]

if sys.platform == "win32":
    from assistant.actions import winapi
    from assistant.actions.apps import AppIndex, scan_installed_apps
    from assistant.actions.input import press_hotkey, text_events, type_unicode
    from assistant.actions.registry import ActionError, ActionRegistry
    from assistant.actions.screen import Screen, WindowsDesktop
    from assistant.actions.volume import VolumeController
    from assistant.config import AppsConfig

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.CreateWindowExW.argtypes = (
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID,
    )  # fmt: skip
    user32.DestroyWindow.argtypes = (wintypes.HWND,)
    user32.GetWindowRect.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.RECT))
    user32.IsWindow.argtypes = (wintypes.HWND,)
    user32.IsZoomed.argtypes = (wintypes.HWND,)
    user32.IsIconic.argtypes = (wintypes.HWND,)
    user32.PeekMessageW.argtypes = (
        ctypes.POINTER(wintypes.MSG),
        wintypes.HWND,
        wintypes.UINT,
        wintypes.UINT,
        wintypes.UINT,
    )
    user32.TranslateMessage.argtypes = (ctypes.POINTER(wintypes.MSG),)
    user32.DispatchMessageW.argtypes = (ctypes.POINTER(wintypes.MSG),)

    LRESULT = ctypes.c_ssize_t
    WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    class WNDCLASSW(ctypes.Structure):
        _fields_ = (
            ("style", wintypes.UINT),
            ("lpfnWndProc", WNDPROC),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
        )

    user32.DefWindowProcW.restype = LRESULT
    user32.DefWindowProcW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
    user32.RegisterClassW.restype = wintypes.ATOM
    user32.RegisterClassW.argtypes = (ctypes.POINTER(WNDCLASSW),)
    user32.UnregisterClassW.argtypes = (wintypes.LPCWSTR, wintypes.HINSTANCE)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetModuleHandleW.restype = wintypes.HMODULE
    kernel32.GetModuleHandleW.argtypes = (wintypes.LPCWSTR,)
    user32.SendMessageW.restype = LRESULT
    user32.SendMessageW.argtypes = (wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

WS_OVERLAPPEDWINDOW = 0x00CF0000
WS_VISIBLE = 0x10000000
WS_CHILD = 0x40000000
ES_AUTOHSCROLL = 0x0080
BS_AUTOCHECKBOX = 0x0003
BM_GETCHECK = 0x00F0
WM_COMMAND = 0x0111
PM_REMOVE = 0x0001


def pump(seconds: float = 0.4) -> None:
    """Обрабатывает сообщения окон этого потока (ввод с клавиатуры приходит сюда)."""
    msg = wintypes.MSG()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        time.sleep(0.01)


def create_window(class_name: str, title: str, style: int = 0) -> int:
    hwnd = user32.CreateWindowExW(
        0, class_name, title, WS_OVERLAPPEDWINDOW | WS_VISIBLE | style, 200, 200, 640, 160,
        None, None, None, None,
    )  # fmt: skip
    assert hwnd, f"CreateWindowExW: ошибка {ctypes.get_last_error()}"
    pump(0.2)
    return hwnd


def activate_or_skip(hwnd: int) -> None:
    if not winapi.activate_window(hwnd):
        pytest.skip("Нет интерактивного рабочего стола: окно не получает фокус")
    pump(0.2)


def wait_for_keyboard_hook(timeout: float = 5.0) -> None:
    """Ждёт, пока библиотека keyboard поставит перехват клавиатуры.

    add_hotkey возвращается сразу, а перехват (SetWindowsHookEx) ставится в отдельном
    потоке: нажатия, отправленные раньше, до библиотеки не доходят. Поэтому нажимаем
    F24 (такой клавиши на клавиатурах нет), пока библиотека её не увидит.
    """
    import keyboard

    seen = threading.Event()
    remove = keyboard.hook(lambda event: seen.set())
    try:
        deadline = time.monotonic() + timeout
        while not seen.is_set() and time.monotonic() < deadline:
            press_hotkey("f24")
            seen.wait(0.1)
    finally:
        remove()
    assert seen.is_set(), "перехват клавиатуры библиотекой keyboard не включился"


@pytest.fixture
def edit_window() -> Iterator[int]:
    """Окно-поле ввода (стандартный класс EDIT), активное и пустое."""
    hwnd = create_window("EDIT", "", ES_AUTOHSCROLL)
    try:
        activate_or_skip(hwnd)
        yield hwnd
    finally:
        user32.DestroyWindow(hwnd)
        pump(0.1)


def test_input_struct_has_native_size() -> None:
    assert ctypes.sizeof(winapi.INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)


def test_window_queries_work() -> None:
    windows = winapi.list_windows()
    assert isinstance(windows, list)
    assert all(w.title for w in windows)
    assert isinstance(winapi.foreground_window(), winapi.WindowInfo)
    assert winapi.own_process_tree()


def test_text_events_use_unicode_and_surrogates() -> None:
    events = text_events("Ё😀\n")
    assert len(events) == 2 + 4 + 2  # Ё — одна единица UTF-16, эмодзи — две, Enter
    assert events[0].ki.dwFlags & winapi.KEYEVENTF_UNICODE and events[0].ki.wScan == ord("Ё")
    assert events[-1].ki.wVk == 0x0D


def test_key_events_have_scan_codes_and_extended_flag() -> None:
    letter = winapi.key_event(ord("A"), up=False)
    assert letter.ki.wScan != 0 and not letter.ki.dwFlags & winapi.KEYEVENTF_EXTENDEDKEY
    arrow = winapi.key_event(0x25, up=True)
    assert arrow.ki.dwFlags & winapi.KEYEVENTF_EXTENDEDKEY and arrow.ki.dwFlags & winapi.KEYEVENTF_KEYUP


def test_powershell_output_keeps_cyrillic(registry: ActionRegistry) -> None:
    action = registry.get("run_shell")
    assert action.run(action.parse_args({"command": "Write-Output 'Привет, мир'"})) == "Привет, мир"


def test_known_folders_exist() -> None:
    from assistant.actions.files import known_folder

    for key in ("desktop", "documents", "downloads"):
        assert known_folder(key).is_dir(), key


def test_installed_apps_are_found() -> None:
    entries = scan_installed_apps()
    assert entries, "в меню «Пуск» не нашлось ни одного приложения"
    index = AppIndex(AppsConfig(), scanner=lambda: entries)
    index.refresh()
    found = index.find("powershell")
    assert found is not None and "powershell" in found.name.lower()


def test_volume_controller_round_trip() -> None:
    controller = VolumeController()
    try:
        before = controller.get()
    except ActionError:
        pytest.skip("Нет устройства вывода звука")
    try:
        assert 0 <= before <= 100
        assert controller.set(before) == before
        assert controller.get() == before
    finally:
        controller.set(before)


@pytest.mark.desktop
@pytest.mark.skipif(
    os.environ.get("JARVIS_DESKTOP_TESTS") != "1", reason="включаются через JARVIS_DESKTOP_TESTS=1"
)
class TestDesktop:
    def test_types_cyrillic_and_punctuation(self, edit_window: int) -> None:
        type_unicode("Привет, мир! Ёж 123 — test")
        pump()
        assert winapi.window_title(edit_window) == "Привет, мир! Ёж 123 — test"

    def test_types_emoji(self, edit_window: int) -> None:
        type_unicode("ok 😀")
        pump()
        assert winapi.window_title(edit_window) == "ok 😀"

    def test_hotkeys_and_extended_keys(self, edit_window: int) -> None:
        type_unicode("ac")
        press_hotkey("left")
        type_unicode("b")
        press_hotkey("home")
        type_unicode("_")
        pump()
        assert winapi.window_title(edit_window) == "_abc"
        press_hotkey("backspace")
        press_hotkey("end")
        type_unicode("!")
        pump()
        assert winapi.window_title(edit_window) == "abc!"

    def test_window_actions(self, edit_window: int, registry: ActionRegistry) -> None:
        action = registry.get("window_action")
        action.run(action.parse_args({"action": "maximize"}))
        pump()
        assert user32.IsZoomed(edit_window)
        action.run(action.parse_args({"action": "restore"}))
        pump()
        assert not user32.IsZoomed(edit_window)
        with pytest.raises(ActionError, match="самого ассистента"):
            action.run(action.parse_args({"action": "close"}))
        action.run(action.parse_args({"action": "minimize"}))
        pump()
        assert user32.IsIconic(edit_window)

    def test_focus_window_by_title(self, registry: ActionRegistry) -> None:
        alpha = create_window("STATIC", "JarvisTestAlpha")
        bravo = create_window("STATIC", "JarvisTestBravo")
        try:
            activate_or_skip(alpha)
            action = registry.get("focus_window")
            assert "JarvisTestBravo" in action.run(action.parse_args({"title": "JarvisTestBravo"}))
            pump()
            assert winapi.api().GetForegroundWindow() == bravo
        finally:
            user32.DestroyWindow(alpha)
            user32.DestroyWindow(bravo)
            pump(0.1)

    def test_global_hotkeys(self) -> None:
        from assistant.actions.input import parse_hotkey
        from assistant.hotkeys import HotkeyManager

        manager = HotkeyManager()
        pressed, down, up = threading.Event(), threading.Event(), threading.Event()
        try:
            manager.add_press("ctrl+alt+f9", pressed.set)
            manager.add_hold("ctrl+alt+f10", down.set, up.set)
            wait_for_keyboard_hook()
            # Клавиши держим, пока библиотека не увидит нажатие. Её поток обработки сверяет
            # сочетание с клавишами, нажатыми в момент обработки (keyboard/__init__.py,
            # pre_process_event), а SendInput отпускает их раньше, чем он успевает. Человек
            # держит клавиши десятки миллисекунд, поэтому в программе этой гонки нет.
            codes = parse_hotkey("ctrl+alt+f9")
            winapi.send_inputs([winapi.key_event(vk, up=False) for vk in codes])
            try:
                assert pressed.wait(3), "нажатие ctrl+alt+f9 не дошло"
            finally:
                winapi.send_inputs([winapi.key_event(vk, up=True) for vk in reversed(codes)])
            codes = parse_hotkey("ctrl+alt+f10")
            winapi.send_inputs([winapi.key_event(vk, up=False) for vk in codes])
            assert down.wait(3), "удержание ctrl+alt+f10 не началось"
            assert not up.is_set()
            winapi.send_inputs([winapi.key_event(vk, up=True) for vk in reversed(codes)])
            assert up.wait(3), "отпускание ctrl+alt+f10 не дошло"
        finally:
            manager.close()

    def test_clipboard_round_trip(self, registry: ActionRegistry) -> None:
        put, get = registry.get("clipboard_set"), registry.get("clipboard_get")
        put.run(put.parse_args({"text": "Проверка буфера 😀"}))
        assert get.run(get.parse_args({})) == "Проверка буфера 😀"

    def test_mouse_click_moves_cursor(self, edit_window: int, registry: ActionRegistry) -> None:
        rect = wintypes.RECT()
        user32.GetWindowRect(edit_window, ctypes.byref(rect))
        x, y = (rect.left + rect.right) // 2, (rect.top + rect.bottom) // 2
        action = registry.get("mouse_click")
        action.run(action.parse_args({"x": x, "y": y}))
        pump(0.1)
        assert winapi.cursor_position() == (x, y)

    def test_close_app_closes_notepad(self, registry: ActionRegistry) -> None:
        process = subprocess.Popen(["notepad.exe"])
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if any(w.process.lower() == "notepad.exe" for w in winapi.list_windows()):
                    break
                time.sleep(0.2)
            else:
                pytest.skip("Окно Блокнота не появилось")
            action = registry.get("close_app")
            action.run(action.parse_args({"name": "notepad"}))
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and any(
                w.process.lower() == "notepad.exe" for w in winapi.list_windows()
            ):
                time.sleep(0.2)
            assert not any(w.process.lower() == "notepad.exe" for w in winapi.list_windows())
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


class Form:
    """Окно с настоящими кнопками Windows; нажатия (WM_COMMAND от кнопок) записываются.

    У каждого окна свой класс: процедура окна — метод этого объекта, и класс, оставшийся от
    прошлого теста, вызывал бы процедуру уже удалённого объекта.
    """

    SAVE, CANCEL, REMEMBER = 101, 102, 103

    def __init__(self) -> None:
        self.clicked: list[int] = []
        # Ссылка на процедуру окна держится, пока окно живо: иначе её соберёт сборщик мусора.
        self._proc = WNDPROC(self._window_proc)
        self.instance = kernel32.GetModuleHandleW(None)
        self.class_name = f"JarvisScreenTest{id(self)}"
        window_class = WNDCLASSW(
            lpfnWndProc=self._proc, hInstance=self.instance, lpszClassName=self.class_name
        )
        window_class.hbrBackground = 6  # COLOR_WINDOW + 1
        assert user32.RegisterClassW(ctypes.byref(window_class)), f"RegisterClassW: {ctypes.get_last_error()}"
        self.hwnd = user32.CreateWindowExW(
            0, self.class_name, "Проверка экрана", WS_OVERLAPPEDWINDOW | WS_VISIBLE, 200, 200, 640, 260,
            None, None, self.instance, None,
        )  # fmt: skip
        assert self.hwnd, f"CreateWindowExW: ошибка {ctypes.get_last_error()}"
        self.save = self._control("BUTTON", "Сохранить", 0, 20, 20, self.SAVE)
        self.cancel = self._control("BUTTON", "Отмена", 0, 160, 20, self.CANCEL)
        self.remember = self._control("BUTTON", "Запомнить меня", BS_AUTOCHECKBOX, 20, 80, self.REMEMBER)
        pump(0.2)

    def _control(self, kind: str, text: str, style: int, x: int, y: int, control_id: int) -> int:
        hwnd = user32.CreateWindowExW(
            0, kind, text, WS_CHILD | WS_VISIBLE | style, x, y, 130, 32, self.hwnd, control_id, self.instance,
            None,
        )  # fmt: skip
        assert hwnd, f"CreateWindowExW({text}): ошибка {ctypes.get_last_error()}"
        return hwnd

    def _window_proc(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == WM_COMMAND and lparam:
            self.clicked.append(wparam & 0xFFFF)
        return user32.DefWindowProcW(hwnd, message, wparam, lparam)

    def checked(self) -> bool:
        return user32.SendMessageW(self.remember, BM_GETCHECK, 0, 0) == 1

    def close(self) -> None:
        user32.DestroyWindow(self.hwnd)
        pump(0.1)
        user32.UnregisterClassW(self.class_name, self.instance)


def while_pumping[T](work: Callable[[], T], timeout: float = 30) -> T:
    """Выполняет work в другом потоке, а этот обрабатывает сообщения окон.

    UI Automation спрашивает окна теста сообщениями: если поток окна ждёт, ответ не придёт.
    """
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = work()
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        pump(0.05)
    assert not thread.is_alive(), "действие не закончилось"
    if "error" in box:
        raise box["error"]
    return box["value"]


@pytest.fixture
def form() -> Iterator[Form]:
    window = Form()
    try:
        activate_or_skip(window.hwnd)
        yield window
    finally:
        window.close()


@pytest.mark.skipif(
    os.environ.get("JARVIS_DESKTOP_TESTS") != "1", reason="включаются через JARVIS_DESKTOP_TESTS=1"
)
class TestScreen:
    """describe_screen, click_ui_element и click_screen на окне с настоящими кнопками."""

    def test_describe_lists_controls_and_makes_shot(self, form: Form) -> None:
        screen = Screen(WindowsDesktop(skip_own=False))
        text = while_pumping(screen.describe)
        assert text.startswith("Окно «Проверка экрана»"), text
        assert re.search(r"- Button «Сохранить» \(\d+, \d+\) \d+×\d+", text), text
        assert "CheckBox «Запомнить меня»" in text
        image = screen.take_image()
        assert image is not None and image[:2] == b"\xff\xd8"

    def test_click_element_presses_button_and_toggles_checkbox(self, form: Form) -> None:
        screen = Screen(WindowsDesktop(skip_own=False))
        assert while_pumping(lambda: screen.click_element("сохранить")) == "Нажал кнопку «Сохранить»"
        pump(0.3)
        assert form.clicked == [Form.SAVE]
        result = while_pumping(lambda: screen.click_element("флажок запомнить"))
        assert result == "Переключил флажок «Запомнить меня»"
        pump(0.3)
        assert form.checked()
        with pytest.raises(ActionError, match="Не нашёл «Печать»"):
            while_pumping(lambda: screen.click_element("Печать"))

    def test_click_screen_hits_button_from_description(self, form: Form) -> None:
        # Координаты из описания (на снимке) пересчитываются обратно в экранные.
        screen = Screen(WindowsDesktop(skip_own=False))
        text = while_pumping(screen.describe)
        found = re.search(r"- Button «Отмена» \((\d+), (\d+)\)", text)
        assert found, text
        while_pumping(lambda: screen.click_point(int(found[1]), int(found[2])))
        pump(0.3)
        assert form.clicked == [Form.CANCEL]
