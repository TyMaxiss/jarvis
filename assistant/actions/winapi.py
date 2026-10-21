"""Тонкие обёртки над WinAPI через ctypes: окна, фокус, синтетический ввод.

Модуль импортируется и на других ОС (для тестов и разработки), но функции WinAPI
доступны только в Windows: при вызове вне Windows поднимается ActionError.
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import sys
import time
from collections.abc import Sequence
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

import psutil

from assistant.actions.registry import ActionError

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

ULONG_PTR = ctypes.c_size_t

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
MAPVK_VK_TO_VSC = 0

SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_RESTORE = 9
WM_CLOSE = 0x0010
GW_OWNER = 4
GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
DWMWA_CLOAKED = 14
VK_MENU = 0x12

# Окна рабочего стола и панели задач: их нельзя «закрывать» и «сворачивать».
SHELL_WINDOW_CLASSES = frozenset({"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"})


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = (("type", wintypes.DWORD), ("u", _INPUTUNION))


class _Api:
    """Набор функций WinAPI с объявленными типами аргументов."""

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        dwmapi = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined]
        hwnd, uint, dword, boolean = wintypes.HWND, wintypes.UINT, wintypes.DWORD, wintypes.BOOL

        self.WNDENUMPROC = ctypes.WINFUNCTYPE(boolean, hwnd, wintypes.LPARAM)  # type: ignore[attr-defined]

        def fn(dll: Any, name: str, restype: Any, *argtypes: Any) -> Any:
            func = getattr(dll, name)
            func.restype = restype
            func.argtypes = argtypes
            return func

        self.SendInput = fn(user32, "SendInput", uint, uint, ctypes.POINTER(INPUT), ctypes.c_int)
        self.MapVirtualKeyW = fn(user32, "MapVirtualKeyW", uint, uint, uint)
        self.GetAsyncKeyState = fn(user32, "GetAsyncKeyState", ctypes.c_short, ctypes.c_int)
        self.GetForegroundWindow = fn(user32, "GetForegroundWindow", hwnd)
        self.SetForegroundWindow = fn(user32, "SetForegroundWindow", boolean, hwnd)
        self.BringWindowToTop = fn(user32, "BringWindowToTop", boolean, hwnd)
        self.GetWindowTextLengthW = fn(user32, "GetWindowTextLengthW", ctypes.c_int, hwnd)
        self.GetWindowTextW = fn(user32, "GetWindowTextW", ctypes.c_int, hwnd, wintypes.LPWSTR, ctypes.c_int)
        self.GetClassNameW = fn(user32, "GetClassNameW", ctypes.c_int, hwnd, wintypes.LPWSTR, ctypes.c_int)
        self.GetWindowThreadProcessId = fn(
            user32, "GetWindowThreadProcessId", dword, hwnd, ctypes.POINTER(dword)
        )
        self.EnumWindows = fn(user32, "EnumWindows", boolean, self.WNDENUMPROC, wintypes.LPARAM)
        self.IsWindowVisible = fn(user32, "IsWindowVisible", boolean, hwnd)
        self.IsIconic = fn(user32, "IsIconic", boolean, hwnd)
        self.GetWindow = fn(user32, "GetWindow", hwnd, hwnd, uint)
        self.GetWindowLongW = fn(user32, "GetWindowLongW", wintypes.LONG, hwnd, ctypes.c_int)
        self.ShowWindow = fn(user32, "ShowWindow", boolean, hwnd, ctypes.c_int)
        self.PostMessageW = fn(user32, "PostMessageW", boolean, hwnd, uint, wintypes.WPARAM, wintypes.LPARAM)
        self.AttachThreadInput = fn(user32, "AttachThreadInput", boolean, dword, dword, boolean)
        self.GetCurrentThreadId = fn(kernel32, "GetCurrentThreadId", dword)
        self.DwmGetWindowAttribute = fn(
            dwmapi, "DwmGetWindowAttribute", ctypes.c_long, hwnd, dword, ctypes.c_void_p, dword
        )


@functools.cache
def api() -> _Api:
    """Возвращает функции WinAPI (загружаются один раз).

    Raises:
        ActionError: программа запущена не в Windows.
    """
    if not IS_WINDOWS:
        raise ActionError("Это действие работает только в Windows")
    return _Api()


# --- Окна ------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowInfo:
    """Сведения об окне верхнего уровня."""

    hwnd: int
    title: str
    class_name: str
    pid: int
    process: str

    @property
    def process_stem(self) -> str:
        """Имя процесса без .exe: Telegram.exe → Telegram."""
        return self.process.removesuffix(".exe").removesuffix(".EXE")


EMPTY_WINDOW = WindowInfo(0, "", "", 0, "")


def window_title(hwnd: int) -> str:
    """Заголовок окна."""
    length = api().GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    api().GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def window_class(hwnd: int) -> str:
    """Имя класса окна."""
    buffer = ctypes.create_unicode_buffer(256)
    api().GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def window_pid(hwnd: int) -> int:
    """PID процесса, которому принадлежит окно."""
    pid = wintypes.DWORD()
    api().GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def process_name(pid: int) -> str:
    """Имя исполняемого файла процесса или пустая строка, если узнать не удалось."""
    try:
        return psutil.Process(pid).name()
    except (psutil.Error, OSError):
        return ""


def window_info(hwnd: int) -> WindowInfo:
    """Собирает сведения об окне."""
    if not hwnd:
        return EMPTY_WINDOW
    pid = window_pid(hwnd)
    return WindowInfo(hwnd, window_title(hwnd), window_class(hwnd), pid, process_name(pid))


def foreground_window() -> WindowInfo:
    """Активное окно (или пустое описание, если его нет)."""
    return window_info(api().GetForegroundWindow() or 0)


def _is_cloaked(hwnd: int) -> bool:
    """Окно «спрятано» DWM (например, приостановленное UWP-приложение)."""
    cloaked = wintypes.DWORD()
    result = api().DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    return result == 0 and cloaked.value != 0


def list_windows() -> list[WindowInfo]:
    """Видимые окна верхнего уровня — примерно то, что показывает Alt+Tab."""
    handles: list[int] = []

    def collect(hwnd: int, _lparam: int) -> bool:
        handles.append(hwnd)
        return True

    api().EnumWindows(api().WNDENUMPROC(collect), 0)
    windows = []
    for hwnd in handles:
        if not api().IsWindowVisible(hwnd) or api().GetWindow(hwnd, GW_OWNER):
            continue
        ex_style = api().GetWindowLongW(hwnd, GWL_EXSTYLE)
        if ex_style & (WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE) or _is_cloaked(hwnd):
            continue
        info = window_info(hwnd)
        if info.title and info.class_name not in SHELL_WINDOW_CLASSES:
            windows.append(info)
    return windows


def own_process_tree() -> set[int]:
    """PID нашего процесса и всех его родителей (консоль, терминал, uv)."""
    me = psutil.Process()
    pids = {me.pid}
    with contextlib.suppress(psutil.Error):
        pids.update(parent.pid for parent in me.parents())
    return pids


def activate_window(hwnd: int) -> bool:
    """Выводит окно на передний план и делает его активным.

    Windows не даёт фоновому процессу перехватывать фокус, поэтому используются
    два известных обхода: временное присоединение ввода к потоку активного окна
    и синтетическое нажатие Alt.

    Returns:
        True, если окно стало активным.
    """
    user = api()
    if user.IsIconic(hwnd):
        user.ShowWindow(hwnd, SW_RESTORE)
    if user.SetForegroundWindow(hwnd) and user.GetForegroundWindow() == hwnd:
        return True

    foreground = user.GetForegroundWindow()
    fg_thread = user.GetWindowThreadProcessId(foreground, None) if foreground else 0
    our_thread = user.GetCurrentThreadId()
    attached = bool(fg_thread and fg_thread != our_thread) and bool(
        user.AttachThreadInput(our_thread, fg_thread, True)
    )
    try:
        user.BringWindowToTop(hwnd)
        user.SetForegroundWindow(hwnd)
    finally:
        if attached:
            user.AttachThreadInput(our_thread, fg_thread, False)
    if user.GetForegroundWindow() == hwnd:
        return True

    try:
        send_inputs([key_event(VK_MENU, up=False), key_event(VK_MENU, up=True)])
    except ActionError:
        log.debug("Не удалось отправить Alt для смены фокуса")
    user.SetForegroundWindow(hwnd)
    time.sleep(0.05)
    return user.GetForegroundWindow() == hwnd


def wait_foreground_change(before: int, timeout: float) -> bool:
    """Ждёт, пока активным станет другое окно (например, после запуска программы)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = api().GetForegroundWindow() or 0
        if current and current != before:
            return True
        time.sleep(0.1)
    return False


# --- Синтетический ввод ------------------------------------------------------------

# Клавиши, для которых нужен флаг KEYEVENTF_EXTENDEDKEY.
EXTENDED_KEYS = frozenset(
    {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B, 0x5C, 0x5D,
     0x6F, 0x90, 0xA3, 0xA5, *range(0xA6, 0xB8)}
)  # fmt: skip


def key_event(vk: int, *, up: bool) -> INPUT:
    """Событие нажатия или отпускания клавиши по виртуальному коду."""
    flags = KEYEVENTF_KEYUP if up else 0
    if vk in EXTENDED_KEYS:
        flags |= KEYEVENTF_EXTENDEDKEY
    scan = api().MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) & 0xFF
    event = INPUT(type=INPUT_KEYBOARD)
    event.ki = KEYBDINPUT(wVk=vk, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=0)
    return event


def unicode_event(code_unit: int, *, up: bool) -> INPUT:
    """Событие ввода одного UTF-16-символа (работает для кириллицы и эмодзи)."""
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    event = INPUT(type=INPUT_KEYBOARD)
    event.ki = KEYBDINPUT(wVk=0, wScan=code_unit, dwFlags=flags, time=0, dwExtraInfo=0)
    return event


def send_inputs(events: Sequence[INPUT]) -> int:
    """Отправляет события ввода одним вызовом SendInput.

    Raises:
        ActionError: система не приняла ввод (например, его блокирует UIPI —
            активно окно, запущенное от администратора).
    """
    if not events:
        return 0
    array = (INPUT * len(events))(*events)
    sent = api().SendInput(len(events), array, ctypes.sizeof(INPUT))
    if sent != len(events):
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        log.warning("SendInput отправил %d из %d событий, ошибка %d", sent, len(events), error)
        raise ActionError("Windows не принял ввод. Возможно, активное окно запущено от администратора")
    return sent


MODIFIER_KEYS = (0x10, 0x11, 0x12, 0x5B, 0x5C)  # Shift, Ctrl, Alt, левый и правый Win


def wait_modifiers_released(timeout: float = 1.5) -> None:
    """Ждёт, пока пользователь отпустит Ctrl/Alt/Shift/Win.

    Иначе физически зажатые модификаторы (например, после горячей клавиши ассистента)
    смешаются с синтетическим вводом.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not any(api().GetAsyncKeyState(vk) & 0x8000 for vk in MODIFIER_KEYS):
            return
        time.sleep(0.03)
