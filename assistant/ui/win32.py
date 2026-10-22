"""WinAPI для окон интерфейса: оверлей без фокуса и его движение, скругление, глобальная клавиша, WebView2.

Вне Windows функции ничего не делают (интерфейс там не запускается, а модуль импортируется
в тестах).
"""

from __future__ import annotations

import ctypes
import functools
import logging
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

from assistant.actions import winapi
from assistant.actions.input import parse_hotkey
from assistant.actions.registry import ActionError

log = logging.getLogger(__name__)

GWL_EXSTYLE = -20
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_APPWINDOW = 0x00040000
WS_EX_NOACTIVATE = 0x08000000
HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_FRAMECHANGED = 0x0020
SW_SHOWNOACTIVATE = 4
MONITOR_DEFAULTTONEAREST = 2
RGN_AND = 1
SPI_GETCLIENTAREAANIMATION = 0x1042
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, MOD_NOREPEAT = 0x1, 0x2, 0x4, 0x8, 0x4000
MB_ICONWARNING, MB_YESNO, IDYES = 0x30, 0x4, 6
FRAME_TIME = 1 / 60  # с: кадр, если не удалось дождаться его у Windows
# Модификаторы для RegisterHotKey по виртуальным кодам клавиш из parse_hotkey.
_MODIFIERS = {0x11: MOD_CONTROL, 0x12: MOD_ALT, 0x10: MOD_SHIFT, 0x5B: MOD_WIN, 0x5C: MOD_WIN}

# Evergreen WebView2 Runtime (learn.microsoft.com: «Detect if a WebView2 Runtime is already installed»).
WEBVIEW2_CLIENT = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WEBVIEW2_DOWNLOAD = "https://developer.microsoft.com/microsoft-edge/webview2/consumer/"


class MONITORINFO(ctypes.Structure):
    _fields_ = (
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    )


class _Win32:
    """Функции user32 и dwmapi с объявленными типами."""

    def __init__(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)  # type: ignore[attr-defined]
        gdi32 = ctypes.WinDLL("gdi32")  # type: ignore[attr-defined]
        dwmapi = ctypes.WinDLL("dwmapi")  # type: ignore[attr-defined]
        hwnd, uint, boolean, rgn = wintypes.HWND, wintypes.UINT, wintypes.BOOL, wintypes.HRGN

        def fn(dll: Any, name: str, restype: Any, *argtypes: Any) -> Any:
            func = getattr(dll, name)
            func.restype = restype
            func.argtypes = argtypes
            return func

        self.GetWindowLongPtrW = fn(user32, "GetWindowLongPtrW", ctypes.c_ssize_t, hwnd, ctypes.c_int)
        self.SetWindowLongPtrW = fn(
            user32, "SetWindowLongPtrW", ctypes.c_ssize_t, hwnd, ctypes.c_int, ctypes.c_ssize_t
        )
        self.SetWindowPos = fn(
            user32,
            "SetWindowPos",
            boolean,
            hwnd,
            hwnd,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            uint,
        )
        self.ShowWindowAsync = fn(user32, "ShowWindowAsync", boolean, hwnd, ctypes.c_int)
        self.SetWindowRgn = fn(user32, "SetWindowRgn", ctypes.c_int, hwnd, rgn, boolean)
        self.CreateRectRgn = fn(gdi32, "CreateRectRgn", rgn, *[ctypes.c_int] * 4)
        self.CreateRoundRectRgn = fn(gdi32, "CreateRoundRectRgn", rgn, *[ctypes.c_int] * 6)
        self.CombineRgn = fn(gdi32, "CombineRgn", ctypes.c_int, rgn, rgn, rgn, ctypes.c_int)
        self.DeleteObject = fn(gdi32, "DeleteObject", boolean, wintypes.HGDIOBJ)
        self.SystemParametersInfoW = fn(
            user32, "SystemParametersInfoW", boolean, uint, uint, ctypes.c_void_p, uint
        )
        self.GetDpiForWindow = fn(user32, "GetDpiForWindow", uint, hwnd)
        self.GetWindowRect = fn(user32, "GetWindowRect", boolean, hwnd, ctypes.POINTER(wintypes.RECT))
        self.MonitorFromPoint = fn(
            user32, "MonitorFromPoint", wintypes.HMONITOR, wintypes.POINT, wintypes.DWORD
        )
        self.GetMonitorInfoW = fn(
            user32, "GetMonitorInfoW", boolean, wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)
        )
        self.RegisterHotKey = fn(user32, "RegisterHotKey", boolean, hwnd, ctypes.c_int, uint, uint)
        self.UnregisterHotKey = fn(user32, "UnregisterHotKey", boolean, hwnd, ctypes.c_int)
        self.GetMessageW = fn(user32, "GetMessageW", boolean, ctypes.POINTER(wintypes.MSG), hwnd, uint, uint)
        self.PostThreadMessageW = fn(
            user32, "PostThreadMessageW", boolean, wintypes.DWORD, uint, wintypes.WPARAM, wintypes.LPARAM
        )
        self.MessageBoxW = fn(
            user32, "MessageBoxW", ctypes.c_int, hwnd, wintypes.LPCWSTR, wintypes.LPCWSTR, uint
        )
        self.DwmSetWindowAttribute = fn(
            dwmapi,
            "DwmSetWindowAttribute",
            ctypes.c_long,
            hwnd,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
        )
        self.DwmFlush = fn(dwmapi, "DwmFlush", ctypes.c_long)


@functools.cache
def _api() -> _Win32 | None:
    return _Win32() if winapi.IS_WINDOWS else None


# --- Окна --------------------------------------------------------------------------


def make_overlay(hwnd: int) -> None:
    """Окно не забирает фокус и не видно в панели задач и Alt+Tab.

    WinForms пересчитывает расширенные стили при некоторых изменениях формы, поэтому
    вызывается перед каждым показом.
    """
    api = _api()
    if api is None or not hwnd:
        return
    style = api.GetWindowLongPtrW(hwnd, GWL_EXSTYLE)
    api.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, (style | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW)
    api.SetWindowPos(
        hwnd, 0, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_NOACTIVATE
    )


def show_without_focus(hwnd: int, x: int, y: int) -> None:
    """Показывает окно поверх всех в точке (x, y) физических пикселей, не активируя его."""
    api = _api()
    if api is None or not hwnd:
        return
    api.SetWindowPos(hwnd, HWND_TOPMOST, x, y, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE)
    api.ShowWindowAsync(hwnd, SW_SHOWNOACTIVATE)


def move_window(hwnd: int, x: int, y: int) -> None:
    """Сдвигает окно в точку (x, y) физических пикселей, не меняя порядок окон и фокус."""
    api = _api()
    if api is not None and hwnd:
        api.SetWindowPos(hwnd, 0, x, y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)


def clip_window(hwnd: int, width: int, height: int | None, radius: int = 0) -> None:
    """Оставляет видной только верхнюю полосу окна высотой height физических пикселей.

    Верхние углы полосы скруглены радиусом radius, нижний край прямой. None — окно видно
    целиком: регион снят, углы снова скругляет Windows.
    """
    api = _api()
    if api is None or not hwnd:
        return
    if height is None:
        api.SetWindowRgn(hwnd, None, True)
        return
    clip = api.CreateRectRgn(0, 0, width, height)
    if radius and height:
        # Скруглённый прямоугольник выше края полосы: его нижние углы срезает прямоугольник полосы.
        rounded = api.CreateRoundRectRgn(0, 0, width + 1, height + 2 * radius + 1, 2 * radius, 2 * radius)
        api.CombineRgn(clip, clip, rounded, RGN_AND)
        api.DeleteObject(rounded)
    if not api.SetWindowRgn(hwnd, clip, True):
        api.DeleteObject(clip)  # принятым регионом владеет система, непринятый удаляем сами


def wait_frame() -> None:
    """Ждёт, пока Windows выведет следующий кадр экрана (DwmFlush): окно двигается в такт с ним.

    Если кадра ждать не пришлось (экран выключен, DwmFlush вернулся сразу) — обычная пауза в кадр.
    """
    api = _api()
    started = time.perf_counter()
    if api is not None and api.DwmFlush() == 0 and time.perf_counter() - started > 0.002:
        return
    time.sleep(FRAME_TIME)


def animations_enabled() -> bool:
    """Включены ли в Windows «Эффекты анимации» (Параметры → Специальные возможности → Визуальные эффекты)."""
    api = _api()
    if api is None:
        return False
    enabled = wintypes.BOOL(True)
    if not api.SystemParametersInfoW(SPI_GETCLIENTAREAANIMATION, 0, ctypes.byref(enabled), 0):
        return True
    return bool(enabled.value)


def rounds_corners() -> bool:
    """Windows скругляет углы окон: Windows 11 (сборка 22000 и новее)."""
    return winapi.IS_WINDOWS and sys.getwindowsversion().build >= 22000  # type: ignore[attr-defined]


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    """Положение и размер окна в физических пикселях: x, y, ширина, высота."""
    api = _api()
    rect = wintypes.RECT()
    if api is None or not hwnd or not api.GetWindowRect(hwnd, ctypes.byref(rect)):
        return 0, 0, 0, 0
    return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top


def window_size(hwnd: int) -> tuple[int, int]:
    """Размер окна в физических пикселях."""
    return window_rect(hwnd)[2:]


def set_bounds(hwnd: int, x: int, y: int, width: int, height: int) -> None:
    """Положение и размер окна в физических пикселях."""
    api = _api()
    if api is not None and hwnd:
        api.SetWindowPos(hwnd, 0, x, y, width, height, SWP_NOZORDER | SWP_NOACTIVATE)


def dpi_scale(hwnd: int) -> float:
    """Масштаб экрана окна: 1.0 при 100 %, 1.5 при 150 %."""
    api = _api()
    if api is None or not hwnd:
        return 1.0
    dpi = api.GetDpiForWindow(hwnd)
    return dpi / 96 if dpi else 1.0


def work_area() -> tuple[int, int, int, int]:
    """Рабочая область (без панели задач) монитора, на котором курсор: left, top, right, bottom."""
    api = _api()
    if api is None:
        return 0, 0, 1920, 1080
    x, y = winapi.cursor_position()
    monitor = api.MonitorFromPoint(wintypes.POINT(x, y), MONITOR_DEFAULTTONEAREST)
    info = MONITORINFO()
    info.cbSize = ctypes.sizeof(MONITORINFO)
    if not api.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return 0, 0, 1920, 1080
    work = info.rcWork
    return work.left, work.top, work.right, work.bottom


def bottom_center(size: tuple[int, int], area: tuple[int, int, int, int], margin: int) -> tuple[int, int]:
    """Точка левого верхнего угла окна по центру внизу рабочей области."""
    width, height = size
    left, _, right, bottom = area
    return left + (right - left - width) // 2, bottom - height - margin


def round_corners(hwnd: int) -> None:
    """Скруглённые углы окна в Windows 11 (в Windows 10 ничего не меняется)."""
    api = _api()
    if api is None or not hwnd:
        return
    preference = ctypes.c_int(DWMWCP_ROUND)
    api.DwmSetWindowAttribute(
        hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, ctypes.byref(preference), ctypes.sizeof(preference)
    )


def bring_to_front(hwnd: int) -> bool:
    """Делает окно активным (быстрый чат должен сразу принимать ввод)."""
    if not winapi.IS_WINDOWS or not hwnd:
        return False
    try:
        return winapi.activate_window(hwnd)
    except ActionError:
        return False


def ask_yes_no(title: str, text: str) -> bool:
    """Системный вопрос «Да / Нет» (когда интерфейса ещё нет)."""
    api = _api()
    if api is None:
        return False
    return api.MessageBoxW(None, text, title, MB_ICONWARNING | MB_YESNO) == IDYES


# --- WebView2 ----------------------------------------------------------------------


def webview2_version() -> str | None:
    """Версия WebView2 Runtime или None, если он не установлен.

    Ключ ищется у установки на компьютер (HKLM, 64-битная Windows — в WOW6432Node) и у
    установки для пользователя (HKCU); версия «0.0.0.0» означает, что установки нет.
    """
    if not winapi.IS_WINDOWS:
        return None
    import winreg

    places = (
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\{WEBVIEW2_CLIENT}"),
        (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\{WEBVIEW2_CLIENT}"),
        (winreg.HKEY_CURRENT_USER, rf"Software\{WEBVIEW2_CLIENT}"),
    )
    for root, path in places:
        try:
            with winreg.OpenKey(root, path) as key:
                version, _ = winreg.QueryValueEx(key, "pv")
        except OSError:
            continue
        if version and str(version) != "0.0.0.0":
            return str(version)
    return None


# --- Глобальная клавиша через RegisterHotKey ----------------------------------------


def hotkey_modifiers(combo: str) -> tuple[int, int]:
    """Модификаторы RegisterHotKey и виртуальный код клавиши для «ctrl+alt+enter».

    Raises:
        ActionError: неизвестная клавиша или нет основной клавиши.
    """
    modifiers, keys = 0, []
    for vk in parse_hotkey(combo):
        if vk in _MODIFIERS:
            modifiers |= _MODIFIERS[vk]
        else:
            keys.append(vk)
    if len(keys) != 1:
        raise ActionError(f"В сочетании «{combo}» должна быть одна основная клавиша")
    return modifiers | MOD_NOREPEAT, keys[0]


class SystemHotkey:
    """Сочетание клавиш через RegisterHotKey в своём потоке с очередью сообщений.

    В отличие от перехвата клавиатуры (библиотека keyboard), такое нажатие Windows
    считает вводом нашей программы — и позволяет ей вывести окно на передний план.

    Args:
        combo: сочетание вида «ctrl+alt+enter».
        callback: что сделать по нажатию (в потоке клавиши: быстро передать дальше).
    """

    _ID = 1

    def __init__(self, combo: str, callback: Callable[[], None]) -> None:
        self.combo = combo
        self._callback = callback
        self._thread_id = 0
        self._ready = threading.Event()
        self._error = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Регистрирует сочетание.

        Raises:
            ActionError: сочетание записано с ошибкой или уже занято другой программой.
        """
        modifiers, vk = hotkey_modifiers(self.combo)
        if _api() is None:
            return
        self._thread = threading.Thread(target=self._run, args=(modifiers, vk), name="hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(5)
        if self._error:
            raise ActionError(self._error)

    def _run(self, modifiers: int, vk: int) -> None:
        api = _api()
        assert api is not None
        self._thread_id = winapi.api().GetCurrentThreadId()
        if not api.RegisterHotKey(None, self._ID, modifiers, vk):
            self._error = f"Сочетание {self.combo} уже занято другой программой"
            self._ready.set()
            return
        self._ready.set()
        log.info("Горячая клавиша (система) %s", self.combo)
        message = wintypes.MSG()
        try:
            while api.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY:
                    try:
                        self._callback()
                    except Exception:
                        log.exception("Ошибка в обработчике %s", self.combo)
        finally:
            api.UnregisterHotKey(None, self._ID)

    def close(self) -> None:
        """Снимает сочетание."""
        api = _api()
        if api is not None and self._thread_id:
            api.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._thread_id = 0
