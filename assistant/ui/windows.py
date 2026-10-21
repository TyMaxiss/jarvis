"""Окна pywebview: главное, оверлей без фокуса и быстрый чат.

Оверлей показывается через WinAPI (без активации), остальные окна — средствами pywebview.
Окна без рамки: перетаскивание — за элементы с классом pywebview-drag-region, размер —
уголком в интерфейсе, «развернуть» — на рабочую область монитора.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

log = logging.getLogger(__name__)

UI_DIST = Path(__file__).resolve().parents[2] / "ui" / "dist"
MAIN_SIZE, MAIN_MIN = (1100, 720), (880, 600)
OVERLAY_SIZE = (480, 64)
CHAT_SIZE = (620, 400)
OVERLAY_MARGIN = 24  # от панели задач, логические пиксели
# Фон окна до загрузки страницы — как у интерфейса, чтобы не мигало белым.
BACKGROUNDS = {"dark": ("#0F1013", "#15161A"), "light": ("#F6F6F8", "#FFFFFF")}


def ui_urls(theme: str, accent: str, dev_url: str | None = None) -> dict[str, str]:
    """Адреса трёх окон: собранный интерфейс (ui/dist) или сервер разработки Vite.

    Окно выбирается параметром ?view=: «#/overlay» pywebview в Windows портит при разборе пути.
    Тема передаётся тоже, чтобы страница применила её до первой отрисовки.
    """
    base = dev_url.rstrip("/") + "/" if dev_url else str(UI_DIST / "index.html")
    style = urlencode({"theme": theme, "accent": accent})
    return {name: f"{base}?view={name}&{style}" for name in ("main", "overlay", "chat")}


class Windows:
    """Окна pywebview: главное, оверлей без фокуса и быстрый чат (WindowHost).

    Args:
        webview: модуль webview.
        urls: адреса окон.
        apis: объекты js_api для окон.
        theme: тема (цвет фона до загрузки страницы).
        start_hidden: главное окно при запуске не показывать (сразу в трей).
    """

    def __init__(
        self, webview: Any, urls: dict[str, str], apis: dict[str, Any], theme: str, start_hidden: bool
    ) -> None:
        from assistant.ui import win32

        self._win32 = win32
        self._quitting = False
        self._handles: dict[str, int] = {}
        self._saved_bounds: tuple[int, int, int, int] | None = None
        self._visible = {"main": not start_hidden, "overlay": False, "chat": False}
        background, surface = BACKGROUNDS.get(theme, BACKGROUNDS["dark"])
        common = {"frameless": True, "easy_drag": False}
        self._windows = {
            "main": webview.create_window(
                "Джарвис", urls["main"], js_api=apis["main"], width=MAIN_SIZE[0], height=MAIN_SIZE[1],
                min_size=MAIN_MIN, hidden=start_hidden, background_color=background, **common,
            ),
            "overlay": webview.create_window(
                "Джарвис — оверлей", urls["overlay"], js_api=apis["overlay"], width=OVERLAY_SIZE[0],
                height=OVERLAY_SIZE[1], min_size=OVERLAY_SIZE, resizable=False, shadow=False, on_top=True,
                focus=False, hidden=True, background_color=surface, **common,
            ),
            "chat": webview.create_window(
                "Джарвис — быстрый чат", urls["chat"], js_api=apis["chat"], width=CHAT_SIZE[0],
                height=CHAT_SIZE[1], min_size=CHAT_SIZE, resizable=False, on_top=True, hidden=True,
                background_color=surface, **common,
            ),
        }  # fmt: skip
        for name, window in self._windows.items():
            window.events.before_show += self._on_before_show(name)
            window.events.closing += self._on_closing(name)

    def _on_before_show(self, name: str) -> Callable[..., None]:
        def handler(window: Any) -> None:
            # Поток окон: здесь безопасно взять дескриптор окна.
            try:
                self._handles[name] = int(window.native.Handle.ToInt64())
            except Exception:
                log.debug("Нет дескриптора окна %s", name, exc_info=True)
                return
            self._win32.round_corners(self._handles[name])
            if name == "overlay":
                self._win32.make_overlay(self._handles[name])

        return handler

    def _on_closing(self, name: str) -> Callable[[], bool | None]:
        def handler() -> bool | None:
            if self._quitting:
                return None
            # Закрытие (Alt+F4, панель задач) прячет окно; главное — в трей.
            self._windows[name].hide()
            self._visible[name] = False
            return False

        return handler

    def watch_session_end(self) -> None:
        """Выход из Windows закрывает окна по-настоящему, а не прячет их (иначе выход ждёт)."""
        try:
            from Microsoft.Win32 import SystemEvents  # type: ignore[import-not-found]

            def on_ending(sender: Any, args: Any) -> None:
                self._quitting = True

            SystemEvents.SessionEnding += on_ending
        except Exception:
            log.debug("Не удалось подписаться на завершение сеанса Windows", exc_info=True)

    def _handle(self, name: str) -> int:
        if name not in self._handles:
            try:
                self._handles[name] = int(self._windows[name].native.Handle.ToInt64())
            except Exception:
                return 0
        return self._handles[name]

    def show(self, name: str) -> None:
        window, hwnd = self._windows[name], self._handle(name)
        self._visible[name] = True
        if name == "overlay":
            # Без фокуса: пользователь продолжает печатать в своём окне.
            self._win32.make_overlay(hwnd)
            area = self._win32.work_area()
            margin = round(OVERLAY_MARGIN * self._win32.dpi_scale(hwnd))
            x, y = self._win32.bottom_center(self._win32.window_size(hwnd), area, margin)
            self._win32.show_without_focus(hwnd, x, y)
            return
        window.show()
        if name == "main":
            window.restore()
        self._win32.bring_to_front(hwnd)

    def hide(self, name: str) -> None:
        self._visible[name] = False
        self._windows[name].hide()

    def visible(self, name: str) -> bool:
        return self._visible[name]

    def minimize(self) -> None:
        self._windows["main"].minimize()

    def toggle_maximize(self) -> None:
        """Развернуть на рабочую область монитора и обратно (свой вариант для окна без рамки)."""
        hwnd = self._handle("main")
        if self._saved_bounds is not None:
            self._win32.set_bounds(hwnd, *self._saved_bounds)
            self._saved_bounds = None
            return
        self._saved_bounds = self._win32.window_rect(hwnd)
        left, top, right, bottom = self._win32.work_area()
        self._win32.set_bounds(hwnd, left, top, right - left, bottom - top)

    def resize_main(self, width: int, height: int) -> None:
        self._saved_bounds = None
        self._windows["main"].resize(width, height)

    def sender(self, name: str) -> Callable[[str], None]:
        return self._windows[name].run_js

    def quit(self) -> None:
        self._quitting = True
        for name in ("chat", "overlay", "main"):
            try:
                self._windows[name].destroy()
            except Exception:
                log.debug("Окно %s уже закрыто", name, exc_info=True)


def window_icon(accent: str) -> str | None:
    """Значок окон и панели задач (.ico из значка трея) во временной папке."""
    try:
        from assistant.ui.tray import tray_image

        path = Path(tempfile.gettempdir()) / "jarvis-assistant.ico"
        tray_image(accent, 256).save(path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])
        return str(path)
    except Exception:
        log.debug("Значок окна не создан", exc_info=True)
        return None
