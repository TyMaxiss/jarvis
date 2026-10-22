"""Окна pywebview: главное, оверлей без фокуса и быстрый чат.

Оверлей показывается через WinAPI (без активации): плашка выезжает снизу (slide.py), сфера —
в прозрачном окне, её появление рисует страница. Остальные окна — средствами pywebview.
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

from assistant.ui.slide import Slide, Track

log = logging.getLogger(__name__)

UI_DIST = Path(__file__).resolve().parents[2] / "ui" / "dist"
MAIN_SIZE, MAIN_MIN = (1100, 720), (880, 600)
OVERLAY_SIZES = {"classic": (480, 64), "orb": (440, 420)}  # плашка и сфера, логические пиксели
OVERLAY_MIN = (min(w for w, _ in OVERLAY_SIZES.values()), min(h for _, h in OVERLAY_SIZES.values()))
CHAT_SIZE = (620, 400)
OVERLAY_MARGIN = 24  # от панели задач, логические пиксели
OVERLAY_RADIUS = 8  # скругление углов окна в Windows 11, логические пиксели
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
        overlay_style: оформление оверлея: classic — плашка, orb — сфера.
        animate: двигать ли оверлей при показе; по умолчанию — если в Windows включены анимации.
    """

    def __init__(
        self,
        webview: Any,
        urls: dict[str, str],
        apis: dict[str, Any],
        theme: str,
        start_hidden: bool,
        overlay_style: str = "classic",
        animate: Callable[[], bool] | None = None,
    ) -> None:
        from assistant.ui import win32

        self._win32 = win32
        self._quitting = False
        self._handles: dict[str, int] = {}
        self._saved_bounds: tuple[int, int, int, int] | None = None
        self._overlay_clipped = False  # у оверлея есть регион (его низ обрезан)
        self._overlay_style = overlay_style if overlay_style in OVERLAY_SIZES else "classic"
        self._overlay_look = "classic"  # каким окно оверлея сделано сейчас: непрозрачным, скруглённым
        self._visible = {"main": not start_hidden, "overlay": False, "chat": False}
        background, surface = BACKGROUNDS.get(theme, BACKGROUNDS["dark"])
        self._surface = surface
        overlay_size = OVERLAY_SIZES[self._overlay_style]
        common = {"frameless": True, "easy_drag": False}
        self._windows = {
            "main": webview.create_window(
                "Джарвис", urls["main"], js_api=apis["main"], width=MAIN_SIZE[0], height=MAIN_SIZE[1],
                min_size=MAIN_MIN, hidden=start_hidden, background_color=background, **common,
            ),
            "overlay": webview.create_window(
                "Джарвис — оверлей", urls["overlay"], js_api=apis["overlay"], width=overlay_size[0],
                height=overlay_size[1], min_size=OVERLAY_MIN, resizable=False, shadow=False, on_top=True,
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
        self._overlay = Slide(
            self._overlay_appear,
            self._overlay_place,
            self._windows["overlay"].hide,
            animate or win32.animations_enabled,
            win32.wait_frame,
        )

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
            self.hide(name)
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
        self._visible[name] = True
        if name == "overlay":
            self._overlay.show()
            return
        window = self._windows[name]
        window.show()
        if name == "main":
            window.restore()
        self._win32.bring_to_front(self._handle(name))

    def hide(self, name: str) -> None:
        self._visible[name] = False
        if name == "overlay":
            self._overlay.hide()
            return
        self._windows[name].hide()

    def set_overlay_style(self, style: str) -> None:
        """Оформление оверлея: classic — плашка, orb — сфера; окно станет таким при следующем показе."""
        if style in OVERLAY_SIZES:
            self._overlay_style = style

    def _overlay_appear(self) -> Track:
        """Ставит оверлей внизу по центру монитора с курсором и показывает без фокуса.

        Плашка встаёт за краем рабочей области (её не видно) и выезжает. Сфера встаёт сразу на
        место в прозрачном окне: появление рисует страница, а путь нулевой длины только
        выдерживает время, пока страница доигрывает уход.
        """
        hwnd, style = self._handle("overlay"), self._overlay_style
        # Без фокуса: пользователь продолжает печатать в своём окне.
        self._win32.make_overlay(hwnd)
        self._apply_overlay_look(hwnd, style)
        area = self._win32.work_area()
        scale = self._win32.dpi_scale(hwnd)
        width, height = (int(side * scale) for side in OVERLAY_SIZES[style])
        x, top = self._win32.bottom_center((width, height), area, round(OVERLAY_MARGIN * scale))
        if self._win32.window_size(hwnd) != (width, height):
            self._win32.set_bounds(hwnd, x, top, width, height)
        if style == "orb":
            if self._overlay_clipped:
                self._win32.clip_window(hwnd, width, None)
                self._overlay_clipped = False
            self._win32.show_without_focus(hwnd, x, top)
            return Track(x, top, top, width, height)
        radius = round(OVERLAY_RADIUS * scale) if self._win32.rounds_corners() else 0
        track = Track(x, top, area[3], width, height, radius)
        self._win32.clip_window(hwnd, width, 0, radius)
        self._overlay_clipped = True
        self._win32.show_without_focus(hwnd, x, track.edge)
        return track

    def _apply_overlay_look(self, hwnd: int, style: str) -> None:
        """Окно оверлея под оформление: сфере — прозрачное без рамки, плашке — непрозрачное, скруглённое."""
        if style == self._overlay_look:
            return
        orb = style == "orb"
        if not self._win32.set_transparent(hwnd, orb) and orb:
            log.warning("Окно сферы не стало прозрачным: вокруг шара будет тёмный фон")
        self._win32.round_corners(hwnd, rounded=not orb)
        self._set_background("overlay", None if orb else self._surface)
        self._overlay_look = style

    def _set_background(self, name: str, color: str | None) -> None:
        """Фон окна под страницей — у формы WinForms и у WebView2: цвет #rrggbb или прозрачный (None)."""
        try:
            from System import Action  # type: ignore[import-not-found]
            from System.Drawing import Color, ColorTranslator  # type: ignore[import-not-found]

            form = self._windows[name].native

            def apply() -> None:
                # Под прозрачным окном чёрный фон формы прозрачен: Windows смешивает окно по
                # альфа-каналу, а чёрная заливка формы его не трогает (он остаётся нулевым).
                back = Color.Black if color is None else ColorTranslator.FromHtml(color)
                form.BackColor = back
                form.webview.DefaultBackgroundColor = Color.Transparent if color is None else back

            form.Invoke(Action(apply))
        except Exception:
            log.warning("Фон окна %s не изменён", name, exc_info=True)

    def _overlay_place(self, track: Track, y: int) -> None:
        """Кадр анимации: оверлей сдвинут, а то, что ниже края рабочей области, обрезано."""
        hwnd = self._handle("overlay")
        self._win32.move_window(hwnd, track.x, y)
        visible = track.visible(y)
        if visible is not None or self._overlay_clipped:  # регион снимается один раз
            self._win32.clip_window(hwnd, track.width, visible, track.radius)
        self._overlay_clipped = visible is not None

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
        self._overlay.close()
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
