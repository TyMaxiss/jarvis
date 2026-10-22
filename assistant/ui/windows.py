"""Окна pywebview: главное, оверлей без фокуса и быстрый чат.

Оверлей показывается через WinAPI (без активации): плашка выезжает снизу, сфера раскрывается
на своём месте из центра (slide.py). Окно сферы обрезано по форме шара и карточки, а с «Эффектами
прозрачности» Windows ещё и смешивается с рабочим столом по альфа-каналу (полупрозрачные внешнее
кольцо и карточка). Остальные окна — средствами pywebview.
Окна без рамки: перетаскивание — за элементы с классом pywebview-drag-region, размер —
уголком в интерфейсе, «развернуть» — на рабочую область монитора.
"""

from __future__ import annotations

import logging
import math
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
# Форма окна сферы, логические пиксели (числа — из ui/src/index.css, их сверяет тест): круг шара
# 378 с левым верхним углом в (31, 17) и карточка 358×126 со скруглением 18 в (41, 266) — точно
# по их краю: без «Эффектов прозрачности» окно непрозрачное, и за краем фигур была бы видна
# заливка формы. Скруглённые прямоугольники: left, top, right, bottom (включительно), радиус
# скругления (радиус в полстороны — круг). Вне формы окна нет: ни фона, ни кликов.
ORB_SHAPE = ((31, 17, 408, 394, 189), (41, 266, 398, 391, 18))
ORB_CENTER = ((31 + 409) / 2, (17 + 395) / 2)  # центр шара: из него сфера раскрывается
# Круг такого радиуса вокруг центра шара открывает всю форму окна: дальше всего — углы карточки.
ORB_REACH = math.ceil(
    max(
        math.hypot(x - ORB_CENTER[0], y - ORB_CENTER[1])
        for left, top, right, bottom, _ in ORB_SHAPE[1:]
        for x in (left, right + 1)
        for y in (top, bottom + 1)
    )
)
# Фон окна до загрузки страницы — как у интерфейса, чтобы не мигало белым.
BACKGROUNDS = {"dark": ("#0F1013", "#15161A"), "light": ("#F6F6F8", "#FFFFFF")}


def scaled_shape(shape: tuple[tuple[int, ...], ...], scale: float) -> tuple[tuple[int, ...], ...]:
    """Форма окна (как ORB_SHAPE) в физических пикселях: края фигур — по масштабу, а не их последние
    пиксели (иначе при 200 % фигура выйдет на пиксель уже, чем на странице)."""
    return tuple(
        (round(left * scale), round(top * scale), round((right + 1) * scale) - 1,
         round((bottom + 1) * scale) - 1, round(radius * scale))
        for left, top, right, bottom, radius in shape
    )  # fmt: skip


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
        self._overlay_style = overlay_style if overlay_style in OVERLAY_SIZES else "classic"
        self._overlay_look = "classic"  # каким окно оверлея сделано сейчас: непрозрачным, скруглённым
        self._overlay_shape: tuple[tuple[int, ...], ...] | None = None  # форма окна сферы, пиксели
        self._overlay_circle = (0, 0, 0)  # центр шара и радиус, открывающий всю форму, пиксели
        self._overlay_region: tuple[Any, ...] | None = None  # как окно обрезано сейчас
        self._appear_style = self._overlay_style  # оформление, с которым оверлей показан сейчас
        self._glass_wanted = False  # оверлей показан полупрозрачным (так сказано и странице)
        self._overlay_glass = False  # окно оверлея сейчас смешивается с рабочим столом
        self._glass_works = True  # Windows приняла прозрачность окна (если нет — больше не пробуем)
        # Показанный оверлей устарел (выключили «Эффекты прозрачности»): его прячут, следующий
        # показ будет уже по-новому. Подставляет UiApp.
        self.on_overlay_stale: Callable[[], None] | None = None
        self._backdrop_guard: Callable[[Any, Any], None] | None = None  # обработчик настроек Windows
        self._visible = {"main": not start_hidden, "overlay": False, "chat": False}
        background, surface = BACKGROUNDS.get(theme, BACKGROUNDS["dark"])
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
                # До make_overlay: подложку Windows применяет при смене рамки окна (SWP_FRAMECHANGED).
                self._win32.remove_backdrop(self._handles[name])
                self._keep_overlay_without_backdrop()
                self._win32.make_overlay(self._handles[name])
                self._clear_background(window)

        return handler

    def _keep_overlay_without_backdrop(self) -> None:
        """pywebview ставит окнам Mica в тёмной теме Windows — при создании окна и снова при каждом
        изменении настроек Windows (SystemEvents.UserPreferenceChanged). У оверлея она — серый
        прямоугольник во всё окно, поэтому после pywebview подложка снова снимается.

        Подписка — в потоке окон после pywebview (before_show — после создания формы): .NET вызывает
        обработчики по порядку подписки, так что этот срабатывает следом за его.
        """
        try:
            from Microsoft.Win32 import SystemEvents  # type: ignore[import-not-found]

            def on_change(sender: Any, args: Any) -> None:
                self._settings_changed()

            SystemEvents.UserPreferenceChanged += on_change
            self._backdrop_guard = on_change  # обработчик живёт, пока живут окна
        except Exception:
            log.debug("Не удалось подписаться на изменение настроек Windows", exc_info=True)

    def _settings_changed(self) -> None:
        """Настройки Windows изменились: снова без Mica, а если оверлей показан не с той прозрачностью,
        которую теперь просят «Эффекты прозрачности», — он устарел (его спрячут)."""
        self._win32.remove_backdrop(self._handles.get("overlay", 0))
        stale = self._wants_glass(self._appear_style) != self._glass_wanted
        if stale and self._visible["overlay"] and self.on_overlay_stale is not None:
            self.on_overlay_stale()

    @staticmethod
    def _clear_background(window: Any) -> None:
        """Под страницей оверлея — ничего: страница рисует себя сама (плашка — целиком, сфера — шар
        и карточку). Фон WebView2 прозрачный с самого создания окна (обработчик — в потоке окон)."""
        try:
            from System.Drawing import Color  # type: ignore[import-not-found]

            # Чёрный — прозрачный, когда окно смешивается с рабочим столом (overlay_glass); иначе он
            # виден разве что в сглаженных краях шара и карточки.
            window.native.BackColor = Color.Black
            window.native.webview.DefaultBackgroundColor = Color.Transparent
        except Exception:
            log.warning("Фон оверлея не сделан прозрачным", exc_info=True)

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
            # Оформление и прозрачность решаются здесь, один раз на показ: их же узнаёт страница
            # (overlay_glass), а окно под них перестраивает поток анимации (_overlay_appear).
            self._appear_style = self._overlay_style
            self._glass_wanted = self._wants_glass(self._appear_style)
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

    def overlay_glass(self) -> bool:
        """Показанный оверлей полупрозрачный: окно сферы смешивается с рабочим столом, внешнее
        кольцо шара и карточка — как стекло."""
        return self._glass_wanted

    def overlay_glass_next(self) -> bool:
        """Оверлей, если показать его сейчас, будет полупрозрачным (см. overlay_glass)."""
        return self._wants_glass(self._overlay_style)

    def _wants_glass(self, style: str) -> bool:
        """Сфера полупрозрачная только с «Эффектами прозрачности» Windows: кто их выключил, видит её
        непрозрачной. Плашка всегда непрозрачная."""
        return style == "orb" and self._glass_works and self._win32.transparency_enabled()

    def _overlay_appear(self) -> Track:
        """Ставит оверлей внизу по центру монитора с курсором — плашку за краем, сферу на место, обе
        невидимыми — и показывает без фокуса. Окно сферы — только форма шара и карточки (ORB_SHAPE)."""
        hwnd, style = self._handle("overlay"), self._appear_style
        # Без подложки Windows (Mica) — на случай, если pywebview поставил её снова — и без фокуса:
        # пользователь продолжает печатать в своём окне.
        self._win32.remove_backdrop(hwnd)
        self._win32.make_overlay(hwnd)
        self._apply_overlay_look(hwnd, style)
        self._apply_overlay_glass(hwnd, self._glass_wanted)
        area = self._win32.work_area()
        scale = self._win32.dpi_scale(hwnd)
        width, height = (int(side * scale) for side in OVERLAY_SIZES[style])
        x, top = self._win32.bottom_center((width, height), area, round(OVERLAY_MARGIN * scale))
        if self._win32.window_size(hwnd) != (width, height):
            self._win32.set_bounds(hwnd, x, top, width, height)
        if style == "orb":
            # Сфера не едет, а раскрывается на месте из центра шара: путь неподвижный.
            track = Track(x, top, top, width, height)
            self._overlay_shape = scaled_shape(ORB_SHAPE, scale)
            center_x, center_y = ORB_CENTER
            reach = math.ceil(ORB_REACH * scale)
            self._overlay_circle = (round(center_x * scale), round(center_y * scale), reach)
        else:
            radius = round(OVERLAY_RADIUS * scale) if self._win32.rounds_corners() else 0
            track = Track(x, top, area[3], width, height, radius)
            self._overlay_shape = None
        self._overlay_region = None  # окно могло остаться обрезанным иначе — ставим заново
        self._clip_overlay(hwnd, track, 0.0)
        self._win32.show_without_focus(hwnd, x, track.y(0))
        return track

    def _apply_overlay_look(self, hwnd: int, style: str) -> None:
        """Окно оверлея под оформление: сфере — без рамки и скругления Windows, плашке — скруглённое.

        Форма окна сферы — регион (ORB_SHAPE): им же плашка выезжает из-за панели задач. Серый
        прямоугольник вокруг шара в 0.7.0–0.7.2 был подложкой Mica, которую pywebview ставит окнам
        в тёмной теме Windows (её снимает remove_backdrop).
        """
        if style == self._overlay_look:
            return
        self._win32.round_corners(hwnd, rounded=style != "orb")
        self._overlay_look = style

    def _apply_overlay_glass(self, hwnd: int, glass: bool) -> None:
        """Прозрачность окна оверлея по альфа-каналу страницы (сфере с «Эффектами прозрачности»)."""
        if glass == self._overlay_glass:
            return
        if self._win32.set_transparent(hwnd, glass) or not glass:
            self._overlay_glass = glass
            return
        if hwnd:
            # Без прозрачности полупрозрачные места страницы легли бы на чёрный: со следующего показа
            # сфера непрозрачная.
            self._glass_works = False
            log.warning("Windows не приняла прозрачность окна сферы — дальше она будет непрозрачной")

    def _overlay_place(self, track: Track, shown: float) -> None:
        """Кадр анимации: плашка сдвинута, а то, что ниже края рабочей области, обрезано; сфера
        раскрыта на долю shown."""
        hwnd = self._handle("overlay")
        if not track.still:
            self._win32.move_window(hwnd, track.x, track.y(shown))
        self._clip_overlay(hwnd, track, shown)

    def _clip_overlay(self, hwnd: int, track: Track, shown: float) -> None:
        """Обрезает окно под долю появления shown: у плашки видна полоса над краем рабочей области,
        у сферы — её форма в круге из центра шара.

        Полупрозрачную сферу раскрывает сама страница (растёт из центра и проявляется), окну — вся
        форма с первого кадра анимации; до него окно пустое: у страницы есть кадр, чтобы сделать сферу
        невидимой, а не показать то, что было в окне в прошлый раз. Непрозрачную — круг: уменьшенная
        страница легла бы на чёрный фон окна.
        """
        if self._overlay_shape is None:
            visible = track.visible(track.y(shown))
            if ("panel", visible) != self._overlay_region:  # обрезано так же — регион не трогаем
                self._win32.clip_window(hwnd, track.width, visible, track.radius)
                self._overlay_region = ("panel", visible)
            return
        x, y, reach = self._overlay_circle
        whole = shown >= 1 or (self._overlay_glass and shown > 0)
        circle = None if whole else (x, y, round(reach * shown))
        if ("orb", circle) != self._overlay_region:
            self._win32.shape_window(hwnd, self._overlay_shape, None, circle)
            self._overlay_region = ("orb", circle)

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
