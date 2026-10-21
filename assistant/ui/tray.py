"""Значок в трее (pystray): открыть окно, быстрый чат, пауза, перечитать команды, выход.

pystray в Windows держит свой цикл сообщений в отдельном потоке (run_detached), а пункты
меню вызываются в этом потоке — поэтому обработчики только передают команду дальше.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class TrayActions:
    """Что делают пункты меню."""

    open_main: Callable[[], None]
    open_chat: Callable[[], None]
    toggle_pause: Callable[[], None]
    is_paused: Callable[[], bool]
    can_pause: Callable[[], bool]
    reload_commands: Callable[[], None]
    open_config: Callable[[], None]
    open_log: Callable[[], None]
    quit: Callable[[], None]


def tray_image(accent: str, size: int = 64) -> Any:
    """Значок: круг акцентного цвета с «волной» из трёх столбиков."""
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, size - 1, size - 1), fill=accent)
    bar = size // 9
    gap = size // 7
    heights = (0.32, 0.56, 0.32)
    center_x, center_y = size // 2, size // 2
    for index, height in enumerate(heights):
        x = center_x + (index - 1) * gap
        half = int(size * height / 2)
        draw.rounded_rectangle(
            (x - bar // 2, center_y - half, x + bar // 2, center_y + half), radius=bar // 2, fill="white"
        )
    return image


class Tray:
    """Значок в трее.

    Args:
        title: подсказка при наведении.
        accent: цвет значка.
        actions: обработчики пунктов меню.
    """

    def __init__(self, title: str, accent: str, actions: TrayActions) -> None:
        self._title = title
        self._accent = accent
        self._actions = actions
        self._icon: Any = None

    def start(self) -> None:
        """Показывает значок (цикл сообщений — в своём потоке)."""
        import pystray

        actions = self._actions
        menu = pystray.Menu(
            pystray.MenuItem("Открыть", lambda: actions.open_main(), default=True),
            pystray.MenuItem("Быстрый чат", lambda: actions.open_chat()),
            pystray.MenuItem(
                "Пауза прослушивания",
                lambda: actions.toggle_pause(),
                checked=lambda item: actions.is_paused(),
                enabled=lambda item: actions.can_pause(),
            ),
            pystray.MenuItem("Перечитать команды", lambda: actions.reload_commands()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Открыть config.yaml", lambda: actions.open_config()),
            pystray.MenuItem("Открыть лог", lambda: actions.open_log()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Выход", lambda: actions.quit()),
        )
        self._icon = pystray.Icon("jarvis", tray_image(self._accent), self._title, menu)
        self._icon.run_detached()
        log.info("Значок в трее показан")

    def update(self, title: str | None = None) -> None:
        """Обновляет подсказку и состояние пунктов меню (например, галочку паузы)."""
        if self._icon is None:
            return
        if title is not None and title != self._title:
            self._title = title
            self._icon.title = title[:127]  # Windows ограничивает подсказку 128 символами
        self._icon.update_menu()

    def notify(self, text: str) -> None:
        """Всплывающее уведомление Windows рядом с часами."""
        if self._icon is not None:
            try:
                self._icon.notify(text, self._title)
            except Exception:
                log.debug("Уведомление в трее не показано", exc_info=True)

    def stop(self) -> None:
        """Убирает значок."""
        if self._icon is not None:
            self._icon.stop()
            self._icon = None
