"""Глобальные горячие клавиши (библиотека keyboard, работает только в Windows).

Сочетания пишутся латиницей: "ctrl+alt+space", "ctrl+alt+q". Клавиши сопоставляются по
физическим скан-кодам, поэтому работают и при русской раскладке.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


class HotkeyError(ValueError):
    """Сочетание клавиш записано с ошибкой."""


class HotkeyManager:
    """Регистрирует глобальные сочетания и снимает их при выходе.

    Колбэки вызываются в потоке библиотеки keyboard, поэтому должны быть быстрыми:
    только передать событие дальше.
    """

    def __init__(self) -> None:
        self._keyboard: Any = None
        self._removers: list[Callable[[], None]] = []

    def _kb(self) -> Any:
        if self._keyboard is None:
            import keyboard

            self._keyboard = keyboard
        return self._keyboard

    def _parse(self, combo: str) -> Any:
        try:
            steps = self._kb().parse_hotkey(combo)
        except ValueError as exc:
            raise HotkeyError(f"Неизвестное сочетание клавиш «{combo}»: {exc}") from exc
        if len(steps) != 1:
            raise HotkeyError(f"Сочетание «{combo}» должно быть одним шагом, без запятых")
        return steps[0]

    def add_press(self, combo: str, callback: Callable[[], None]) -> None:
        """Вызывает callback при нажатии сочетания."""
        self._parse(combo)
        self._removers.append(self._kb().add_hotkey(combo, callback, suppress=False))
        log.info("Горячая клавиша %s", combo)

    def add_hold(self, combo: str, on_press: Callable[[], None], on_release: Callable[[], None]) -> None:
        """on_press — когда сочетание зажато, on_release — когда отпущена любая из его клавиш.

        Встроенный trigger_on_release библиотеки keyboard ненадёжен, поэтому
        состояние отслеживается по событиям клавиш вручную.
        """
        kb = self._kb()
        codes = {code for alternatives in self._parse(combo) for code in alternatives}
        active = False

        def handler(event: Any) -> None:
            nonlocal active
            if event.scan_code not in codes:
                return
            if event.event_type == kb.KEY_DOWN:
                if not active and kb.is_pressed(combo):
                    active = True
                    on_press()
            elif active:
                active = False
                on_release()

        self._removers.append(kb.hook(handler))
        log.info("Горячая клавиша (удержание) %s", combo)

    def close(self) -> None:
        """Снимает все сочетания."""
        for remove in self._removers:
            with contextlib.suppress(KeyError, ValueError):
                remove()
        self._removers.clear()
