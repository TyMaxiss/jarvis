"""Когда показывать оверлей: пока голос слушает или выполняет команду и пока ждёт подтверждения.

Через hide_after секунд после того, как всё закончилось, оверлей прячется. Окно
показывает и прячет вызывающий код (show и hide); здесь только решение «когда».
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Состояния голоса, при которых оверлей виден.
ACTIVE_VOICE = frozenset({"listening", "capturing", "holding", "recognizing", "processing"})


class OverlayController:
    """Решает, когда показывать и прятать оверлей.

    Args:
        show: показать окно (без фокуса).
        hide: спрятать окно.
        enabled: включён ли оверлей в настройках.
        hide_after: через сколько секунд после конца команды прятать.
        timer: фабрика таймера с интерфейсом threading.Timer (в тестах — подделка).
    """

    def __init__(
        self,
        show: Callable[[], None],
        hide: Callable[[], None],
        enabled: Callable[[], bool],
        hide_after: float = 3.0,
        timer: Callable[[float, Callable[[], None]], Any] = threading.Timer,
    ) -> None:
        self._show = show
        self._hide = hide
        self._enabled = enabled
        self._hide_after = hide_after
        self._timer_factory = timer
        self._timer: Any = None
        self._voice_active = False
        self._confirming = False
        self.visible = False
        self._lock = threading.Lock()

    @property
    def listening(self) -> bool:
        """Голос сейчас слушает (оверлею нужна волна уровня микрофона)."""
        return self.visible and self._voice_active

    def voice_state(self, state: str) -> None:
        """Состояние голосового цикла изменилось."""
        with self._lock:
            self._voice_active = state in ACTIVE_VOICE
            action = self._update()
        self._call(action)

    def confirming(self, active: bool) -> None:
        """Появился или снят вопрос подтверждения по голосовой команде."""
        with self._lock:
            self._confirming = active
            action = self._update()
        self._call(action)

    def hide_now(self) -> None:
        """Спрятать сразу (оверлей выключили в настройках)."""
        with self._lock:
            self._cancel_timer()
            action = self._hide if self.visible else None
            self.visible = False
        self._call(action)

    def _update(self) -> Callable[[], None] | None:
        """Что сделать с окном (вызывается под блокировкой, само окно трогают после неё)."""
        if self._voice_active or self._confirming:
            self._cancel_timer()
            if not self.visible and self._enabled():
                self.visible = True
                return self._show
        elif self.visible and self._timer is None:
            self._timer = self._timer_factory(self._hide_after, self._expire)
            self._timer.daemon = True
            self._timer.start()
        return None

    def _expire(self) -> None:
        with self._lock:
            self._timer = None
            hide = self.visible and not (self._voice_active or self._confirming)
            if hide:
                self.visible = False
        self._call(self._hide if hide else None)

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    @staticmethod
    def _call(action: Callable[[], None] | None) -> None:
        if action is None:
            return
        try:
            action()
        except Exception:
            log.exception("Не удалось показать или спрятать оверлей")
