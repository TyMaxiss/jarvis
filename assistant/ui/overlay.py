"""Когда показывать оверлей: пока голос слушает или выполняет команду и пока ждёт подтверждения.

Через hide_after секунд после того, как всё закончилось, оверлей прячется. Если включено
keep, итог голосовой команды остаётся на экране до следующей голосовой фразы (или клика по
оверлею). Окно показывает и прячет вызывающий код (show и hide); здесь только решение «когда».
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

# Состояния голоса, при которых оверлей виден.
ACTIVE_VOICE = frozenset({"listening", "capturing", "holding", "recognizing", "processing"})
# Голос выключили или поставили на паузу: следующей фразы не будет — итог команды не держим.
QUIET_VOICE = frozenset({"paused", "off"})


class OverlayController:
    """Решает, когда показывать и прятать оверлей.

    Args:
        show: показать окно (без фокуса).
        hide: спрятать окно.
        enabled: включён ли оверлей в настройках.
        keep: держать ли итог голосовой команды до следующей фразы (настройка).
        hide_after: через сколько секунд после конца команды прятать.
        timer: фабрика таймера с интерфейсом threading.Timer (в тестах — подделка).
    """

    def __init__(
        self,
        show: Callable[[], None],
        hide: Callable[[], None],
        enabled: Callable[[], bool],
        keep: Callable[[], bool] = lambda: False,
        hide_after: float = 3.0,
        timer: Callable[[float, Callable[[], None]], Any] = threading.Timer,
    ) -> None:
        self._show = show
        self._hide = hide
        self._enabled = enabled
        self._keep = keep
        self._hide_after = hide_after
        self._timer_factory = timer
        self._timer: Any = None
        self._voice_active = False
        self._confirming = False
        self._holding = False  # на экране итог голосовой команды, следующей фразы ещё не было
        self._stopped = False  # текущую команду остановили — её итог не держим
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
            if state == "recognizing" or state in QUIET_VOICE:
                self._holding = False  # прозвучала новая фраза: прежний итог ей уступает место
            if state == "recognizing":
                self._stopped = False
            action = self._update()
        self._call(action)

    def confirming(self, active: bool) -> None:
        """Появился или снят вопрос подтверждения по голосовой команде."""
        with self._lock:
            self._confirming = active
            action = self._update()
        self._call(action)

    def command_done(self) -> None:
        """Голосовая команда выполнена: её итог остаётся на экране до следующей фразы (если keep)."""
        with self._lock:
            self._holding = self.visible and not self._stopped
            action = self._update()
        self._call(action)

    def stopped(self) -> None:
        """Ассистента остановили («стоп», кнопка, Ctrl+Alt+Q): итог прерванной команды не держим —
        оверлей уйдёт через hide_after, как без keep."""
        with self._lock:
            self._holding = False
            self._stopped = True
            action = self._update()
        self._call(action)

    def refresh(self) -> None:
        """Настройки изменились (keep выключили — держать итог больше не нужно)."""
        with self._lock:
            action = self._update()
        self._call(action)

    def hide_now(self) -> None:
        """Спрятать сразу (клик по оверлею или оверлей выключили в настройках)."""
        with self._lock:
            self._cancel_timer()
            self._holding = False
            action = self._hide if self.visible else None
            self.visible = False
        self._call(action)

    def _held(self) -> bool:
        return self._holding and self._keep()

    def _update(self) -> Callable[[], None] | None:
        """Что сделать с окном (вызывается под блокировкой, само окно трогают после неё)."""
        if self._voice_active or self._confirming:
            self._cancel_timer()
            if not self.visible and self._enabled():
                self.visible = True
                self._holding = False  # новый показ начинается без прошлых итогов (как и страница)
                return self._show
        elif self._held():
            self._cancel_timer()
        elif self.visible and self._timer is None:
            self._timer = self._timer_factory(self._hide_after, self._expire)
            self._timer.daemon = True
            self._timer.start()
        return None

    def _expire(self) -> None:
        with self._lock:
            self._timer = None
            hide = self.visible and not (self._voice_active or self._confirming or self._held())
            if hide:
                self.visible = False
                self._holding = False
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
