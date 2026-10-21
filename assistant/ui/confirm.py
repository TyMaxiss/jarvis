"""Подтверждение рискованных действий в интерфейсе: кнопки «Да / Нет» и голос.

Окна показывают вопрос с кнопками (оверлей, быстрый чат, главное окно). Для голосовых
команд ассистент ещё и спрашивает вслух и слушает «да» или «нет» — засчитывается
первый ответ. Молчание означает отмену: после конца голосового ожидания кнопку ещё
можно нажать несколько секунд, без голоса — до timeout.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any

from assistant.ui.events import EventSink

log = logging.getLogger(__name__)


class _Pending:
    """Вопрос, на который ждут ответ."""

    def __init__(self, question: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.question = question
        self.answer: bool | None = None
        self.answered = threading.Event()
        self.voice_done = threading.Event()
        self.cancel = threading.Event()  # перестать слушать голос

    def resolve(self, answer: bool) -> None:
        if not self.answered.is_set():
            self.answer = answer
            self.answered.set()


class UiConfirmer:
    """Спрашивает подтверждение в окнах и (для голоса) вслух.

    Args:
        events: куда отправлять вопрос (событие confirm) и его снятие (confirm_done).
        voice: голосовое подтверждение (VoiceConfirmer) или None.
        timeout: сколько ждать нажатия кнопки, если голосом не спрашиваем, секунды.
        grace: сколько ещё ждать кнопку после того, как голосом ответа не было.
        on_change: вызывается с True, когда появился вопрос по голосовой команде, и с False,
            когда он снят (показать и спрятать оверлей).
    """

    def __init__(
        self,
        events: EventSink,
        voice: Any | None = None,
        timeout: float = 30.0,
        grace: float = 3.0,
        on_change: Callable[[bool], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._events = events
        self.voice = voice
        self._timeout = timeout
        self._grace = grace
        self.on_change = on_change
        self._clock = clock
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    def for_source(self, source: str) -> _BoundConfirmer:
        """Confirmer для конвейера: голосом спрашивается команда, сказанная голосом или клавишей."""
        return _BoundConfirmer(self, use_voice=source in ("voice", "hotkey"))

    def answer(self, question_id: str, yes: bool) -> bool:
        """Ответ кнопкой из окна. False — такого вопроса уже нет."""
        with self._lock:
            pending = self._pending.get(question_id)
        if pending is None:
            return False
        log.info("Подтверждение кнопкой «%s»: %s", pending.question, "да" if yes else "нет")
        pending.resolve(yes)
        pending.cancel.set()
        return True

    def ask(self, question: str, use_voice: bool) -> bool:
        """Задаёт вопрос и ждёт ответа. True — только при явном «да»."""
        pending = _Pending(question)
        with self._lock:
            self._pending[pending.id] = pending
        voice = self.voice if use_voice else None
        if voice is not None and self.on_change is not None:
            self.on_change(True)
        self._events.emit("confirm", {"id": pending.id, "question": question})
        try:
            if voice is not None:
                listener = threading.Thread(
                    target=self._listen, args=(voice, pending), name="confirm-voice", daemon=True
                )
                listener.start()
            self._wait(pending, voice is not None)
        finally:
            pending.cancel.set()
            with self._lock:
                self._pending.pop(pending.id, None)
            self._events.emit("confirm_done", {"id": pending.id})
            if voice is not None and self.on_change is not None:
                self.on_change(False)
        accepted = pending.answer is True
        log.info("Подтверждение «%s»: %s", question, "да" if accepted else "нет")
        return accepted

    def _listen(self, voice: Any, pending: _Pending) -> None:
        try:
            heard = voice.answer(pending.question, pending.cancel)
        except Exception:
            log.exception("Ошибка голосового подтверждения")
            heard = None
        if heard is not None:
            pending.resolve(heard)
        pending.voice_done.set()

    def _wait(self, pending: _Pending, with_voice: bool) -> None:
        deadline = self._clock() + self._timeout
        grace_until: float | None = None
        while not pending.answered.wait(0.05):
            now = self._clock()
            if now >= deadline:
                return
            if with_voice and pending.voice_done.is_set():
                if grace_until is None:
                    grace_until = now + self._grace
                elif now >= grace_until:
                    return


class _BoundConfirmer:
    """Confirmer (протокол из safety.confirm) для одной команды."""

    def __init__(self, owner: UiConfirmer, use_voice: bool) -> None:
        self._owner = owner
        self._use_voice = use_voice

    def confirm(self, question: str) -> bool:
        return self._owner.ask(question, self._use_voice)
