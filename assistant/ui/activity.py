"""Что происходит с командами — для окон: распознанная фраза, шаги, ответ, итог.

Activity ведёт текущую команду и последние выполненные (для главной), VoiceStatus
передаёт ему сообщения голосового цикла, UiSpeaker — ответы ассистента.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

from assistant.ui.events import EventSink

log = logging.getLogger(__name__)

# Состояния голосового цикла, которые показывает интерфейс.
VOICE_STATES = frozenset({"idle", "listening", "capturing", "holding", "recognizing", "processing", "paused"})


class Activity:
    """Текущая команда и последние выполненные.

    Args:
        events: куда отправлять события окнам.
        now: текущее время (подменяется в тестах).
        limit: сколько последних команд помнить.
    """

    def __init__(
        self, events: EventSink, now: Callable[[], datetime] = datetime.now, limit: int = 20
    ) -> None:
        self._events = events
        self._now = now
        self._limit = limit
        self._lock = threading.Lock()
        self._recent: list[dict[str, Any]] = []
        self._reply = ""
        self._next_id = 1

    @property
    def recent(self) -> list[dict[str, Any]]:
        """Последние команды, новые первыми."""
        with self._lock:
            return list(self._recent)

    def begin(self, text: str, source: str) -> None:
        """Началась команда: фраза распознана (голос) или введена (текст)."""
        with self._lock:
            self._reply = ""
        self._events.emit("recognized", {"text": text, "source": source})

    def partial(self, text: str) -> None:
        """Слова фразы по ходу речи (до Whisper) — только оверлею, он их и показывает."""
        self._events.emit("partial", {"text": text}, to=["overlay"])

    def step(self, number: int, total: int | None, action: str, text: str, ok: bool) -> None:
        """Выполнен шаг команды или вызов инструмента нейросетью."""
        payload = {"number": number, "total": total, "action": action, "text": text, "ok": ok}
        self._events.emit("step", payload)

    def reply(self, text: str) -> None:
        """Ассистент сказал фразу: окна получают весь ответ на текущую команду."""
        with self._lock:
            self._reply = f"{self._reply} {text}".strip()
            reply = self._reply
        self._events.emit("reply", {"text": reply})

    def finish(self, result: Any, source: str) -> dict[str, Any] | None:
        """Команда выполнена: результат конвейера (PipelineResult) становится строкой в «Последних».

        Пустая фраза (ничего не сказали) не записывается.
        """
        if result.match.status == "empty":
            return None
        if result.route == "local":
            ok = all(outcome.run.ok for outcome in result.outcomes)
        else:
            ok = result.agent is not None and result.agent.ok
        return self._record(result.text, result.route, result.reply, ok, result.total_ms, source)

    def failed(self, text: str, source: str, reason: str) -> dict[str, Any]:
        """Команда упала с ошибкой, не дойдя до результата."""
        return self._record(text, "local", reason, False, 0.0, source)

    def _record(
        self, phrase: str, route: str, reply: str, ok: bool, ms: float, source: str
    ) -> dict[str, Any]:
        with self._lock:
            item = {
                "id": self._next_id,
                "time": self._now().strftime("%H:%M"),
                "phrase": phrase,
                "route": route,
                "reply": reply or self._reply,
                "ok": ok,
                "ms": round(ms),
                "source": "voice" if source == "voice" else "text",
            }
            self._next_id += 1
            self._recent = [item, *self._recent][: self._limit]
        self._events.emit("result", item)
        return item


class VoiceStatus:
    """Сообщения голосового цикла (StatusSink) → состояние голоса и команды в окнах.

    Args:
        activity: текущая команда.
        on_state: состояние голоса изменилось (idle, listening… paused).
        on_result: голосовая команда выполнена (оверлей держит её итог).
        on_phrase: распознана новая фраза (оверлей больше не держит прежний итог).
    """

    def __init__(
        self,
        activity: Activity,
        on_state: Callable[[str], None],
        on_result: Callable[[], None] | None = None,
        on_phrase: Callable[[], None] | None = None,
    ) -> None:
        self._activity = activity
        self._on_state = on_state
        self._on_result = on_result
        self._on_phrase = on_phrase

    def voice_status(self, state: str, detail: str = "") -> None:
        if state == "recognized":
            if self._on_phrase is not None:
                self._on_phrase()
            self._activity.begin(detail, "voice")
        elif state == "partial":
            self._activity.partial(detail)
        elif state in VOICE_STATES:
            self._on_state(state)
        elif state == "latency":
            log.debug("Голос: %s", detail)

    def voice_result(self, result: Any) -> None:
        if self._activity.finish(result, "voice") is not None and self._on_result is not None:
            self._on_result()

    def voice_failed(self, text: str, reason: str) -> None:
        """Голосовая команда упала с ошибкой: её итог — ошибка, оверлей держит его, как любой другой."""
        self._activity.failed(text, "voice", reason)
        if self._on_result is not None:
            self._on_result()


class UiSpeaker:
    """«Говорящий» для окон: каждая фраза ассистента добавляется к ответу текущей команды."""

    def __init__(self, activity: Activity) -> None:
        self._activity = activity

    def say(self, text: str) -> None:
        self._activity.reply(text)
