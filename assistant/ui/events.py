"""События для окон интерфейса: window.__assistantEvent({type, payload}).

Скрипт в окне выполняется вызовом run_js, а он ждёт, пока окно его выполнит, поэтому
у каждого окна свой поток отправки с очередью: голосовой цикл и агент не ждут интерфейс.
Окно начинает получать события, когда его страница загрузилась и сама запросила
состояние (attach); всё, что было раньше, она получает в этом состоянии.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol

log = logging.getLogger(__name__)

_STOP = object()


class EventSink(Protocol):
    """Куда отправлять события окнам (EventBus, в тестах — запись в список)."""

    def emit(self, kind: str, payload: Any = None, *, to: Iterable[str] | None = None) -> None: ...


def event_script(kind: str, payload: Any) -> str:
    """JS, который передаёт событие обработчику страницы (если он уже есть)."""
    data = json.dumps({"type": kind, "payload": payload}, ensure_ascii=False)
    return f"window.__assistantEvent && window.__assistantEvent({data})"


class _Target:
    """Окно-получатель: очередь скриптов и поток, который их выполняет."""

    def __init__(self, name: str, send: Callable[[str], None], limit: int) -> None:
        self.name = name
        self.send = send
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=limit)
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, name=f"events-{name}", daemon=True)
        self.thread.start()

    def put(self, script: str) -> None:
        try:
            self.queue.put_nowait(script)
        except queue.Full:
            self.dropped += 1  # окно не успевает (например, зависло) — новые события выбрасываются
            if self.dropped == 1:
                log.warning("Окно %s не успевает обрабатывать события", self.name)

    def _run(self) -> None:
        while (script := self.queue.get()) is not _STOP:
            try:
                self.send(script)
            except Exception:
                log.debug("Событие окну %s не доставлено", self.name, exc_info=True)

    def stop(self) -> None:
        with contextlib.suppress(queue.Full):
            self.queue.put_nowait(_STOP)


class EventBus:
    """Рассылает события окнам интерфейса.

    Args:
        limit: сколько событий может ждать отправки в одно окно.
    """

    def __init__(self, limit: int = 500) -> None:
        self._limit = limit
        self._targets: dict[str, _Target] = {}
        self._lock = threading.Lock()

    def attach(self, name: str, send: Callable[[str], None]) -> None:
        """Окно готово получать события; send выполняет скрипт в нём."""
        with self._lock:
            old = self._targets.get(name)
            if old is not None and old.send == send:
                return
            self._targets[name] = _Target(name, send, self._limit)
        if old is not None:
            old.stop()

    def attached(self, name: str) -> bool:
        """Окно уже получает события."""
        with self._lock:
            return name in self._targets

    def emit(self, kind: str, payload: Any = None, *, to: Iterable[str] | None = None) -> None:
        """Отправляет событие всем окнам (или только перечисленным в to)."""
        script = event_script(kind, {} if payload is None else payload)
        names = None if to is None else set(to)
        with self._lock:
            targets = [t for name, t in self._targets.items() if names is None or name in names]
        for target in targets:
            target.put(script)

    def close(self) -> None:
        """Останавливает потоки отправки."""
        with self._lock:
            targets, self._targets = list(self._targets.values()), {}
        for target in targets:
            target.stop()
