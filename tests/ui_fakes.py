"""Подделки для тестов интерфейса: окна, события, громкость, агент."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

from assistant.config import LlmConfig


class RecordingBus:
    """Вместо EventBus: события копятся в списке (без потоков)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, tuple[str, ...] | None]] = []
        self.windows: dict[str, Callable[[str], None]] = {}

    def attach(self, name: str, send: Callable[[str], None]) -> None:
        self.windows[name] = send

    def attached(self, name: str) -> bool:
        return name in self.windows

    def emit(self, kind: str, payload: Any = None, *, to: Iterable[str] | None = None) -> None:
        # Как настоящий EventBus: событие должно превращаться в JSON.
        json.dumps(payload, ensure_ascii=False)
        self.events.append((kind, payload, tuple(to) if to is not None else None))

    def close(self) -> None:
        pass

    def of(self, kind: str) -> list[Any]:
        """Данные всех событий этого типа."""
        return [payload for event, payload, _ in self.events if event == kind]


class FakeHost:
    """Вместо окон pywebview: запоминает, что с окнами делали."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.shown: set[str] = {"main"}
        self.scripts: dict[str, list[str]] = {}

    def show(self, name: str) -> None:
        self.calls.append(("show", name))
        self.shown.add(name)

    def hide(self, name: str) -> None:
        self.calls.append(("hide", name))
        self.shown.discard(name)

    def visible(self, name: str) -> bool:
        return name in self.shown

    def minimize(self) -> None:
        self.calls.append(("minimize",))

    def toggle_maximize(self) -> None:
        self.calls.append(("maximize",))

    def resize_main(self, width: int, height: int) -> None:
        self.calls.append(("resize", width, height))

    def sender(self, name: str) -> Callable[[str], None]:
        return lambda script: self.scripts.setdefault(name, []).append(script)

    def quit(self) -> None:
        self.calls.append(("quit",))


class FakeVolume:
    """Громкость системы."""

    def __init__(self, percent: int = 40) -> None:
        self.percent = percent
        self.muted = False

    def get(self) -> int:
        return self.percent

    def set(self, percent: int) -> int:
        self.percent = max(0, min(100, percent))
        return self.percent

    def is_muted(self) -> bool:
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.muted = muted


class FakeAgent:
    """Агент нейросети: помнит настройки и перенастройки."""

    def __init__(self, settings: LlmConfig) -> None:
        self.settings = settings
        self.reconfigured: list[LlmConfig] = []

    def reconfigure(self, settings: LlmConfig) -> None:
        self.settings = settings
        self.reconfigured.append(settings)


class FakeLoop:
    """Голосовой цикл: пауза и push-to-talk."""

    def __init__(self) -> None:
        self.paused = False
        self.listen_requests = 0

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def push_to_talk(self) -> None:
        self.listen_requests += 1


class FakeCapture:
    level = 0.42


class FakeVoice:
    """Запущенный голос (VoiceRuntime)."""

    def __init__(self) -> None:
        self.loop = FakeLoop()
        self.capture = FakeCapture()
        self.confirmer = None
        self.wake_word = True
        self.closed = False

    def close(self) -> None:
        self.closed = True
