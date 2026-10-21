"""Интерфейсы озвучки.

На этапе 1 ответы выводятся в консоль; движки TTS (Piper, Silero, edge-tts, SAPI)
появятся на этапе 2 и будут реализовывать эти же интерфейсы.
"""

from __future__ import annotations

from typing import Protocol


class Speaker(Protocol):
    """То, что умеет произнести (или показать) ответ ассистента."""

    def say(self, text: str) -> None:
        """Произносит текст; не должен надолго блокировать вызывающий поток."""
        ...
