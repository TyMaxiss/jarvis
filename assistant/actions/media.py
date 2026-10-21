"""Управление воспроизведением через системные медиаклавиши."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from assistant.actions.input import tap_key
from assistant.actions.registry import ActionRegistry

_MEDIA_KEYS = {
    "play_pause": (0xB3, "Пауза или воспроизведение"),
    "next": (0xB0, "Следующий трек"),
    "prev": (0xB1, "Предыдущий трек"),
    "stop": (0xB2, "Остановлено"),
}


def register(registry: ActionRegistry) -> None:
    """Регистрирует медиадействия."""

    @registry.action(
        "media",
        "Управление воспроизведением музыки и видео: пауза/продолжить, следующий, предыдущий трек, стоп",
    )
    def media(
        action: Annotated[
            Literal["play_pause", "next", "prev", "stop"],
            Field(description="play_pause — пауза или продолжить, next, prev, stop"),
        ],
    ) -> str:
        vk, text = _MEDIA_KEYS[action]
        tap_key(vk)
        return text
