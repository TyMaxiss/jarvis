"""Системные мелочи: время, пауза между шагами, произнести фразу."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from pydantic import Field

from assistant.actions.registry import ActionRegistry
from assistant.tts.base import Speaker

_WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
_MONTHS = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)  # fmt: skip


def format_now(now: datetime) -> str:
    """Время и дата по-русски: «Сейчас 14:35, вторник, 21 октября»."""
    return f"Сейчас {now:%H:%M}, {_WEEKDAYS[now.weekday()]}, {now.day} {_MONTHS[now.month - 1]}"


def register(
    registry: ActionRegistry,
    speaker: Speaker,
    clock: Callable[[], datetime] = datetime.now,
) -> None:
    """Регистрирует системные действия.

    Args:
        registry: реестр действий.
        speaker: кто произносит фразы для действия say.
        clock: источник текущего времени (подменяется в тестах).
    """

    @registry.action("get_time", "Узнать текущее время и дату")
    def get_time() -> str:
        return format_now(clock())

    @registry.action("wait", "Подождать указанное число секунд (например, пока откроется программа)")
    def wait(
        seconds: Annotated[float, Field(gt=0, le=60, description="Сколько секунд ждать, до 60")],
    ) -> str:
        time.sleep(seconds)
        return f"Подождал {seconds:g} с"

    @registry.action("say", "Произнести фразу вслух")
    def say(text: Annotated[str, Field(min_length=1, description="Что сказать")]) -> str:
        speaker.say(text)
        return "Сказано"
