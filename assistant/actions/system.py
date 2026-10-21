"""Системные мелочи: время, пауза между шагами, фраза вслух, буфер обмена, звук."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from pydantic import Field

from assistant.actions.files import resolve_user_path
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.actions.shell import shorten
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

    @registry.action("clipboard_get", "Прочитать текст из буфера обмена")
    def clipboard_get() -> str:
        import pyperclip

        try:
            text = pyperclip.paste()
        except pyperclip.PyperclipException as exc:
            raise ActionError(f"Буфер обмена недоступен: {exc}") from exc
        return shorten(text) if text and text.strip() else "Буфер обмена пуст"

    @registry.action("clipboard_set", "Положить текст в буфер обмена")
    def clipboard_set(
        text: Annotated[str, Field(min_length=1, max_length=100_000, description="Что скопировать")],
    ) -> str:
        import pyperclip

        try:
            pyperclip.copy(text)
        except pyperclip.PyperclipException as exc:
            raise ActionError(f"Буфер обмена недоступен: {exc}") from exc
        return "Скопировал в буфер обмена"

    @registry.action("play_sound", "Проиграть звуковой файл WAV")
    def play_sound(
        path: Annotated[str, Field(min_length=1, description="Путь к файлу .wav")],
    ) -> str:
        file = resolve_user_path(path)
        if not file.is_file():
            raise ActionError(f"Файла {file} нет")
        if file.suffix.lower() != ".wav":
            raise ActionError("Умею проигрывать только файлы WAV")
        try:
            import winsound
        except ImportError as exc:
            raise ActionError("Проигрывать звук можно только в Windows") from exc
        winsound.PlaySound(str(file), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return f"Проигрываю {file.name}"
