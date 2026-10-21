"""Подтверждение рискованных действий.

На этапе 1 подтверждение вводится в консоли. Голосовое подтверждение (Vosk с грамматикой
«да/нет») и кнопки в оверлее появятся на следующих этапах и реализуют тот же протокол.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

log = logging.getLogger(__name__)

YES_WORDS = frozenset({"да", "д", "y", "yes", "ага", "подтверждаю", "конечно", "давай"})


class Confirmer(Protocol):
    """Спрашивает пользователя, выполнять ли действие."""

    def confirm(self, question: str) -> bool:
        """Возвращает True только при явном согласии; молчание и ошибки — отмена."""
        ...


class ConsoleConfirmer:
    """Подтверждение вводом в консоли. Пустой ввод означает отмену.

    Args:
        ask: функция чтения ответа (по умолчанию input).
    """

    def __init__(self, ask: Callable[[str], str] = input) -> None:
        self._ask = ask

    def confirm(self, question: str) -> bool:
        try:
            answer = self._ask(f"{question} (да/нет): ")
        except (EOFError, KeyboardInterrupt):
            return False
        accepted = answer.strip().lower().replace("ё", "е") in YES_WORDS
        log.info("Подтверждение «%s»: %s", question, "да" if accepted else "нет")
        return accepted


class StaticConfirmer:
    """Всегда отвечает одинаково. Для тестов и неинтерактивного режима."""

    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def confirm(self, question: str) -> bool:
        self.questions.append(question)
        return self.answer
