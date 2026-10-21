"""Подтверждение рискованных действий.

Фразы, введённые текстом, подтверждаются в консоли, голосовые — голосом (Vosk с
грамматикой «да/нет», молчание — отмена). Кнопки в оверлее появятся на этапе 4.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

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


# «Давай» в голосовых ответах нет: Vosk слышал его в «Сделай громче» — ложное «да» опаснее «нет».
YES_PHRASES = ("да", "подтверждаю", "конечно", "выполняй")
NO_PHRASES = ("нет", "отмена", "не надо", "стоп", "отменить")
_YES_WORDS = frozenset(word for phrase in YES_PHRASES for word in phrase.split())


class VoiceConfirmer:
    """Подтверждение голосом: ассистент задаёт вопрос и слушает «да» или «нет» (Vosk).

    Молчание в течение timeout секунд означает отмену.

    Args:
        speech: озвучка (say, wait_idle).
        source: микрофон.
        spotter_factory: создаёт распознаватель YES_PHRASES + NO_PHRASES (со словами-приманками).
        timeout: сколько секунд ждать ответа.
        min_confidence: минимальная уверенность Vosk в ответе; неуверенное «да» — не согласие.
    """

    def __init__(
        self,
        speech: Any,
        source: Any,
        spotter_factory: Callable[[], Any],
        timeout: float = 5.0,
        min_confidence: float = 0.7,
        block_ms: int = 32,
    ) -> None:
        self._speech = speech
        self._source = source
        self._spotter_factory = spotter_factory
        self._timeout_ms = int(timeout * 1000)
        self._min_confidence = min_confidence
        self._block_ms = block_ms

    def confirm(self, question: str) -> bool:
        self._speech.say(question)
        self._speech.wait_idle(timeout=30)
        self._source.flush()
        spotter = self._spotter_factory()
        heard_ms = 0
        answer = None
        while heard_ms < self._timeout_ms and answer is None:
            block = self._source.read(timeout=0.2)
            if block is None:
                heard_ms += 200
                continue
            heard_ms += self._block_ms
            result = spotter.accept(block)
            if result is not None:
                answer = self._answer(result)
        if answer is None:
            answer = self._answer(spotter.finish())
        log.info("Голосовое подтверждение «%s»: %s", question, answer or "нет ответа")
        return answer in YES_PHRASES

    def _answer(self, result: Any) -> str | None:
        """«Нет» важнее «да»: если прозвучало и то и другое, действие не выполняется.

        «Да» засчитывается, только если ответ состоит из одних слов согласия — без
        слов-приманок и [unk]: Vosk с маленькой грамматикой находил «да» в «хорошая
        погода» (уверенность 0.62 при пороге 0.7). Тогда ответа нет, и подтверждение
        слушает дальше.
        """
        log.debug("Ответ Vosk: %s", result.describe())
        no = result.find(NO_PHRASES, self._min_confidence)
        if no or any(word not in _YES_WORDS for word, _ in result.words):
            return no
        return result.find(YES_PHRASES, self._min_confidence)
