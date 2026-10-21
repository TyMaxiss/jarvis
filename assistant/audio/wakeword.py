"""Распознавание ограниченного набора слов через Vosk: имя ассистента, «да/нет», «стоп».

Распознаватель с грамматикой ["джарвис", "[unk]"] лёгкий и работает без интернета.
Слова грамматики должны быть в словаре модели: отсутствующие Vosk молча игнорирует,
поэтому они проверяются заранее (missing_words).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from assistant.audio.capture import SAMPLE_RATE

log = logging.getLogger(__name__)


class VoskModelError(RuntimeError):
    """Модель Vosk не найдена или не загрузилась."""


def load_vosk_model(path: Path) -> Any:
    """Загружает модель Vosk (vosk-model-small-ru).

    Raises:
        VoskModelError: папки модели нет или она повреждена.
    """
    if not path.is_dir():
        raise VoskModelError(
            f"Модель Vosk не найдена: {path}. Скачайте её: uv run python -m assistant --download-models"
        )
    import vosk

    vosk.SetLogLevel(-1)
    try:
        return vosk.Model(str(path))
    except Exception as exc:
        raise VoskModelError(f"Не удалось загрузить модель Vosk из {path}: {exc}") from exc


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


class GrammarSpotter:
    """Ищет в потоке звука одну из заданных фраз.

    Args:
        model: модель Vosk.
        phrases: фразы грамматики («джарвис», «да», «не надо»…).
    """

    def __init__(self, model: Any, phrases: Iterable[str]) -> None:
        import vosk

        wanted = [p for p in dict.fromkeys(_normalize(p) for p in phrases) if p]
        self.missing_words = sorted(
            {word for phrase in wanted for word in phrase.split() if model.vosk_model_find_word(word) < 0}
        )
        self.phrases = [p for p in wanted if not set(p.split()) & set(self.missing_words)]
        if self.missing_words:
            log.error("Слов %s нет в словаре модели Vosk — они не будут распознаваться", self.missing_words)
        grammar = json.dumps([*self.phrases, "[unk]"], ensure_ascii=False)
        self._recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE, grammar)
        self.last_text = ""  # последний результат Vosk целиком, например «джарвис [unk]»

    @property
    def usable(self) -> bool:
        """Есть хотя бы одна фраза, которую модель может распознать."""
        return bool(self.phrases)

    def accept(self, block: np.ndarray) -> str | None:
        """Подаёт блок звука. Возвращает найденную фразу или None."""
        data = block.astype(np.int16).tobytes()
        if self._recognizer.AcceptWaveform(data):
            text = json.loads(self._recognizer.Result()).get("text", "")
        else:
            text = json.loads(self._recognizer.PartialResult()).get("partial", "")
        if text:
            self.last_text = text
        return self._find(text)

    def finish(self) -> str | None:
        """Завершает фразу (после паузы) и возвращает найденное в окончательном результате."""
        text = json.loads(self._recognizer.FinalResult()).get("text", "")
        self._recognizer.Reset()
        self.last_text = text
        return self._find(text)

    def reset(self) -> None:
        """Забывает услышанное."""
        self._recognizer.Reset()
        self.last_text = ""

    def _find(self, text: str) -> str | None:
        padded = f" {_normalize(text)} "
        for phrase in self.phrases:
            if f" {phrase} " in padded:
                return phrase
        return None
