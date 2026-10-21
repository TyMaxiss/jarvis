"""Распознавание ограниченного набора слов через Vosk: имя ассистента, «да/нет».

Распознаватель с грамматикой ["джарвис", "[unk]"] лёгкий и работает без интернета.
С маленькой грамматикой Vosk склонен «натягивать» любую речь на известные слова,
поэтому решение принимается только по окончательному результату фразы и только если
уверенность (conf) каждого слова не ниже порога.

Слова грамматики должны быть в словаре модели: отсутствующие Vosk молча игнорирует,
поэтому они проверяются заранее (missing_words).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class SpotResult:
    """Окончательный результат Vosk по фразе: слова с уверенностью."""

    text: str = ""
    words: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def find(self, phrases: Sequence[str], min_conf: float) -> str | None:
        """Первая фраза, все слова которой распознаны подряд с уверенностью не ниже min_conf."""
        for phrase in phrases:
            parts = phrase.split()
            for start in range(len(self.words) - len(parts) + 1):
                window = self.words[start : start + len(parts)]
                if [w for w, _ in window] == parts and all(conf >= min_conf for _, conf in window):
                    return phrase
        return None

    @property
    def known_words(self) -> list[str]:
        """Распознанные слова без [unk]."""
        return [w for w, _ in self.words if w != "[unk]"]

    @property
    def has_unknown(self) -> bool:
        """Были ли во фразе слова не из грамматики."""
        return any(w == "[unk]" for w, _ in self.words)

    def __add__(self, other: SpotResult) -> SpotResult:
        """Склеивает результаты, если Vosk разбил одну фразу на части."""
        return SpotResult(f"{self.text} {other.text}".strip(), self.words + other.words)

    def describe(self) -> str:
        """Строка для лога: слова с уверенностью."""
        return " ".join(f"{w}({conf:.2f})" for w, conf in self.words) or "—"


def parse_result(raw: str) -> SpotResult:
    """Разбирает JSON результата Vosk (с SetWords(True))."""
    data = json.loads(raw)
    words = tuple(
        (_normalize(str(item["word"])), float(item.get("conf", 0.0))) for item in data.get("result", [])
    )
    return SpotResult(_normalize(data.get("text", "")), words)


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
        self._recognizer.SetWords(True)

    @property
    def usable(self) -> bool:
        """Есть хотя бы одна фраза, которую модель может распознать."""
        return bool(self.phrases)

    def accept(self, block: np.ndarray) -> SpotResult | None:
        """Подаёт блок звука. Возвращает результат, когда Vosk сам решил, что фраза кончилась."""
        if self._recognizer.AcceptWaveform(block.astype(np.int16).tobytes()):
            return parse_result(self._recognizer.Result())
        return None

    def finish(self) -> SpotResult:
        """Завершает фразу (после паузы) и возвращает окончательный результат."""
        result = parse_result(self._recognizer.FinalResult())
        self._recognizer.Reset()
        return result

    def reset(self) -> None:
        """Забывает услышанное."""
        self._recognizer.Reset()
