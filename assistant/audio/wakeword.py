"""Распознавание ограниченного набора слов через Vosk: имя ассистента, «да/нет».

Распознаватель с грамматикой ["джарвис", "[unk]"] лёгкий и работает без интернета.
С маленькой грамматикой Vosk склонен «натягивать» любую речь на известные слова:
«Который час?» он слышал как «джарвис» с уверенностью 1.0. Поэтому:
* в грамматику имени добавлены частые слова-«приманки» (DECOY_WORDS): обычной речи
  есть на что лечь, кроме имени;
* решение принимается только по окончательному результату фразы и только если
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
from assistant.frozen import run_hint

log = logging.getLogger(__name__)

# Частые слова русской речи и слова команд. Слов, похожих на имя по звучанию
# (начало «джа-», «жа-», «ча-», «да-»: «давай», «жарко», «чай»), здесь нет —
# иначе они перехватывали бы само имя.
DECOY_WORDS = (
    "ты", "он", "она", "мы", "вы", "они", "это", "этот", "эта", "то", "тот", "что", "как", "так",
    "вот", "все", "не", "нет", "ну", "но", "или", "если", "когда", "где", "куда", "почему", "зачем",
    "кто", "какой", "какая", "который", "сколько", "там", "тут", "здесь", "сейчас", "потом", "уже",
    "еще", "тоже", "очень", "можно", "нужно", "надо", "будет", "был", "была", "были", "есть", "мне",
    "меня", "тебе", "тебя", "нам", "вам", "его", "ее", "их", "мой", "твой", "наш", "сам", "себя",
    "один", "два", "три", "много", "мало", "больше", "меньше", "хорошо", "плохо", "ладно", "конечно",
    "просто", "может", "быть", "хочу", "знаю", "думаю", "могу", "сказал", "сказала", "говорит",
    "иди", "пойдем", "смотри", "слушай", "подожди", "позвони", "купи", "сделай", "открой", "закрой",
    "включи", "выключи", "найди", "покажи", "запусти", "поставь", "убери", "верни", "время", "час",
    "день", "сегодня", "завтра", "вчера", "утром", "вечером", "ночью", "дома", "дом", "работа",
    "работу", "магазин", "хлеб", "кино", "фильм", "музыка", "музыку", "видео", "телефон", "компьютер",
    "окно", "свет", "погода", "дела", "деньги", "минуту", "привет", "пока", "спасибо", "пожалуйста",
    "громче", "тише", "пауза", "стоп", "дальше", "громкость", "звук",
)  # fmt: skip


class VoskModelError(RuntimeError):
    """Модель Vosk не найдена или не загрузилась."""


def load_vosk_model(path: Path) -> Any:
    """Загружает модель Vosk (vosk-model-small-ru).

    Raises:
        VoskModelError: папки модели нет или она повреждена.
    """
    if not path.is_dir():
        raise VoskModelError(f"Модель Vosk не найдена: {path}. Скачайте её: {run_hint('--download-models')}")
    import vosk

    vosk.SetLogLevel(-1)
    try:
        return vosk.Model(str(path))
    except Exception as exc:
        raise VoskModelError(
            f"Не удалось загрузить модель Vosk из {path}: {exc}. Удалите эту папку — "
            "при следующем запуске модель скачается заново."
        ) from exc


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
        decoys: слова-«приманки»: Vosk может их распознать, но искать их никто не будет.
            Слова, которых нет в словаре модели, пропускаются молча.
    """

    def __init__(self, model: Any, phrases: Iterable[str], decoys: Iterable[str] = ()) -> None:
        import vosk

        wanted = [p for p in dict.fromkeys(_normalize(p) for p in phrases) if p]
        self.missing_words = sorted(
            {word for phrase in wanted for word in phrase.split() if model.vosk_model_find_word(word) < 0}
        )
        self.phrases = [p for p in wanted if not set(p.split()) & set(self.missing_words)]
        if self.missing_words:
            log.error("Слов %s нет в словаре модели Vosk — они не будут распознаваться", self.missing_words)
        own_words = {word for phrase in wanted for word in phrase.split()}
        self.decoys = [
            d for d in dict.fromkeys(_normalize(d) for d in decoys)
            if d and d not in own_words and model.vosk_model_find_word(d) >= 0
        ]  # fmt: skip
        grammar = json.dumps([*self.phrases, *self.decoys, "[unk]"], ensure_ascii=False)
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


def phrase_spotter(model: Any, phrases: Iterable[str], exclude_decoys: Iterable[str] = ()) -> GrammarSpotter:
    """Распознаватель заданных фраз среди слов-приманок (DECOY_WORDS).

    Если искомое слово совпадает с приманкой («компьютер», «нет»), оно остаётся только искомым.
    exclude_decoys — приманки, слишком похожие на искомое слово: короткое «стоп» Vosk
    принимал за «тот» и «сам».
    """
    excluded = set(exclude_decoys)
    return GrammarSpotter(model, phrases, decoys=[word for word in DECOY_WORDS if word not in excluded])


def wake_spotter(model: Any, name: str) -> GrammarSpotter:
    """Распознаватель имени ассистента."""
    return phrase_spotter(model, [name])
