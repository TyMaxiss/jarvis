"""Поддельные компоненты для тестов голосового цикла (без микрофона и моделей).

Звук строится из блоков с постоянным значением: значение кодирует, что «сказано».
FakeVAD считает речью всё громче 1000, FakeSTT возвращает текст по самому частому
значению во фразе, FakeSpotter «слышит» имя, если во фразе были блоки со значением WAKE
(или «стоп» — для блоков STOP).
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from assistant.audio.capture import BLOCK_MS, BLOCK_SIZE, SAMPLE_RATE
from assistant.audio.wakeword import SpotResult
from assistant.stt.whisper_stt import Transcript

WAKE = 7777
STOP = 8888


def silence(ms: int) -> np.ndarray:
    return np.zeros(SAMPLE_RATE * ms // 1000, dtype=np.int16)


def speech(value: int, ms: int) -> np.ndarray:
    return np.full(SAMPLE_RATE * ms // 1000, value, dtype=np.int16)


def blocks(audio: np.ndarray) -> list[np.ndarray]:
    usable = len(audio) // BLOCK_SIZE * BLOCK_SIZE
    return list(audio[:usable].reshape(-1, BLOCK_SIZE))


class FakeVAD:
    def __call__(self, block: np.ndarray) -> float:
        return 0.9 if abs(int(block[0])) > 1000 else 0.0

    def reset(self) -> None:
        pass


class FakeSTT:
    """Текст по самому частому ненулевому значению во фразе.

    speech_ms — сколько речи (ненулевого звука) было в каждой распознанной фразе.
    """

    def __init__(self, script: dict[int, str]) -> None:
        self.script = script
        self.calls: list[int] = []
        self.speech_ms: list[int] = []

    def transcribe(self, audio: np.ndarray) -> Transcript:
        self.speech_ms.append(int(np.count_nonzero(audio)) * 1000 // SAMPLE_RATE)
        values = Counter(int(v) for v in audio[::BLOCK_SIZE] if v)
        if not values:
            return Transcript("", 1.0)
        value = values.most_common(1)[0][0]
        self.calls.append(value)
        return Transcript(self.script.get(value, ""), 1.0)


class FakeSpotter:
    """Как Vosk с грамматикой: в конце фразы «слышит» слово, если были блоки value.

    conf — уверенность, с которой «распознано» слово; unk=False — остальные слова
    не отмечаются как [unk] (так Vosk иногда «проглатывает» команду после имени).
    Время слова — с первого его блока, секунды от начала фразы (как у GrammarSpotter).
    """

    def __init__(
        self, conf: float = 0.95, word: str = "джарвис", unk: bool = True, value: int = WAKE
    ) -> None:
        self.conf = conf
        self.word = word
        self.unk = unk
        self.value = value
        self.reset()

    def accept(self, block: np.ndarray) -> SpotResult | None:
        at = self._blocks * BLOCK_MS / 1000
        self._blocks += 1
        if int(block[0]) == self.value:
            self._wake_at = at if self._wake_at is None else self._wake_at
        elif int(block[0]):
            self._other_at = at if self._other_at is None else self._other_at
        return None

    def finish(self) -> SpotResult:
        end = self._blocks * BLOCK_MS / 1000
        words: list[tuple[tuple[str, float], tuple[float, float]]] = []
        if self._wake_at is not None:
            words.append(((self.word, self.conf), (self._wake_at, end)))
        if self._other_at is not None and self.unk:
            words.append((("[unk]", 1.0), (self._other_at, end)))
        words.sort(key=lambda item: item[1][0])
        self.reset()
        return SpotResult(
            " ".join(word for (word, _), _ in words), tuple(w for w, _ in words), tuple(t for _, t in words)
        )

    def reset(self) -> None:
        self._blocks = 0
        self._wake_at: float | None = None
        self._other_at: float | None = None


class WordsSpotter:
    """Как Vosk с грамматикой: слово для каждого значения блоков (по порядку), остальное — [unk].

    chunks=True — как Vosk на длинной речи без пауз: кусок фразы отдаётся, как только
    сменилось значение блоков, а не в конце фразы.
    """

    def __init__(self, words: dict[int, tuple[str, float]], chunks: bool = False) -> None:
        self.words = words
        self.chunks = chunks
        self._values: list[int] = []

    def accept(self, block: np.ndarray) -> SpotResult | None:
        value = int(block[0])
        if not value or (self._values and self._values[-1] == value):
            return None
        if self.chunks and self._values:
            result = self._result()
            self._values = [value]
            return result
        self._values.append(value)
        return None

    def finish(self) -> SpotResult:
        result = self._result()
        self.reset()
        return result

    def reset(self) -> None:
        self._values = []

    def _result(self) -> SpotResult:
        words = tuple(self.words.get(value, ("[unk]", 1.0)) for value in self._values)
        return SpotResult(" ".join(word for word, _ in words), words)


class FakeSpeech:
    """Озвучка: «говорит» заданное число блоков после каждой фразы или сигнала."""

    def __init__(self, speak_blocks: int = 0) -> None:
        self.speak_blocks = speak_blocks
        self.remaining = 0
        self.said: list[str] = []
        self.sounds = 0
        self.stops = 0

    @property
    def speaking(self) -> bool:
        if self.remaining > 0:
            self.remaining -= 1
            return True
        return False

    def say(self, text: str) -> None:
        self.said.append(text)
        self.remaining += self.speak_blocks

    def play_sound(self, audio: np.ndarray, samplerate: int) -> None:
        self.sounds += 1
        self.remaining += max(1, len(audio) * 1000 // samplerate // BLOCK_MS)

    def stop(self) -> None:
        self.stops += 1
        self.remaining = 0

    def wait_idle(self, timeout: float | None = None) -> bool:
        self.remaining = 0
        return True


class FakeSource:
    """Блоки из массива; после конца — None. reads — сколько блоков прочитано."""

    def __init__(self, audio: np.ndarray) -> None:
        self._blocks = blocks(audio)
        self.flushes = 0
        self.reads = 0

    @property
    def exhausted(self) -> bool:
        return not self._blocks

    def read(self, timeout: float) -> np.ndarray | None:
        if not self._blocks:
            return None
        self.reads += 1
        return self._blocks.pop(0)

    def flush(self) -> None:
        self.flushes += 1


class RecordingStatus:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.results: list[object] = []

    def voice_status(self, state: str, detail: str = "") -> None:
        self.events.append((state, detail))

    def voice_result(self, result: object) -> None:
        self.results.append(result)


class FakeDictation:
    """Vosk по ходу речи: слово за каждые words_every блоков речи (значение блока > 1000)."""

    def __init__(self, words: tuple[str, ...] = ("сделай", "громче"), words_every: int = 5) -> None:
        self._words = words
        self._every = words_every
        self._blocks = 0
        self.resets = 0

    def accept(self, block: np.ndarray) -> str | None:
        if abs(int(block[0])) <= 1000:
            return None
        self._blocks += 1
        if self._blocks % self._every:
            return None
        return " ".join(self._words[: self._blocks // self._every]) or None

    def reset(self) -> None:
        self._blocks = 0
        self.resets += 1
