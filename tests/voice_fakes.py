"""Поддельные компоненты для тестов голосового цикла (без микрофона и моделей).

Звук строится из блоков с постоянным значением: значение кодирует, что «сказано».
FakeVAD считает речью всё громче 1000, FakeSTT возвращает текст по самому частому
значению во фразе, FakeSpotter «слышит» имя, если во фразе были блоки со значением WAKE.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from assistant.audio.capture import BLOCK_MS, BLOCK_SIZE, SAMPLE_RATE
from assistant.audio.wakeword import SpotResult
from assistant.stt.whisper_stt import Transcript

WAKE = 7777


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
    """Как Vosk с грамматикой: в конце фразы «слышит» имя, если были блоки WAKE.

    conf — уверенность, с которой «распознано» имя; unk=False — остальные слова
    не отмечаются как [unk] (так Vosk иногда «проглатывает» команду после имени).
    """

    def __init__(self, conf: float = 0.95, word: str = "джарвис", unk: bool = True) -> None:
        self.conf = conf
        self.word = word
        self.unk = unk
        self._seen_wake = False
        self._seen_other = False

    def accept(self, block: np.ndarray) -> SpotResult | None:
        if int(block[0]) == WAKE:
            self._seen_wake = True
        elif int(block[0]):
            self._seen_other = True
        return None

    def finish(self) -> SpotResult:
        words = ([(self.word, self.conf)] if self._seen_wake else []) + (
            [("[unk]", 1.0)] if self._seen_other and self.unk else []
        )
        self.reset()
        return SpotResult(" ".join(w for w, _ in words), tuple(words))

    def reset(self) -> None:
        self._seen_wake = self._seen_other = False


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
    """Блоки из массива; после конца — None."""

    def __init__(self, audio: np.ndarray) -> None:
        self._blocks = blocks(audio)
        self.flushes = 0

    def read(self, timeout: float) -> np.ndarray | None:
        return self._blocks.pop(0) if self._blocks else None

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
