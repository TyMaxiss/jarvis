"""Определение речи (Silero VAD через onnxruntime, без torch) и границ фразы.

Модель silero_vad_v6.onnx входит в пакет faster-whisper, отдельно её скачивать не нужно.
Она принимает блок из 512 сэмплов вместе с 64 последними сэмплами предыдущего блока
и состояние LSTM (h, c), которое переносится между вызовами.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from assistant.audio.capture import BLOCK_MS, BLOCK_SIZE

log = logging.getLogger(__name__)

_CONTEXT = 64
_STATE_SHAPE = (1, 1, 128)


def default_model_path() -> Path:
    """Путь к модели Silero VAD из пакета faster-whisper."""
    from faster_whisper.utils import get_assets_path

    return Path(get_assets_path()) / "silero_vad_v6.onnx"


class SileroVAD:
    """Вероятность речи для каждого блока 32 мс (потоковый режим).

    Args:
        model_path: путь к ONNX-модели; по умолчанию — из faster-whisper.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        self._session = onnxruntime.InferenceSession(
            str(model_path or default_model_path()),
            providers=["CPUExecutionProvider"],
            sess_options=options,
        )
        self.reset()

    def reset(self) -> None:
        """Сбрасывает состояние (между фразами и после паузы в прослушивании)."""
        self._h = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, block: np.ndarray) -> float:
        """Вероятность речи (0–1) в блоке из 512 сэмплов int16 или float."""
        if block.shape != (BLOCK_SIZE,):
            raise ValueError(f"VAD ждёт блок из {BLOCK_SIZE} сэмплов, получено {block.shape}")
        samples = block.astype(np.float32)
        if block.dtype == np.int16:
            samples /= 32768.0
        samples = samples.reshape(1, BLOCK_SIZE)
        output, self._h, self._c = self._session.run(
            None, {"input": np.concatenate([self._context, samples], axis=1), "h": self._h, "c": self._c}
        )
        self._context = samples[:, -_CONTEXT:]
        return float(np.asarray(output).ravel()[0])


class SegmentEvent(Enum):
    """Что произошло после очередного блока."""

    NONE = "none"
    START = "start"  # началась речь
    END = "end"  # фраза закончилась, звук в segment_audio
    DISCARD = "discard"  # был короткий шум, не речь


@dataclass(frozen=True)
class SegmenterSettings:
    """Пороги определения фразы."""

    threshold: float = 0.5  # начало речи
    neg_threshold: float = 0.35  # продолжение речи (гистерезис)
    min_silence_ms: int = 700  # пауза, после которой фраза закончена
    min_speech_ms: int = 200  # короче — считается шумом
    pre_roll_ms: int = 320  # сколько звука до начала речи сохранить
    max_seconds: float = 15.0


class SpeechSegmenter:
    """Находит начало и конец фразы по вероятностям VAD.

    feed() вызывается для каждого блока. После события END звук фразы (с небольшим
    запасом до начала речи) лежит в segment_audio.
    """

    def __init__(self, settings: SegmenterSettings | None = None) -> None:
        self.settings = settings or SegmenterSettings()
        self._pre_roll: collections.deque[np.ndarray] = collections.deque(
            maxlen=max(1, self.settings.pre_roll_ms // BLOCK_MS)
        )
        self.segment_audio: np.ndarray = np.zeros(0, dtype=np.int16)
        self.speech_span_ms = 0  # от начала речи до последнего речевого блока, с паузами внутри
        self.truncated = False
        self.reset()

    def reset(self) -> None:
        """Забывает текущую фразу."""
        self._pre_roll.clear()
        self._buffer: list[np.ndarray] = []
        self._in_speech = False
        self._silence_blocks = 0
        self._speech_blocks = 0
        self._start_block = 0  # номер первого речевого блока в _buffer (до него — запас)

    @property
    def in_speech(self) -> bool:
        """Сейчас идёт фраза."""
        return self._in_speech

    def current_audio(self) -> np.ndarray:
        """Звук текущей (ещё не законченной) фразы."""
        return np.concatenate(self._buffer) if self._buffer else np.zeros(0, dtype=np.int16)

    def feed(self, block: np.ndarray, probability: float) -> SegmentEvent:
        """Учитывает очередной блок и его вероятность речи."""
        s = self.settings
        if not self._in_speech:
            self._pre_roll.append(block)
            if probability >= s.threshold:
                self._in_speech = True
                self._buffer = list(self._pre_roll)
                self._start_block = len(self._buffer) - 1
                self._silence_blocks = 0
                self._speech_blocks = 1
                return SegmentEvent.START
            return SegmentEvent.NONE

        self._buffer.append(block)
        if probability >= s.neg_threshold:
            self._silence_blocks = 0
            if probability >= s.threshold:
                self._speech_blocks += 1
        else:
            self._silence_blocks += 1

        too_long = len(self._buffer) * BLOCK_MS >= s.max_seconds * 1000
        if self._silence_blocks * BLOCK_MS >= s.min_silence_ms or too_long:
            return self._finish(truncated=too_long and self._silence_blocks == 0)
        return SegmentEvent.NONE

    def _finish(self, truncated: bool) -> SegmentEvent:
        """Завершает фразу: короткий всплеск — шум, иначе — готовая фраза."""
        speech_ms = self._speech_blocks * BLOCK_MS
        span_ms = (len(self._buffer) - self._start_block - self._silence_blocks) * BLOCK_MS
        # Хвост тишины не нужен распознавателю — оставляем ~200 мс.
        keep_tail = max(0, self._silence_blocks - 200 // BLOCK_MS)
        blocks = self._buffer[: len(self._buffer) - keep_tail] if keep_tail else self._buffer
        audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.int16)
        self.reset()
        if speech_ms < self.settings.min_speech_ms:
            return SegmentEvent.DISCARD
        self.segment_audio = audio
        self.speech_span_ms = span_ms
        self.truncated = truncated
        return SegmentEvent.END
