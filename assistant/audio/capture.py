"""Захват звука с микрофона: 16 кГц, моно, int16, блоки по 512 сэмплов (32 мс).

Колбэк PortAudio работает в своём потоке и только кладёт блоки в очередь;
вся обработка идёт в потоке голосового цикла.
"""

from __future__ import annotations

import logging
import queue
from typing import Any, Protocol

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
BLOCK_SIZE = 512  # 32 мс — столько ждёт на входе Silero VAD
BLOCK_MS = BLOCK_SIZE * 1000 // SAMPLE_RATE


class AudioSource(Protocol):
    """Источник блоков звука (микрофон или, в тестах, заранее записанный звук)."""

    def read(self, timeout: float) -> np.ndarray | None:
        """Следующий блок int16 длиной BLOCK_SIZE или None, если за timeout ничего не пришло."""
        ...

    def flush(self) -> None:
        """Выбрасывает накопившиеся блоки (например, то, что прозвучало из колонок)."""
        ...


class CaptureError(RuntimeError):
    """Не удалось открыть микрофон."""


def list_input_devices() -> list[tuple[int, str]]:
    """Номера и названия устройств записи."""
    import sounddevice as sd

    return [
        (index, str(device["name"]))
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


def resolve_device(device: int | str | None) -> int | None:
    """Номер устройства по номеру или части названия (None — устройство по умолчанию).

    Raises:
        CaptureError: устройство с таким названием не найдено.
    """
    if device is None or isinstance(device, int):
        return device
    needle = device.lower()
    for index, name in list_input_devices():
        if needle in name.lower():
            return index
    raise CaptureError(f"Микрофон «{device}» не найден. Список: python -m assistant --list-devices")


def block_level(block: np.ndarray) -> float:
    """Громкость блока int16 для индикатора: 0 — тишина, 1 — громко (речь вблизи микрофона).

    Берётся среднеквадратичное значение; обычная речь даёт 0,02–0,1 от полной шкалы,
    поэтому оно усиливается в 8 раз.
    """
    if not len(block):
        return 0.0
    rms = float(np.sqrt(np.mean(np.square(block.astype(np.float32))))) / 32768
    return min(1.0, rms * 8)


class MicrophoneCapture:
    """Запись с микрофона в очередь блоков.

    Args:
        device: номер или часть названия микрофона; None — по умолчанию.
        max_queue_seconds: сколько звука держать в очереди, если её не успевают разбирать.
    """

    def __init__(self, device: int | str | None = None, max_queue_seconds: float = 30.0) -> None:
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=int(max_queue_seconds * SAMPLE_RATE / BLOCK_SIZE)
        )
        self._stream: Any = None
        self.dropped_blocks = 0
        self.level = 0.0  # громкость последнего блока, 0–1 (для индикатора в интерфейсе)

    def start(self) -> None:
        """Открывает микрофон.

        Raises:
            CaptureError: микрофона нет или он занят.
        """
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=1,
                dtype="int16",
                device=resolve_device(self._device),
                callback=self._callback,
            )
            self._stream.start()
        except CaptureError:
            raise
        except Exception as exc:
            raise CaptureError(f"Не удалось открыть микрофон: {exc}") from exc
        log.info("Микрофон открыт: %s", self._stream.device)

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Колбэк PortAudio: только копирует блок в очередь."""
        if status:
            log.debug("Статус аудиопотока: %s", status)
        block = indata[:, 0].copy()
        self.level = block_level(block)
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Цикл не успевает (например, долго распознаёт): старый звук выбрасываем.
            self.dropped_blocks += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(block)
            except (queue.Empty, queue.Full):
                pass

    def read(self, timeout: float) -> np.ndarray | None:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def flush(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        """Закрывает микрофон."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class ArraySource:
    """Источник из готового массива int16 — для тестов и проверки на записях.

    После окончания звука возвращает тишину (silence_after секунд), затем None.
    """

    def __init__(self, audio: np.ndarray, silence_after: float = 2.0) -> None:
        pad = np.zeros(int(silence_after * SAMPLE_RATE), dtype=np.int16)
        data = np.concatenate([audio.astype(np.int16), pad])
        usable = len(data) // BLOCK_SIZE * BLOCK_SIZE
        self._blocks = list(data[:usable].reshape(-1, BLOCK_SIZE))
        self._position = 0

    @property
    def exhausted(self) -> bool:
        """Весь звук прочитан."""
        return self._position >= len(self._blocks)

    def read(self, timeout: float) -> np.ndarray | None:
        if self.exhausted:
            return None
        block = self._blocks[self._position]
        self._position += 1
        return block

    def flush(self) -> None:
        """Записи не «накапливаются», поэтому сбрасывать нечего."""


def to_int16(audio: np.ndarray) -> np.ndarray:
    """float [-1, 1] → int16."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def resample(audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Передискретизация через БПФ (без наложения частот) — для синтезированной речи в тестах."""
    if source_rate == target_rate or not len(audio):
        return audio
    target_len = int(round(len(audio) * target_rate / source_rate))
    spectrum = np.fft.rfft(audio.astype(np.float64))
    bins = target_len // 2 + 1
    if bins <= len(spectrum):
        spectrum = spectrum[:bins]
    else:
        spectrum = np.concatenate([spectrum, np.zeros(bins - len(spectrum), dtype=spectrum.dtype)])
    result = np.fft.irfft(spectrum, target_len) * (target_len / len(audio))
    return result.astype(audio.dtype)
