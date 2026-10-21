"""Воспроизведение звука через sounddevice с возможностью прервать."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class SoundDevicePlayer:
    """Проигрывает звук на устройстве вывода по умолчанию.

    Args:
        device: номер устройства вывода; None — по умолчанию.
    """

    def __init__(self, device: int | None = None) -> None:
        self._device = device
        self._stop = threading.Event()

    def reset(self) -> None:
        """Снимает флаг остановки перед следующей фразой."""
        self._stop.clear()

    def play(self, audio: np.ndarray, samplerate: int) -> bool:
        """Проигрывает звук до конца или до вызова stop(). Возвращает False при прерывании."""
        import sounddevice as sd

        if self._stop.is_set():
            return False
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        position = 0
        finished = threading.Event()

        def callback(outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            nonlocal position
            if self._stop.is_set():
                outdata.fill(0)
                raise sd.CallbackStop
            chunk = data[position : position + frames]
            outdata[: len(chunk), 0] = chunk
            outdata[len(chunk) :, 0] = 0
            position += frames
            if position >= len(data):
                raise sd.CallbackStop

        stream = sd.OutputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=callback,
            finished_callback=finished.set,
        )
        with stream:
            while not finished.wait(0.05):
                if self._stop.is_set():
                    stream.abort()
                    break
        return not self._stop.is_set()

    def stop(self) -> None:
        self._stop.set()
