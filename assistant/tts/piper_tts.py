"""Озвучка через Piper — офлайн, нейросетевой русский голос."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from assistant.tts.base import TTSError

log = logging.getLogger(__name__)


class PiperTTS:
    """Голос Piper (например, ru_RU-dmitri-medium).

    Args:
        model_path: файл .onnx голоса; рядом должен лежать .onnx.json.
        rate: скорость речи (1.0 — обычная, больше — быстрее).
    """

    name = "piper"

    def __init__(self, model_path: Path, rate: float = 1.0) -> None:
        if not model_path.is_file() or not Path(f"{model_path}.json").is_file():
            raise TTSError(
                f"Голос Piper не найден: {model_path} (и .onnx.json рядом). "
                "Скачайте его: uv run python -m assistant --download-models"
            )
        from piper import PiperVoice, SynthesisConfig

        started = time.perf_counter()
        self._voice = PiperVoice.load(model_path)
        self._config = SynthesisConfig(length_scale=1.0 / rate)
        log.info("Голос Piper %s загружен за %.1f с", model_path.stem, time.perf_counter() - started)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        chunks = list(self._voice.synthesize(text, syn_config=self._config))
        if not chunks:
            return np.zeros(0, dtype=np.float32), 22_050
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate
