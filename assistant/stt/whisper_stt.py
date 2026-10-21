"""Распознавание речи через faster-whisper.

Модель загружается один раз при старте. Видеокарта NVIDIA (CUDA) определяется
автоматически; если CUDA или её библиотеки недоступны, используется процессор.
Скачанная модель берётся из локального кэша без обращения к интернету.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from assistant.config import SttConfig

log = logging.getLogger(__name__)

# На тишине и шуме Whisper иногда «слышит» фразы из субтитров обучающих видео.
HALLUCINATION_MARKERS = (
    "субтитр",  # «Субтитры сделал DimaTorzok», «Редактор субтитров …»
    "dimatorzok",
    "продолжение следует",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "подпишись на канал",
)
_MAX_PROMPT_CHARS = 600


def is_hallucination(text: str) -> bool:
    """Похоже ли распознанное на известную «галлюцинацию» Whisper."""
    lowered = text.lower()
    return any(marker in lowered for marker in HALLUCINATION_MARKERS)


def build_prompt(name: str, command_phrases: Iterable[str], app_names: Iterable[str], max_apps: int) -> str:
    """Подсказка для Whisper: имя ассистента, частые слова команд и названия программ.

    Подсказка помогает правильно писать «Telegram», «Discord» и слова из команд.
    """
    counter: Counter[str] = Counter()
    for phrase in command_phrases:
        counter.update(w for w in phrase.lower().split() if "{" not in w and len(w) > 2)
    words = [w for w, _ in counter.most_common(25)]
    apps = list(dict.fromkeys(app_names))[:max_apps] if max_apps else []
    prompt = f"{name.capitalize()}, {', '.join(words)}."
    if apps:
        prompt += " " + ", ".join(apps) + "."
    return prompt[:_MAX_PROMPT_CHARS]


@dataclass(frozen=True)
class Transcript:
    """Результат распознавания."""

    text: str
    duration_ms: float
    no_speech_prob: float = 0.0


def cuda_available() -> bool:
    """Есть ли видеокарта, которую видит CTranslate2."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_settings(config: SttConfig, has_cuda: bool) -> tuple[str, str, str]:
    """Модель, устройство и тип вычислений с учётом «auto»."""
    device = config.device if config.device != "auto" else ("cuda" if has_cuda else "cpu")
    model = config.model if config.model != "auto" else ("large-v3-turbo" if device == "cuda" else "small")
    compute = config.compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return model, device, compute


class WhisperSTT:
    """Обёртка над faster_whisper.WhisperModel.

    Args:
        config: настройки распознавания.
        download_dir: куда скачивать модели.
        prompt: функция, возвращающая initial_prompt (список программ строится в фоне).
    """

    def __init__(
        self,
        config: SttConfig,
        download_dir: Path,
        prompt: Callable[[], str] | None = None,
    ) -> None:
        self._config = config
        self._download_dir = download_dir
        self._prompt = prompt or (lambda: "")
        self._model: Any = None
        self.model_name = ""
        self.device = ""

    def load(self) -> None:
        """Загружает модель: видеокарта, а при ошибке — процессор."""
        model, device, compute = resolve_settings(self._config, cuda_available())
        if device == "cuda":
            try:
                self._load(model, device, compute)
                self._warm_up()  # ошибки отсутствующих DLL CUDA видны только при распознавании
                return
            except Exception as exc:
                log.warning("Whisper на видеокарте не запустился (%s) — переключаюсь на процессор", exc)
                self._model = None
                cpu_config = self._config.model_copy(update={"device": "cpu"})
                model, device, _ = resolve_settings(cpu_config, has_cuda=False)
                compute = "int8"
        self._load(model, device, compute)

    def _load(self, model: str, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        started = time.perf_counter()
        self._download_dir.mkdir(parents=True, exist_ok=True)
        options = {"device": device, "compute_type": compute, "download_root": str(self._download_dir)}
        try:
            # Сначала только локальный кэш: так работает без интернета и не ждёт сеть.
            self._model = WhisperModel(model, local_files_only=True, **options)
        except Exception:
            log.info("Модели Whisper %s нет в кэше — скачиваю (один раз)", model)
            self._model = WhisperModel(model, local_files_only=False, **options)
        self.model_name, self.device = model, device
        log.info(
            "Whisper %s (%s, %s) загружен за %.1f с", model, device, compute, time.perf_counter() - started
        )

    def _warm_up(self) -> None:
        segments, _ = self._model.transcribe(np.zeros(16_000, dtype=np.float32), language="ru", beam_size=1)
        list(segments)

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Распознаёт фразу (int16 или float32, 16 кГц).

        Возвращает пустой текст, если Whisper не услышал речь или выдал «галлюцинацию».
        """
        if self._model is None:
            raise RuntimeError("Модель Whisper не загружена")
        samples = audio.astype(np.float32)
        if audio.dtype == np.int16:
            samples /= 32768.0
        started = time.perf_counter()
        segments, _ = self._model.transcribe(
            samples,
            language="ru",
            beam_size=self._config.beam_size,
            initial_prompt=self._prompt() or None,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # фраза уже вырезана нашим VAD
        )
        parts, no_speech = [], 0.0
        for segment in segments:
            no_speech = max(no_speech, segment.no_speech_prob)
            if segment.no_speech_prob < 0.8:
                parts.append(segment.text.strip())
        text = " ".join(p for p in parts if p).strip()
        duration_ms = (time.perf_counter() - started) * 1000
        if text and is_hallucination(text):
            log.info("Отброшена «галлюцинация» Whisper: %s", text)
            text = ""
        return Transcript(text, duration_ms, no_speech)
