"""Запасная озвучка через голоса Windows (SAPI5) с помощью pyttsx3.

Речь сохраняется во временный WAV и проигрывается тем же проигрывателем, что и Piper,
поэтому её так же можно прервать. COM-объекты SAPI создаются в потоке, который
синтезирует (поток очереди озвучки), и используются только в нём.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from assistant.tts.base import TTSError

log = logging.getLogger(__name__)

_DEFAULT_RATE_WPM = 200


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Читает 16-битный WAV в float32 моно."""
    with wave.open(str(path), "rb") as wav:
        channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise TTSError(f"Неожиданный формат WAV от SAPI: {width * 8} бит")
    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, rate


class SapiTTS:
    """Голос Windows.

    Args:
        voice_hint: часть названия голоса («Irina»); пусто — первый русский голос.
        rate: скорость речи (1.0 — обычная).
    """

    name = "sapi"

    def __init__(self, voice_hint: str = "", rate: float = 1.0) -> None:
        self._voice_hint = voice_hint.lower()
        self._rate = rate
        self._engine: Any = None
        self._owner_thread: int | None = None

    def _ensure_engine(self) -> Any:
        """Создаёт движок pyttsx3 в текущем потоке (COM требует одного потока)."""
        if self._engine is not None and self._owner_thread == threading.get_ident():
            return self._engine
        try:
            import comtypes

            comtypes.CoInitialize()
        except (ImportError, OSError):
            pass
        try:
            import pyttsx3

            engine = pyttsx3.init("sapi5")
        except Exception as exc:
            raise TTSError(f"Голоса Windows (SAPI) недоступны: {exc}") from exc
        voice = self._pick_voice(engine.getProperty("voices"))
        if voice is not None:
            engine.setProperty("voice", voice.id)
            log.info("Голос SAPI: %s", voice.name)
        else:
            log.warning("Русский голос SAPI не найден — будет голос по умолчанию")
        engine.setProperty("rate", int(_DEFAULT_RATE_WPM * self._rate))
        self._engine, self._owner_thread = engine, threading.get_ident()
        return engine

    def available_voices(self) -> list[str]:
        """Названия и языки установленных голосов Windows."""
        engine = self._ensure_engine()
        return [f"{v.name} ({', '.join(map(str, v.languages or []))})" for v in engine.getProperty("voices")]

    @property
    def has_russian_voice(self) -> bool:
        """Найден ли русский голос (без него русский текст SAPI не произнесёт)."""
        engine = self._ensure_engine()
        return self._pick_voice(engine.getProperty("voices")) is not None

    def _pick_voice(self, voices: list[Any]) -> Any:
        if self._voice_hint:
            for voice in voices:
                if self._voice_hint in str(voice.name).lower():
                    return voice
        for voice in voices:
            languages = [str(lang).lower() for lang in (voice.languages or [])]
            if any(lang.startswith("ru") for lang in languages) or "russian" in str(voice.name).lower():
                return voice
        return None

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        engine = self._ensure_engine()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.wav"
            engine.save_to_file(text, str(path))
            engine.runAndWait()
            if not path.is_file():
                raise TTSError("SAPI не создал файл с речью")
            return read_wav(path)
