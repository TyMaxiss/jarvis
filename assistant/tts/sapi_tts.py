"""Запасная озвучка через голоса Windows (SAPI5).

Речь синтезируется синхронным вызовом SpVoice.Speak в WAV-файл и проигрывается тем же
проигрывателем, что и Piper, поэтому её так же можно прервать. COM-объекты создаются
в потоке, который синтезирует (поток очереди озвучки), и используются только в нём.

В ТЗ предлагался pyttsx3, но pyttsx3 2.99 (save_to_file + runAndWait) в CI на Windows
сохранял WAV без единого сэмпла — даже английским голосом. Поэтому SAPI вызывается
напрямую через comtypes, на котором построен и сам pyttsx3: это те же объекты
SpVoice и SpFileStream, только без цикла событий pyttsx3.
"""

from __future__ import annotations

import contextlib
import logging
import math
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import numpy as np

from assistant.tts.base import TTSError

log = logging.getLogger(__name__)

_SSFM_CREATE_FOR_WRITE = 3
_SAFT_22KHZ_16BIT_MONO = 22
_SVSF_IS_NOT_XML = 16  # синхронно (Speak возвращается, когда речь записана), текст — не XML
_RUSSIAN_LCID = "419"


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


def token_languages(token: Any) -> list[str]:
    """Коды языков голоса SAPI: шестнадцатеричные LCID через «;» («419» — русский)."""
    try:
        value = str(token.GetAttribute("Language") or "")
    except Exception:  # у голоса нет атрибута Language
        return []
    return [code.strip().lower().lstrip("0") for code in value.split(";") if code.strip()]


def sapi_rate(rate: float) -> int:
    """Скорость 0.5–2.0 → шкала SAPI −10…10 (логарифмическая, ±10 ≈ в 3 раза)."""
    return max(-10, min(10, round(10 * math.log(rate) / math.log(3))))


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
        self._voice: Any = None
        self._owner_thread: int | None = None
        self._russian = False

    def _ensure_voice(self) -> Any:
        """Создаёт SpVoice в текущем потоке (COM требует одного потока)."""
        if self._voice is not None and self._owner_thread == threading.get_ident():
            return self._voice
        try:
            import comtypes
            import comtypes.client
        except ImportError as exc:
            raise TTSError("Голоса Windows (SAPI) доступны только в Windows") from exc
        with contextlib.suppress(OSError):  # COM в этом потоке уже инициализирован
            comtypes.CoInitialize()
        try:
            voice = comtypes.client.CreateObject("SAPI.SpVoice")
        except Exception as exc:
            raise TTSError(f"Голоса Windows (SAPI) недоступны: {exc}") from exc
        token = self._pick_token(voice.GetVoices())
        if token is not None:
            voice.Voice = token
            log.info("Голос SAPI: %s", token.GetDescription())
        else:
            log.warning("Русский голос SAPI не найден — будет голос по умолчанию")
        self._russian = token is not None
        voice.Rate = sapi_rate(self._rate)
        self._voice, self._owner_thread = voice, threading.get_ident()
        return voice

    def _pick_token(self, tokens: Any) -> Any:
        """Голос по подсказке из конфига, иначе первый русский."""
        items = [tokens.Item(i) for i in range(tokens.Count)]
        if self._voice_hint:
            for token in items:
                if self._voice_hint in str(token.GetDescription()).lower():
                    return token
        return next((token for token in items if _RUSSIAN_LCID in token_languages(token)), None)

    def available_voices(self) -> list[str]:
        """Названия и коды языков установленных голосов Windows."""
        tokens = self._ensure_voice().GetVoices()
        items = [tokens.Item(i) for i in range(tokens.Count)]
        return [f"{t.GetDescription()} (язык {', '.join(token_languages(t)) or '?'})" for t in items]

    @property
    def has_russian_voice(self) -> bool:
        """Найден ли русский голос (без него русский текст SAPI не произнесёт)."""
        self._ensure_voice()
        return self._russian

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        import comtypes.client

        voice = self._ensure_voice()
        # Если SAPI ещё держит файл, временная папка удалится позже — это не ошибка озвучки.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            path = Path(tmp) / "speech.wav"
            stream = comtypes.client.CreateObject("SAPI.SpFileStream")
            try:
                stream.Format.Type = _SAFT_22KHZ_16BIT_MONO
            except Exception:
                log.debug("Формат файла SAPI не задан, будет формат по умолчанию")
            stream.Open(str(path), _SSFM_CREATE_FOR_WRITE, False)
            try:
                voice.AudioOutputStream = stream
                voice.Speak(text, _SVSF_IS_NOT_XML)
            finally:
                stream.Close()
            if not path.is_file():
                raise TTSError("SAPI не создал файл с речью")
            audio, rate = read_wav(path)
            if not len(audio):
                raise TTSError(f"SAPI сохранил речь без звука (файл {path.stat().st_size} байт)")
            return audio, rate
