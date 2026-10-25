"""Онлайн-голоса Microsoft через edge-tts — те же нейросетевые голоса, что в «Чтении вслух» Edge.

Нужен интернет: без него очередь озвучки говорит запасным голосом. Сервис присылает MP3,
его декодирует PyAV — он уже есть в программе для faster-whisper (сам decode_audio из
faster-whisper 1.2.1 с PyAV 19 не работает: передаёт удалённый параметр metadata_errors).
"""

from __future__ import annotations

import asyncio
import io
import logging
import urllib.request

import numpy as np

from assistant.tts.base import TTSError

log = logging.getLogger(__name__)

# Голоса сервиса для русского: свои (локаль ru-RU) и многоязычные — новее и живее, они сами узнают
# язык текста (может слышаться лёгкий акцент). Калибровка в CI (tests/test_voice_e2e.py): русскую
# фразу, озвученную многоязычными голосами, Whisper распознаёт на 96–100 из 100.
EDGE_VOICES = (
    ("ru-RU-DmitryNeural", "Дмитрий"),
    ("ru-RU-SvetlanaNeural", "Светлана"),
    ("en-US-BrianMultilingualNeural", "Брайан — многоязычный, мужской"),
    ("en-US-AndrewMultilingualNeural", "Эндрю — многоязычный, мужской"),
    ("de-DE-FlorianMultilingualNeural", "Флориан — многоязычный, мужской"),
    ("it-IT-GiuseppeMultilingualNeural", "Джузеппе — многоязычный, мужской"),
    ("fr-FR-RemyMultilingualNeural", "Реми — многоязычный, мужской"),
    ("en-US-EmmaMultilingualNeural", "Эмма — многоязычный, женский"),
    ("de-DE-SeraphinaMultilingualNeural", "Серафина — многоязычный, женский"),
)
SAMPLE_RATE = 24_000  # формат ответа сервиса: audio-24khz-48kbitrate-mono-mp3
CONNECT_TIMEOUT = 5  # секунд: без сети быстрее перейти на запасной голос
RECEIVE_TIMEOUT = 15


def edge_rate(rate: float) -> str:
    """Скорость речи для edge-tts: 1.2 → «+20%», 0.8 → «-20%»."""
    return f"{round((rate - 1.0) * 100):+d}%"


def decode_mp3(data: bytes, rate: int = SAMPLE_RATE) -> np.ndarray:
    """MP3 (или любой звук, который понимает FFmpeg) в float32 моно с частотой rate."""
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
    chunks = []
    with av.open(io.BytesIO(data), mode="r") as container:
        for frame in container.decode(audio=0):
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]  # остаток
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def system_proxy() -> str | None:
    """Прокси из настроек Windows: aiohttp (на нём edge-tts) сам видит только переменные окружения."""
    proxies = urllib.request.getproxies()
    return proxies.get("https") or proxies.get("http") or None


class EdgeTTS:
    """Голос Microsoft (edge-tts).

    Args:
        voice: голос, например ru-RU-DmitryNeural.
        rate: скорость речи (1.0 — обычная).
    """

    name = "edge"

    def __init__(self, voice: str, rate: float = 1.0) -> None:
        self.voice = voice
        self._rate = edge_rate(rate)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Звук фразы (float32, 24 кГц).

        Raises:
            TTSError: нет сети, сервис не ответил или прислал не звук.
        """
        try:
            mp3 = asyncio.run(self._fetch(text))
        except Exception as exc:
            raise TTSError(f"edge-tts: {type(exc).__name__}: {exc}") from exc
        if not mp3:
            raise TTSError("edge-tts не прислал звук")
        try:
            return decode_mp3(mp3), SAMPLE_RATE
        except Exception as exc:
            raise TTSError(f"edge-tts прислал не звук: {exc}") from exc

    async def _fetch(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self._rate,
            proxy=system_proxy(),
            connect_timeout=CONNECT_TIMEOUT,
            receive_timeout=RECEIVE_TIMEOUT,
        )
        chunks = [chunk["data"] async for chunk in communicate.stream() if chunk["type"] == "audio"]
        return b"".join(chunks)
