"""Голоса OpenAI (gpt-4o-mini-tts) — онлайн, по ключу. Манеру речи задаёт текст (instructions).

Ключ — тот же, что у профиля ChatGPT (OPENAI_API_KEY в .env), адрес — свой (tts.openai_base_url):
можно указать OpenAI-совместимый прокси-сервис. Ответ — сырой PCM: 24 кГц, 16 бит, моно.
Числа и латиницу сервис читает сам. Каждое предложение — отдельный запрос; недавние фразы
(«Готово», «Громкость 60%») берутся из памяти, а не синтезируются заново.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from assistant.config import JARVIS_STYLE, TtsConfig
from assistant.tts.base import PhraseCache, TTSError, pcm16_to_float

log = logging.getLogger(__name__)

KEY_ENV = "OPENAI_API_KEY"  # общий с профилем нейросети ChatGPT
SAMPLE_RATE = 24_000  # response_format="pcm"
TIMEOUT = 20.0  # секунд на фразу: без ответа говорит запасной голос
# Голоса OpenAI. Лучшие по качеству, по словам OpenAI, — cedar и marin.
OPENAI_VOICES = (
    ("cedar", "Cedar — мужской, лучший по качеству"),
    ("onyx", "Onyx — низкий мужской"),
    ("ash", "Ash — мужской"),
    ("ballad", "Ballad — мягкий мужской"),
    ("echo", "Echo — мужской"),
    ("verse", "Verse — мужской"),
    ("fable", "Fable — с британскими нотками"),
    ("alloy", "Alloy — нейтральный"),
    ("marin", "Marin — женский, лучший по качеству"),
    ("coral", "Coral — женский"),
    ("nova", "Nova — женский"),
    ("sage", "Sage — женский"),
    ("shimmer", "Shimmer — женский"),
)


def openai_engine(tts: TtsConfig) -> OpenAITTS:
    """Голос OpenAI по настройкам озвучки."""
    return OpenAITTS(tts.openai_voice, tts.openai_style, tts.openai_model, tts.openai_base_url, tts.rate)


def takes_instructions(model: str) -> bool:
    """Манеру речи понимают модели gpt-4o-mini-tts; tts-1 и tts-1-hd — нет."""
    return not model.startswith("tts-1")


class OpenAITTS:
    """Голос OpenAI.

    Args:
        voice: голос (cedar, onyx…).
        style: как говорить (instructions); пусто — без указаний.
        model: модель озвучки.
        base_url: адрес OpenAI-совместимого API.
        rate: скорость речи (1.0 — обычная).
        factory: конструктор клиента (по умолчанию openai.OpenAI) — подменяется в тестах.
        key: ключ вместо OPENAI_API_KEY (пример голоса с ещё не сохранённым ключом).
    """

    name = "openai"

    def __init__(
        self,
        voice: str,
        style: str = JARVIS_STYLE,
        model: str = "gpt-4o-mini-tts",
        base_url: str = "https://api.openai.com/v1",
        rate: float = 1.0,
        factory: Callable[..., Any] | None = None,
        key: str = "",
    ) -> None:
        self.voice = voice
        self.style = style.strip()
        self.model = model
        self.base_url = base_url
        self.rate = rate
        self._factory = factory
        self._key = key
        self._client: Any = None
        self._client_key = ""
        self._cache = PhraseCache()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Звук фразы (float32, 24 кГц).

        Raises:
            TTSError: нет ключа, нет связи, ключ не подходит или сервис отказал.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached, SAMPLE_RATE
        request: dict[str, Any] = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
            "speed": self.rate,
        }
        if self.style and takes_instructions(self.model):
            request["instructions"] = self.style
        try:
            pcm = self._connect().audio.speech.create(**request).read()
        except TTSError:
            raise
        except Exception as exc:
            raise TTSError(f"OpenAI: {describe_error(exc, self.base_url)}") from exc
        audio = pcm16_to_float(pcm)
        if not len(audio):
            raise TTSError("OpenAI не прислал звук")
        self._cache.put(text, audio)
        return audio, SAMPLE_RATE

    def _connect(self) -> Any:
        """Клиент OpenAI; ключ читается при каждой фразе — новый ключ из настроек работает сразу."""
        key = self._key or os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise TTSError(
                "нет ключа OpenAI: вставьте его в «Настройки → Голос ассистента» (или OPENAI_API_KEY в .env)"
            )
        if self._client is None or key != self._client_key:
            factory = self._factory
            if factory is None:
                from openai import OpenAI

                factory = OpenAI
            self._client = factory(base_url=self.base_url, api_key=key, max_retries=1, timeout=TIMEOUT)
            self._client_key = key
        return self._client


def describe_error(exc: Exception, base_url: str) -> str:
    """Понятное объяснение ошибки сервиса озвучки (ключ в текст не попадает)."""
    import openai

    detail = str(getattr(exc, "message", "") or exc)
    if isinstance(exc, openai.AuthenticationError):
        return "ключ не подходит (401) — проверьте его в «Настройки → Голос ассистента»"
    if isinstance(exc, openai.PermissionDeniedError):
        return f"доступ запрещён (403): {detail}"
    if isinstance(exc, openai.RateLimitError):
        return "превышен лимит запросов или кончились деньги на счёте (429)"
    if isinstance(exc, openai.NotFoundError):
        return f"нет такой модели или голоса у {base_url} (404): {detail}"
    if isinstance(exc, openai.APITimeoutError):
        return "сервис не ответил вовремя"
    if isinstance(exc, openai.APIConnectionError):
        return f"нет связи с {base_url}"
    if isinstance(exc, openai.APIStatusError):
        return f"запрос отклонён ({exc.status_code}): {detail}"
    return detail or type(exc).__name__
