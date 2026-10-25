"""Голоса Yandex SpeechKit (API v3) — онлайн, по ключу; работают и в России.

Ключ — API-ключ сервисного аккаунта Yandex Cloud с ролью ai.speechkit-tts.user (YANDEX_API_KEY
в .env). Голос и амплуа — tts.yandex_voice («kirill:strict» — Кирилл, строгое амплуа), скорость —
общая. Ответ — поток JSON-объектов {"result": {"audioChunk": {"data": <base64>}}} с сырым PCM:
48 кГц, 16 бит, моно. Знак «+» у SpeechKit — ударение, поэтому в тексте он читается словом.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import numpy as np

from assistant.tts.base import PhraseCache, TTSError, pcm16_to_float

log = logging.getLogger(__name__)

URL = "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis"
KEY_ENV = "YANDEX_API_KEY"
KEY_URL = "https://yandex.cloud/ru/docs/iam/operations/authentication/manage-api-keys"
SAMPLE_RATE = 48_000
TIMEOUT = 20.0  # секунд на фразу: без ответа говорит запасной голос
LIMIT = 250  # символов в одном запросе; длиннее — unsafeMode: сервис сам делит текст
# Голоса SpeechKit для русского (голос[:амплуа]). Kirill, alexander и anton есть только в API v3.
YANDEX_VOICES = (
    ("kirill:strict", "Кирилл — строгий, в духе Джарвиса"),
    ("kirill", "Кирилл"),
    ("alexander", "Александр"),
    ("anton", "Антон"),
    ("filipp", "Филипп"),
    ("ermil", "Ермил"),
    ("zahar", "Захар"),
    ("alena", "Алёна"),
    ("dasha", "Даша"),
    ("julia", "Юлия"),
    ("lera", "Лера"),
    ("marina", "Марина"),
    ("masha", "Маша"),
)

# Отправка запроса: (адрес, заголовки, тело, таймаут) → (код HTTP, тело ответа). Тестам — своя.
Poster = Callable[[str, dict[str, str], bytes, float], tuple[int, bytes]]


def http_post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST через urllib: прокси он берёт из настроек Windows сам.

    Raises:
        OSError: нет связи или сервис не ответил вовремя.
    """
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:  # ответ с кодом ошибки — тоже ответ: в нём причина
        return exc.code, exc.read()


class YandexTTS:
    """Голос Yandex SpeechKit.

    Args:
        voice: голос, можно с амплуа через двоеточие: «kirill:strict».
        rate: скорость речи (1.0 — обычная).
        post: отправка запроса (по умолчанию http_post) — подменяется в тестах.
        key: ключ вместо YANDEX_API_KEY (пример голоса с ещё не сохранённым ключом).
    """

    name = "yandex"

    def __init__(self, voice: str, rate: float = 1.0, post: Poster | None = None, key: str = "") -> None:
        self.voice, _, self.role = voice.partition(":")
        self.rate = rate
        self._post = post or http_post
        self._key = key
        self._cache = PhraseCache()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Звук фразы (float32, 48 кГц).

        Raises:
            TTSError: нет ключа, нет связи, ключ не подходит или сервис отказал.
        """
        cached = self._cache.get(text)
        if cached is not None:
            return cached, SAMPLE_RATE
        key = self._key or os.environ.get(KEY_ENV, "").strip()
        if not key:
            raise TTSError(
                "нет ключа Yandex SpeechKit: вставьте его в «Настройки → Голос ассистента» "
                "(или YANDEX_API_KEY в .env)"
            )
        headers = {"Authorization": f"Api-Key {key}", "Content-Type": "application/json"}
        try:
            status, data = self._post(URL, headers, json.dumps(self.request(text)).encode("utf-8"), TIMEOUT)
        except OSError as exc:
            raise TTSError(f"Yandex SpeechKit: нет связи ({exc})") from exc
        audio = pcm16_to_float(read_audio(status, data))
        if not len(audio):
            raise TTSError("Yandex SpeechKit не прислал звук")
        self._cache.put(text, audio)
        return audio, SAMPLE_RATE

    def request(self, text: str) -> dict[str, Any]:
        """Тело запроса utteranceSynthesis."""
        spoken = text.replace("+", " плюс ")
        hints: list[dict[str, Any]] = [{"voice": self.voice}]
        if self.role:
            hints.append({"role": self.role})
        hints.append({"speed": round(self.rate, 2)})
        raw = {"audioEncoding": "LINEAR16_PCM", "sampleRateHertz": SAMPLE_RATE}
        body: dict[str, Any] = {
            "text": spoken,
            "hints": hints,
            "outputAudioSpec": {"rawAudio": raw},
            "loudnessNormalizationType": "LUFS",
        }
        if len(spoken) > LIMIT:
            body["unsafeMode"] = True
        return body


def read_audio(status: int, data: bytes) -> bytes:
    """PCM из ответа: поток JSON-объектов с кусками звука в base64.

    Raises:
        TTSError: сервис ответил ошибкой или не JSON.
    """
    text = data.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    chunks: list[bytes] = []
    position = 0
    while True:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise TTSError(f"Yandex SpeechKit ответил не JSON (код {status}): {text[:200]}") from exc
        if not isinstance(item, dict):
            continue
        problem = item.get("error") if isinstance(item.get("error"), dict) else None
        if problem is None and "message" in item and "result" not in item:
            problem = item  # ошибка не в потоке: {"code": 16, "message": "…"}
        if problem is not None:
            raise TTSError(f"Yandex SpeechKit: {describe_error(status, problem)}")
        chunk = (item.get("result") or {}).get("audioChunk", {}).get("data")
        if chunk:
            chunks.append(base64.b64decode(chunk))
    if status >= 400:
        raise TTSError(f"Yandex SpeechKit: {describe_error(status, {})}")
    return b"".join(chunks)


def describe_error(status: int, problem: dict[str, Any]) -> str:
    """Понятное объяснение ошибки SpeechKit (ключ в текст не попадает)."""
    message = str(problem.get("message") or "").strip()
    code = int(problem.get("httpCode") or status or 0)
    if code == 401 or problem.get("code") == 16:
        return "ключ не подходит (401) — проверьте его в «Настройки → Голос ассистента»"
    if code == 403 or problem.get("code") == 7:
        return (
            "нет доступа (403): у сервисного аккаунта ключа должна быть роль ai.speechkit-tts.user, "
            "а у облака — привязанный платёжный аккаунт"
        )
    if code == 429:
        return "превышен лимит запросов (429)"
    if code >= 500:
        return f"сервис временно недоступен ({code})"
    return f"запрос отклонён ({code}): {message}" if message else f"запрос отклонён ({code})"
