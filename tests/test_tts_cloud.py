"""Голоса по ключу: OpenAI (gpt-4o-mini-tts) и Yandex SpeechKit — без сети, на подделках."""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np
import pytest

from assistant.config import JARVIS_STYLE, AppConfig
from assistant.tts.base import PhraseCache, TTSError, pcm16_to_float
from assistant.tts.openai_tts import OpenAITTS, openai_engine
from assistant.tts.yandex_tts import YandexTTS, read_audio

PCM = np.array([0, 16384, -16384, 32767], dtype="<i2").tobytes()  # 0, 0.5, -0.5, ~1.0


def test_pcm_and_phrase_cache() -> None:
    assert pcm16_to_float(PCM + b"\x01").tolist() == pytest.approx([0.0, 0.5, -0.5, 32767 / 32768])
    cache = PhraseCache(size=2)
    one, two = np.zeros(1, np.float32), np.ones(1, np.float32)
    cache.put("раз", one)
    cache.put("два", two)
    assert cache.get("раз") is one  # «раз» теперь самая свежая
    cache.put("три", two)
    assert cache.get("два") is None and cache.get("раз") is one
    long = "Длинный ответ нейросети почти не повторяется, а память занимает."
    cache.put(long, one)
    assert cache.get(long) is None and cache.get("раз") is one  # и короткие не вытеснил


# --- OpenAI ------------------------------------------------------------------------------


class FakeOpenAI:
    """Как openai.OpenAI: audio.speech.create(...) → ответ с .read(); запоминает запросы."""

    made: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        FakeOpenAI.made.append(kwargs)
        self.audio = self
        self.speech = self

    def create(self, **request: Any) -> Any:
        FakeOpenAI.requests.append(request)
        if FakeOpenAI.error is not None:
            raise FakeOpenAI.error
        return type("Response", (), {"read": lambda self: PCM})()


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> type[FakeOpenAI]:
    FakeOpenAI.made, FakeOpenAI.requests, FakeOpenAI.error = [], [], None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-1234")
    return FakeOpenAI


def test_openai_voice_speaks_in_jarvis_manner(fake_openai: type[FakeOpenAI]) -> None:
    import openai

    tts = AppConfig().tts.model_copy(update={"rate": 1.1})
    engine = openai_engine(tts)
    engine._factory = fake_openai
    audio, rate = engine.synthesize("Добрый вечер.")
    assert rate == 24_000 and audio.dtype == np.float32 and audio.tolist()[1] == 0.5
    assert fake_openai.requests == [{
        "model": "gpt-4o-mini-tts", "voice": "cedar", "input": "Добрый вечер.", "response_format": "pcm",
        "speed": 1.1, "instructions": JARVIS_STYLE,
    }]  # fmt: skip
    assert fake_openai.made == [
        {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test-key-1234",
            "max_retries": 1,
            # Без сети — 5 с на соединение; с повтором фраза ждёт не дольше ~20 с, потом запасной голос.
            "timeout": openai.Timeout(10.0, connect=5.0),
        }
    ]
    engine.synthesize("Добрый вечер.")  # повтор — из памяти, без запроса
    assert len(fake_openai.requests) == 1


def test_openai_instructions_only_for_models_that_take_them(fake_openai: type[FakeOpenAI]) -> None:
    OpenAITTS("onyx", "Говори медленно", model="tts-1-hd", factory=fake_openai).synthesize("Раз.")
    OpenAITTS("onyx", "", factory=fake_openai).synthesize("Два.")  # без манеры речи
    assert ["instructions" in request for request in fake_openai.requests] == [False, False]


def test_openai_new_key_works_without_restart(
    fake_openai: type[FakeOpenAI], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = OpenAITTS("cedar", factory=fake_openai)
    engine.synthesize("Раз.")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-new-key-5678")  # ключ сохранили в настройках
    engine.synthesize("Два.")
    assert [made["api_key"] for made in fake_openai.made] == ["sk-test-key-1234", "sk-new-key-5678"]
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(TTSError, match="нет ключа OpenAI"):
        engine.synthesize("Три.")


def test_openai_errors_are_explained_without_the_key(fake_openai: type[FakeOpenAI]) -> None:
    import httpx2
    import openai

    request = httpx2.Request("POST", "https://api.openai.com/v1/audio/speech")
    rejected = openai.AuthenticationError(
        "Incorrect API key provided: sk-test-...1234",
        response=httpx2.Response(401, request=request),
        body=None,
    )
    fake_openai.error = rejected
    with pytest.raises(TTSError, match=r"OpenAI: ключ не подходит \(401\)") as error:
        OpenAITTS("cedar", factory=fake_openai).synthesize("Раз.")
    assert "sk-test" not in str(error.value)
    fake_openai.error = openai.APIConnectionError(request=request)
    with pytest.raises(TTSError, match="нет связи с https://proxy.example/v1"):
        OpenAITTS("cedar", base_url="https://proxy.example/v1", factory=fake_openai).synthesize("Два.")


# --- Yandex SpeechKit --------------------------------------------------------------------


def chunks(*parts: bytes) -> bytes:
    """Ответ SpeechKit: поток JSON-объектов с кусками звука в base64."""
    items = [
        {"result": {"audioChunk": {"data": base64.b64encode(part).decode()}, "startMs": "0"}}
        for part in parts
    ]
    return "\n".join(json.dumps(item) for item in items).encode()


class FakePost:
    def __init__(self, status: int = 200, body: bytes = b"") -> None:
        self.status, self.body = status, body
        self.calls: list[tuple[str, dict[str, str], dict[str, Any], float]] = []

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
        self.calls.append((url, headers, json.loads(body), timeout))
        if isinstance(self.body, Exception):
            raise self.body
        return self.status, self.body


def test_yandex_voice_with_role_speed_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YANDEX_API_KEY", "AQVN-test-key")
    post = FakePost(body=chunks(PCM[:4], PCM[4:]))
    engine = YandexTTS("kirill:strict", rate=0.95, post=post)
    audio, rate = engine.synthesize("Температура 42°, а C++ — язык.")
    assert rate == 48_000 and audio.tolist() == pytest.approx(pcm16_to_float(PCM).tolist())
    url, headers, body, timeout = post.calls[0]
    assert url == "https://tts.api.cloud.yandex.net/tts/v3/utteranceSynthesis" and timeout == 10.0
    assert headers["Authorization"] == "Api-Key AQVN-test-key"
    assert body == {
        "text": "Температура 42°, а C плюс  плюс  — язык.",  # «+» у SpeechKit — ударение
        "hints": [{"voice": "kirill"}, {"role": "strict"}, {"speed": 0.95}],
        "outputAudioSpec": {"rawAudio": {"audioEncoding": "LINEAR16_PCM", "sampleRateHertz": 48_000}},
        "loudnessNormalizationType": "LUFS",
    }
    engine.synthesize("Температура 42°, а C++ — язык.")  # повтор — из памяти
    assert len(post.calls) == 1
    long = YandexTTS("alena", post=FakePost(body=chunks(PCM)))
    long.synthesize("слово " * 60)  # длиннее 250 символов: сервис сам поделит текст
    assert long._post.calls[0][2]["unsafeMode"] is True  # type: ignore[attr-defined]
    assert long._post.calls[0][2]["hints"][:1] == [{"voice": "alena"}]  # type: ignore[attr-defined]


def test_yandex_errors_are_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YANDEX_API_KEY", raising=False)
    with pytest.raises(TTSError, match="нет ключа Yandex SpeechKit"):
        YandexTTS("kirill", post=FakePost()).synthesize("Раз.")
    monkeypatch.setenv("YANDEX_API_KEY", "AQVN-test-key")
    unauthenticated = json.dumps({"code": 16, "message": "Unknown api key"}).encode()
    with pytest.raises(TTSError, match=r"ключ не подходит \(401\)"):
        YandexTTS("kirill", post=FakePost(401, unauthenticated)).synthesize("Раз.")
    with pytest.raises(TTSError, match=r"роль ai\.speechkit-tts\.user"):
        YandexTTS("kirill", post=FakePost(403, b'{"code": 7, "message": "Permission denied"}')).synthesize(
            "Раз."
        )
    streamed = chunks(PCM) + b'\n{"error": {"grpcCode": 3, "httpCode": 400, "message": "Unsupported voice"}}'
    with pytest.raises(TTSError, match=r"запрос отклонён \(400\): Unsupported voice"):
        YandexTTS("madirus", post=FakePost(200, streamed)).synthesize("Раз.")
    with pytest.raises(TTSError, match="нет связи"):
        YandexTTS("kirill", post=FakePost(body=OSError("timed out"))).synthesize("Раз.")  # type: ignore[arg-type]
    with pytest.raises(TTSError, match="не прислал звук"):
        YandexTTS("kirill", post=FakePost(200, b"")).synthesize("Раз.")
    assert read_audio(200, chunks(b"ab", b"cd")) == b"abcd"
    with pytest.raises(TTSError, match="не JSON"):
        read_audio(502, b"<html>Bad gateway</html>")
