"""Сквозная проверка голоса на настоящих моделях: Piper → VAD → Vosk → Whisper → матчер.

Речь синтезируется голосом Piper и подаётся в голосовой цикл вместо микрофона.
Нужны скачанные модели (uv run python -m assistant --download-models), поэтому тест
включается переменной JARVIS_E2E=1 (так сделано в CI на Windows).

Логика цикла и безопасность подтверждения проверяются на всех скачанных голосах.
Точность распознавания — на эталонном голосе irina: синтетический dmitri произносит
отдельные слова так, что Whisper small и Vosk их путают («Открой ютуб» → «Пробою тут»),
это свойство синтезатора, а не конвейера. Его результаты видны в калибровке.

Калибровка печатает длительность речи по VAD, уверенность Vosk и текст Whisper
(beam 5 и 1) для фраз с именем и без — по ним подобраны пороги в config.example.yaml.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assistant.audio.capture import ArraySource, resample, to_int16
from assistant.audio.vad import SegmentEvent, SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import GrammarSpotter, SpotResult, load_vosk_model
from assistant.config import AppConfig
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer
from assistant.safety.confirm import NO_PHRASES, YES_PHRASES, VoiceConfirmer
from assistant.stt.whisper_stt import WhisperSTT
from assistant.tts.piper_tts import PiperTTS
from assistant.voice import VoiceLoop, VoiceSettings
from tests.conftest import NO_WINDOW, ROOT
from tests.voice_fakes import FakeSpeech, RecordingStatus

pytestmark = pytest.mark.skipif(os.environ.get("JARVIS_E2E") != "1", reason="включается через JARVIS_E2E=1")
log = logging.getLogger(__name__)
CONFIG = AppConfig()
NORMALIZER = Normalizer(CONFIG.wake_words, CONFIG.matcher.filler_words)
PIPER_DIR = ROOT / "models" / "piper"
REFERENCE_VOICE = "ru_RU-irina-medium"


def voice_paths() -> list[Path]:
    return sorted(PIPER_DIR.glob("ru_RU-*.onnx")) if PIPER_DIR.is_dir() else []


def reference_voice() -> Path | None:
    """Голос для проверки точности: irina, а если её нет — первый скачанный."""
    voices = voice_paths()
    return next((p for p in voices if p.stem == REFERENCE_VOICE), voices[0] if voices else None)


@pytest.fixture(scope="module", params=voice_paths() or [None], ids=lambda p: p.stem if p else "нет")
def piper(request: pytest.FixtureRequest) -> PiperTTS:
    if request.param is None:
        pytest.skip("голоса Piper не скачаны")
    return PiperTTS(request.param)


@pytest.fixture(scope="module")
def reference_piper() -> PiperTTS:
    path = reference_voice()
    if path is None:
        pytest.skip("голоса Piper не скачаны")
    log.warning("Эталонный голос: %s", path.stem)
    return PiperTTS(path)


@pytest.fixture(scope="module")
def vosk_model() -> object:
    return load_vosk_model(ROOT / CONFIG.voice.vosk_model)


@pytest.fixture(scope="module")
def stt() -> WhisperSTT:
    prompt = "Джарвис, открой, сделай, громче, громкость, найди, ютуб."
    stt = WhisperSTT(CONFIG.stt, ROOT / CONFIG.stt.download_dir, lambda: prompt)
    stt.load()
    log.warning("Whisper: %s на %s, beam %d", stt.model_name, stt.device, CONFIG.stt.beam_size)
    return stt


def say(piper: PiperTTS, text: str) -> np.ndarray:
    """Синтезированная фраза в формате микрофона: 16 кГц, int16."""
    audio, rate = piper.synthesize(text)
    return to_int16(resample(audio, rate))


def gap(seconds: float) -> np.ndarray:
    return np.zeros(int(16_000 * seconds), dtype=np.int16)


def spot(vosk_model: object, phrases: list[str], audio: np.ndarray) -> SpotResult:
    spotter = GrammarSpotter(vosk_model, phrases)
    result = SpotResult()
    for start in range(0, len(audio) - 511, 512):
        part = spotter.accept(audio[start : start + 512])
        if part is not None:
            result = result + part
    return result + spotter.finish()


def speech_span_ms(audio: np.ndarray) -> int:
    """Длительность речи так, как её видит голосовой цикл (VAD и границы фразы)."""
    vad, segmenter = SileroVAD(), SpeechSegmenter()
    padded = np.concatenate([audio, gap(1.0)])
    for start in range(0, len(padded) - 511, 512):
        block = padded[start : start + 512]
        if segmenter.feed(block, vad(block)) is SegmentEvent.END:
            return segmenter.speech_span_ms
    return -1


def run_loop(audio: np.ndarray, vosk_model: object, stt: WhisperSTT) -> list[str]:
    handled: list[str] = []
    loop = VoiceLoop(
        source=ArraySource(audio, silence_after=2.0),
        vad=SileroVAD(),
        segmenter=SpeechSegmenter(),
        spotter=GrammarSpotter(vosk_model, [CONFIG.general.name]),
        stt=stt,
        handle_text=lambda text, conf: handled.append(text),
        speech=FakeSpeech(),
        normalizer=NORMALIZER,
        wake_names=CONFIG.wake_words,
        settings=VoiceSettings(
            followup_seconds=3, listen_timeout=4, wake_confidence=CONFIG.voice.wake_confidence
        ),
        status=RecordingStatus(),
    )
    loop.run(stop_when_exhausted=True)
    log.warning("Цикл передал в конвейер: %s", handled)
    return handled


def confirm_by_voice(piper: PiperTTS, vosk_model: object, answer: str) -> bool:
    source = ArraySource(np.concatenate([gap(0.5), say(piper, answer), gap(1.0)]), silence_after=0.5)
    confirmer = VoiceConfirmer(
        FakeSpeech(),
        source,
        lambda: GrammarSpotter(vosk_model, YES_PHRASES + NO_PHRASES),
        timeout=5,
        min_confidence=CONFIG.voice.confirm_confidence,
    )
    return confirmer.confirm("Выключить компьютер?")


def test_wake_word_is_in_vosk_vocabulary(vosk_model: object) -> None:
    spotter = GrammarSpotter(vosk_model, ["джарвис"])
    log.warning("Слово «джарвис» в словаре Vosk: %s", spotter.usable)
    assert spotter.usable
    confirm = GrammarSpotter(vosk_model, YES_PHRASES + NO_PHRASES)
    assert confirm.missing_words == []


CALIBRATION_PHRASES = [
    "Джарвис.",
    "Джарвис, открой ютуб.",
    "Джарвис, сделай громче.",
    "Сделай громче.",
    "Открой ютуб.",
    "Громкость на пятьдесят.",
    "Который час?",
    "Сегодня хорошая погода.",
    "Я пошёл в магазин за хлебом.",
    "Давай посмотрим кино вечером.",
]
CALIBRATION_ANSWERS = [
    "Да.",
    "Нет.",
    "Подтверждаю.",
    "Конечно.",
    "Не надо.",
    "Отмена.",
    "Сделай громче.",
    "Сегодня хорошая погода.",
]


def test_calibration_vosk_and_whisper(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    """Печатает длительность речи, уверенность Vosk и текст Whisper (beam 5 и 1)."""
    for number, phrase in enumerate(CALIBRATION_PHRASES):
        speech = say(piper, phrase)
        audio = np.concatenate([gap(0.3), speech, gap(0.5)])
        heard = spot(vosk_model, [CONFIG.general.name], audio)
        timings = []
        for beam in (5, 1) if number < 7 else (5,):
            started = time.perf_counter()
            text = stt.transcribe(audio, beam_size=beam).text
            timings.append(f"beam{beam}: «{text}» {(time.perf_counter() - started) * 1000:.0f} мс")
        log.warning(
            "КАЛИБРОВКА «%s» | речь %d мс | Vosk: %s | %s",
            phrase, speech_span_ms(speech), heard.describe(), " | ".join(timings),
        )  # fmt: skip
    for answer in CALIBRATION_ANSWERS:
        audio = np.concatenate([gap(0.3), say(piper, answer), gap(0.5)])
        heard = spot(vosk_model, list(YES_PHRASES + NO_PHRASES), audio)
        log.warning("КАЛИБРОВКА ответ «%s» | Vosk: %s", answer, heard.describe())


@pytest.mark.parametrize(
    ("phrase", "command_id"),
    [
        ("Сделай громче.", "volume_up"),
        ("Громкость на пятьдесят.", "volume_set"),
        ("Открой ютуб.", "open_youtube"),
        ("Который час?", "time"),
    ],
)
def test_whisper_recognizes_commands(
    reference_piper: PiperTTS, stt: WhisperSTT, matcher: Matcher, phrase: str, command_id: str
) -> None:
    transcript = stt.transcribe(say(reference_piper, phrase))
    log.warning("Whisper: «%s» → «%s» за %.0f мс", phrase, transcript.text, transcript.duration_ms)
    result = matcher.match(transcript.text, NO_WINDOW)
    assert result.matched and result.matches[0].command.id == command_id, transcript.text


def test_wake_word_and_command_in_one_phrase(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    audio = np.concatenate([gap(1.0), say(piper, "Джарвис, открой ютуб."), gap(1.5)])
    handled = run_loop(audio, vosk_model, stt)
    assert len(handled) == 1, handled


def test_name_then_pause_then_command(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    audio = np.concatenate(
        [gap(1.0), say(piper, "Джарвис."), gap(1.2), say(piper, "Сделай громче."), gap(1.5)]
    )
    handled = run_loop(audio, vosk_model, stt)
    assert len(handled) == 1, handled


@pytest.mark.parametrize(
    "phrase", ["Сделай громче.", "Сегодня хорошая погода.", "Давай посмотрим кино вечером."]
)
def test_phrase_without_name_is_ignored(
    piper: PiperTTS, vosk_model: object, stt: WhisperSTT, phrase: str
) -> None:
    audio = np.concatenate([gap(1.0), say(piper, phrase), gap(1.5)])
    assert run_loop(audio, vosk_model, stt) == []


@pytest.mark.parametrize(
    ("answer", "expected"), [("Да.", True), ("Подтверждаю.", True), ("Нет.", False), ("Не надо.", False)]
)
def test_voice_confirmation(
    reference_piper: PiperTTS, vosk_model: object, answer: str, expected: bool
) -> None:
    assert confirm_by_voice(reference_piper, vosk_model, answer) is expected


@pytest.mark.parametrize("phrase", ["Сделай громче.", "Сегодня хорошая погода."])
def test_unrelated_speech_is_not_consent(piper: PiperTTS, vosk_model: object, phrase: str) -> None:
    """Посторонняя фраза во время подтверждения не должна выполнить рискованное действие."""
    assert confirm_by_voice(piper, vosk_model, phrase) is False


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI есть только в Windows")
def test_sapi_voice_synthesizes() -> None:
    """Голос Windows синтезирует речь (в CI установлены только английские голоса)."""
    import threading

    from assistant.tts.sapi_tts import SapiTTS

    result: dict[str, Any] = {}

    def run() -> None:
        try:
            sapi = SapiTTS()
            result["voices"] = sapi.available_voices()
            result["text"] = "Проверка голоса" if sapi.has_russian_voice else "Voice check"
            result["audio"] = sapi.synthesize(result["text"])
        except Exception as exc:  # ошибку из потока показываем в тексте теста
            result["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=30)
    log.warning("Голоса SAPI: %s", result.get("voices"))
    assert not worker.is_alive(), "SAPI завис при синтезе"
    assert "error" not in result, result.get("error")
    audio, rate = result["audio"]
    log.warning("SAPI: «%s» → %.1f с звука, %d Гц", result["text"], len(audio) / rate, rate)
    assert len(audio) > rate * 0.3
    assert float(np.abs(audio).max()) > 0.01  # не тишина
