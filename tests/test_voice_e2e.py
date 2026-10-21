"""Сквозная проверка голоса на настоящих моделях: Piper → VAD → Vosk → Whisper → матчер.

Речь синтезируется голосом Piper и подаётся в голосовой цикл вместо микрофона.
Нужны скачанные модели (uv run python -m assistant --download-models), поэтому тест
включается переменной JARVIS_E2E=1 (так сделано в CI на Windows).
"""

from __future__ import annotations

import logging
import os
import sys
import time

import numpy as np
import pytest

from assistant.audio.capture import ArraySource, resample, to_int16
from assistant.audio.vad import SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import GrammarSpotter, load_vosk_model
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
NORMALIZER = Normalizer(AppConfig().wake_words, AppConfig().matcher.filler_words)


@pytest.fixture(scope="module")
def config() -> AppConfig:
    config = AppConfig()
    config._base_dir = ROOT
    return config


@pytest.fixture(scope="module")
def piper(config: AppConfig) -> PiperTTS:
    return PiperTTS(config.resolve(config.tts.piper_voice))


@pytest.fixture(scope="module")
def vosk_model(config: AppConfig) -> object:
    return load_vosk_model(config.resolve(config.voice.vosk_model))


@pytest.fixture(scope="module")
def stt(config: AppConfig) -> WhisperSTT:
    stt = WhisperSTT(config.stt, config.resolve(config.stt.download_dir), lambda: "Джарвис, открой, сделай.")
    stt.load()
    return stt


def say(piper: PiperTTS, text: str) -> np.ndarray:
    """Синтезированная фраза в формате микрофона: 16 кГц, int16."""
    audio, rate = piper.synthesize(text)
    return to_int16(resample(audio, rate))


def gap(seconds: float) -> np.ndarray:
    return np.zeros(int(16_000 * seconds), dtype=np.int16)


def run_loop(audio: np.ndarray, vosk_model: object, stt: WhisperSTT, name: str = "джарвис") -> list[str]:
    handled: list[str] = []
    loop = VoiceLoop(
        source=ArraySource(audio, silence_after=2.0),
        vad=SileroVAD(),
        segmenter=SpeechSegmenter(),
        spotter=GrammarSpotter(vosk_model, [name]),
        stt=stt,
        handle_text=lambda text, conf: handled.append(text),
        speech=FakeSpeech(),
        normalizer=NORMALIZER,
        settings=VoiceSettings(followup_seconds=3, listen_timeout=4),
        status=RecordingStatus(),
    )
    loop.run(stop_when_exhausted=True)
    return handled


def test_wake_word_is_in_vosk_vocabulary(vosk_model: object) -> None:
    spotter = GrammarSpotter(vosk_model, ["джарвис"])
    log.warning("Слово «джарвис» в словаре Vosk: %s", spotter.usable)
    assert spotter.usable, "«джарвис» нет в словаре vosk-model-small-ru — нужно другое имя или подход"


def test_whisper_recognizes_commands(
    piper: PiperTTS, stt: WhisperSTT, store: object, matcher: Matcher
) -> None:
    for phrase, command_id in [
        ("Сделай громче.", "volume_up"),
        ("Громкость на пятьдесят.", "volume_set"),
        ("Открой ютуб.", "open_youtube"),
        ("Который час?", "time"),
    ]:
        audio = say(piper, phrase)
        started = time.perf_counter()
        transcript = stt.transcribe(audio)
        elapsed = (time.perf_counter() - started) * 1000
        log.warning("Whisper: «%s» → «%s» за %.0f мс", phrase, transcript.text, elapsed)
        result = matcher.match(transcript.text, NO_WINDOW)
        assert result.matched and result.matches[0].command.id == command_id, transcript.text


def test_wake_word_and_command_in_one_phrase(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    audio = np.concatenate([gap(1.0), say(piper, "Джарвис, открой ютуб."), gap(1.5)])
    handled = run_loop(audio, vosk_model, stt)
    assert len(handled) == 1, handled
    assert NORMALIZER.normalize(handled[0]) == "открой ютуб"


def test_name_then_pause_then_command(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    audio = np.concatenate(
        [gap(1.0), say(piper, "Джарвис."), gap(1.2), say(piper, "Сделай громче."), gap(1.5)]
    )
    handled = run_loop(audio, vosk_model, stt)
    assert [NORMALIZER.normalize(t) for t in handled] == ["сделай громче"]


def test_phrase_without_name_is_ignored(piper: PiperTTS, vosk_model: object, stt: WhisperSTT) -> None:
    audio = np.concatenate([gap(1.0), say(piper, "Сделай громче."), gap(1.5)])
    assert run_loop(audio, vosk_model, stt) == []


@pytest.mark.parametrize(("answer", "expected"), [("Да.", True), ("Нет.", False), ("Подтверждаю.", True)])
def test_voice_confirmation(piper: PiperTTS, vosk_model: object, answer: str, expected: bool) -> None:
    source = ArraySource(np.concatenate([gap(0.5), say(piper, answer), gap(1.0)]), silence_after=0.5)
    confirmer = VoiceConfirmer(
        FakeSpeech(), source, lambda: GrammarSpotter(vosk_model, YES_PHRASES + NO_PHRASES), timeout=5
    )
    assert confirmer.confirm("Выключить компьютер?") is expected


@pytest.mark.skipif(sys.platform != "win32", reason="SAPI есть только в Windows")
def test_sapi_voice_synthesizes() -> None:
    from assistant.tts.base import TTSError
    from assistant.tts.sapi_tts import SapiTTS

    try:
        audio, rate = SapiTTS().synthesize("Проверка голоса")
    except TTSError as exc:
        pytest.skip(f"SAPI недоступен: {exc}")
    log.warning("SAPI: %.1f с звука, %d Гц", len(audio) / rate, rate)
    assert len(audio) > rate * 0.3
