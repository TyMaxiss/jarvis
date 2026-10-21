"""Тесты частей голоса: VAD, границы фразы, очередь озвучки, подтверждение, Whisper, загрузчик."""

from __future__ import annotations

import io
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from assistant.audio.capture import BLOCK_SIZE, ArraySource, resample, to_int16
from assistant.audio.sounds import SOUND_RATE, listen_chime
from assistant.audio.vad import SegmenterSettings, SegmentEvent, SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import SpotResult, parse_result
from assistant.config import AppConfig, SttConfig
from assistant.models import _safe_extract, download_all
from assistant.safety.confirm import VoiceConfirmer
from assistant.stt.whisper_stt import WhisperSTT, build_prompt, is_hallucination, resolve_settings
from assistant.tts.base import SpeechQueue, TeeSpeaker, split_sentences
from tests.voice_fakes import FakeSource, FakeSpeech, silence, speech

# --- VAD и границы фразы -----------------------------------------------------------


def test_silero_vad_streaming() -> None:
    vad = SileroVAD()
    quiet = [vad(np.zeros(BLOCK_SIZE, dtype=np.int16)) for _ in range(20)]
    assert max(quiet) < 0.1
    with pytest.raises(ValueError):
        vad(np.zeros(100, dtype=np.int16))
    vad.reset()
    assert vad(np.zeros(BLOCK_SIZE, dtype=np.float32)) < 0.1


def feed_probs(segmenter: SpeechSegmenter, probs: list[float]) -> list[SegmentEvent]:
    return [segmenter.feed(np.full(BLOCK_SIZE, i, dtype=np.int16), p) for i, p in enumerate(probs)]


def test_segment_start_and_end_with_pre_roll() -> None:
    seg = SpeechSegmenter(SegmenterSettings(min_silence_ms=640, pre_roll_ms=320))
    probs = [0.0] * 15 + [0.9] * 20 + [0.0] * 25
    events = feed_probs(seg, probs)
    assert events.index(SegmentEvent.START) == 15
    end = events.index(SegmentEvent.END)
    assert end == 15 + 20 + 20 - 1  # 20 блоков тишины = 640 мс
    first_block = int(seg.segment_audio[0])
    assert first_block == 15 - 9  # 10 блоков (320 мс) до начала речи, включая текущий
    assert len(seg.segment_audio) // BLOCK_SIZE <= 10 + 20 + 7  # хвост тишины обрезан до ~200 мс


def test_short_noise_is_discarded() -> None:
    seg = SpeechSegmenter(SegmenterSettings(min_silence_ms=320, min_speech_ms=200))
    events = feed_probs(seg, [0.9] * 3 + [0.0] * 12)
    assert SegmentEvent.DISCARD in events and SegmentEvent.END not in events


def test_hysteresis_keeps_phrase_going() -> None:
    seg = SpeechSegmenter(SegmenterSettings(min_silence_ms=320))
    events = feed_probs(seg, [0.9] * 10 + [0.4] * 30 + [0.0] * 12)
    assert events.count(SegmentEvent.END) == 1
    assert events.index(SegmentEvent.END) > 40


def test_max_phrase_length() -> None:
    seg = SpeechSegmenter(SegmenterSettings(max_seconds=2.0))
    events = feed_probs(seg, [0.9] * 100)
    assert SegmentEvent.END in events and seg.truncated


def test_audio_helpers() -> None:
    tone = np.sin(np.linspace(0, 100, 22050)).astype(np.float32)
    resampled = resample(tone, 22050, 16000)
    assert abs(len(resampled) - 16000) <= 1
    assert to_int16(np.array([2.0, -2.0, 0.5])).tolist() == [32767, -32767, 16383]
    source = ArraySource(np.ones(BLOCK_SIZE * 2, dtype=np.int16), silence_after=0.064)
    assert [source.read(0) is None for _ in range(5)] == [False, False, False, False, True]
    chime = listen_chime()
    assert 0.1 < len(chime) / SOUND_RATE < 0.3 and np.abs(chime).max() <= 0.3


# --- Озвучка -----------------------------------------------------------------------


class FakeEngine:
    def __init__(self, name: str = "fake", fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.texts: list[str] = []

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        if self.fail:
            raise RuntimeError("нет голоса")
        self.texts.append(text)
        return np.full(100, 0.8, dtype=np.float32), 1000


class FakePlayer:
    def __init__(self, block: threading.Event | None = None) -> None:
        self.played: list[np.ndarray] = []
        self.block = block
        self.stopped = threading.Event()

    def reset(self) -> None:
        self.stopped.clear()

    def play(self, audio: np.ndarray, samplerate: int) -> bool:
        self.played.append(audio)
        if self.block is not None:
            while not self.block.is_set() and not self.stopped.is_set():
                time.sleep(0.01)
        return not self.stopped.is_set()

    def stop(self) -> None:
        self.stopped.set()


def test_split_sentences() -> None:
    assert split_sentences("Готово. Громкость 50%! А теперь? Да…  Ок") == [
        "Готово.", "Громкость 50%!", "А теперь?", "Да…", "Ок",
    ]  # fmt: skip
    assert split_sentences("  ") == []


def test_speech_queue_is_non_blocking_and_ordered() -> None:
    release = threading.Event()
    engine, player = FakeEngine(), FakePlayer(release)
    queue = SpeechQueue(engine, player, volume=2.0)
    started = time.perf_counter()
    queue.say("Раз. Два.")
    assert time.perf_counter() - started < 0.05  # say() не ждёт синтеза
    assert queue.speaking
    release.set()
    assert queue.wait_idle(timeout=5)
    assert engine.texts == ["Раз.", "Два."]
    assert not queue.speaking
    assert player.played[0].max() == 1.0  # громкость ×2 с ограничением
    queue.close()


def test_speech_queue_stop_drops_pending() -> None:
    release = threading.Event()
    engine, player = FakeEngine(), FakePlayer(release)
    queue = SpeechQueue(engine, player)
    queue.say("Первая. Вторая. Третья.")
    time.sleep(0.1)
    queue.stop()
    assert queue.wait_idle(timeout=5)
    assert engine.texts == ["Первая."]  # остальные не синтезировались
    queue.say("Новая.")
    release.set()
    assert queue.wait_idle(timeout=5)
    assert engine.texts == ["Первая.", "Новая."]
    queue.close()


def test_speech_queue_fallback_engine() -> None:
    fallback = FakeEngine("sapi")
    queue = SpeechQueue(FakeEngine("piper", fail=True), FakePlayer(), fallback=fallback)
    queue.say("Привет")
    assert queue.wait_idle(timeout=5)
    assert fallback.texts == ["Привет"]
    queue.close()


def test_tee_speaker() -> None:
    first, second = FakeSpeech(), FakeSpeech()
    TeeSpeaker(first, second).say("ok")
    assert first.said == second.said == ["ok"]


# --- Голосовое подтверждение -------------------------------------------------------


class AnswerSpotter:
    """Как Vosk: когда в звуке есть речь, «распознаёт» заданное слово с заданной уверенностью."""

    def __init__(self, answer: str | None, conf: float = 0.95) -> None:
        self.answer = answer
        self.conf = conf

    def accept(self, block: np.ndarray) -> SpotResult | None:
        if block.any() and self.answer:
            return SpotResult(self.answer, tuple((w, self.conf) for w in self.answer.split()))
        return None

    def finish(self) -> SpotResult:
        return SpotResult()

    def reset(self) -> None:
        pass


@pytest.mark.parametrize(
    ("answer", "expected"), [("да", True), ("подтверждаю", True), ("нет", False), ("не надо", False)]
)
def test_voice_confirmer_answers(answer: str, expected: bool) -> None:
    out = FakeSpeech()
    source = FakeSource(np.concatenate([silence(300), speech(3000, 300)]))
    confirmer = VoiceConfirmer(out, source, lambda: AnswerSpotter(answer), timeout=5)
    assert confirmer.confirm("Выключить компьютер?") is expected
    assert out.said == ["Выключить компьютер?"]
    assert source.flushes == 1


def test_voice_confirmer_silence_means_no() -> None:
    confirmer = VoiceConfirmer(
        FakeSpeech(), FakeSource(silence(6000)), lambda: AnswerSpotter("да"), timeout=5
    )
    assert confirmer.confirm("Точно?") is False


def test_voice_confirmer_ignores_unsure_yes() -> None:
    source = FakeSource(np.concatenate([silence(300), speech(3000, 300)]))
    confirmer = VoiceConfirmer(
        FakeSpeech(), source, lambda: AnswerSpotter("да", conf=0.4), timeout=1, min_confidence=0.7
    )
    assert confirmer.confirm("Точно?") is False


def test_spot_result_find() -> None:
    result = SpotResult("джарвис [unk]", (("джарвис", 0.9), ("[unk]", 1.0)))
    assert result.find(["джарвис"], 0.8) == "джарвис"
    assert result.find(["джарвис"], 0.95) is None
    assert result.known_words == ["джарвис"]
    both = SpotResult("да", (("да", 0.9),)) + SpotResult("не надо", (("не", 0.9), ("надо", 0.8)))
    assert both.find(["не надо", "да"], 0.7) == "не надо"
    assert both.find(["не надо"], 0.85) is None
    assert parse_result('{"text": "да", "result": [{"word": "да", "conf": 0.75}]}').words == (("да", 0.75),)


# --- Whisper ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Субтитры сделал DimaTorzok", True),
        ("Продолжение следует...", True),
        ("Спасибо за просмотр!", True),
        ("Открой редактор кода", False),
        ("Сделай громче", False),
    ],
)
def test_hallucinations(text: str, expected: bool) -> None:
    assert is_hallucination(text) is expected


def test_prompt_contains_name_words_and_apps() -> None:
    phrases = ["сделай громче", "сделай тише", "открой {name}", "открой ютуб"]
    prompt = build_prompt("джарвис", phrases, ["Telegram", "Discord", "Telegram"], max_apps=5)
    assert prompt.startswith("Джарвис, сделай, открой")
    assert "Telegram, Discord." in prompt and prompt.count("Telegram") == 1
    assert "{name}" not in prompt
    long_prompt = build_prompt("джарвис", phrases, [f"Приложение {i}" for i in range(500)], max_apps=500)
    assert len(long_prompt) <= 600


@pytest.mark.parametrize(
    ("settings", "has_cuda", "expected"),
    [
        (SttConfig(), True, ("large-v3-turbo", "cuda", "float16")),
        (SttConfig(), False, ("small", "cpu", "int8")),
        (SttConfig(model="medium", device="cpu"), True, ("medium", "cpu", "int8")),
        (SttConfig(compute_type="int8_float16"), True, ("large-v3-turbo", "cuda", "int8_float16")),
    ],
)
def test_resolve_settings(settings: SttConfig, has_cuda: bool, expected: tuple[str, str, str]) -> None:
    assert resolve_settings(settings, has_cuda) == expected


def test_cuda_failure_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded: list[tuple[str, str, str]] = []

    def fake_load(self: WhisperSTT, model: str, device: str, compute: str) -> None:
        loaded.append((model, device, compute))
        self._model = object()

    def failing_warm_up(self: WhisperSTT) -> None:
        raise RuntimeError("Library cublas64_12.dll is not found")

    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr(WhisperSTT, "_load", fake_load)
    monkeypatch.setattr(WhisperSTT, "_warm_up", failing_warm_up)
    WhisperSTT(SttConfig(), tmp_path).load()
    assert loaded == [("large-v3-turbo", "cuda", "float16"), ("small", "cpu", "int8")]


# --- Загрузка моделей ----------------------------------------------------------------


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../evil.txt", "x")
    with zipfile.ZipFile(buffer) as archive, pytest.raises(ValueError):
        _safe_extract(archive, tmp_path / "models")
    assert not (tmp_path / "evil.txt").exists()


def test_download_all_reports_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("нет сети")

    monkeypatch.setattr("assistant.models.ensure_vosk", boom)
    monkeypatch.setattr("assistant.models.ensure_piper", lambda *a, **k: None)
    monkeypatch.setattr("assistant.models.ensure_whisper", lambda *a, **k: None)
    out = io.StringIO()
    assert download_all(AppConfig(), out) is False
    assert "Vosk: ошибка — нет сети" in out.getvalue()
