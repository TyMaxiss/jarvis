"""Тесты частей голоса: VAD, границы фразы, очередь озвучки, подтверждение, Whisper, загрузчик."""

from __future__ import annotations

import io
import json
import threading
import time
import zipfile
from pathlib import Path

import numpy as np
import pytest

from assistant.audio.capture import BLOCK_MS, BLOCK_SIZE, ArraySource, block_level, resample, to_int16
from assistant.audio.sounds import SOUND_RATE, alarm_chime, listen_chime
from assistant.audio.vad import SegmenterSettings, SegmentEvent, SileroVAD, SpeechSegmenter
from assistant.audio.wakeword import SpotResult, parse_result, phrase_spotter, wake_spotter
from assistant.config import AppConfig, SttConfig
from assistant.models import _safe_extract, download_all
from assistant.safety.confirm import VoiceConfirmer
from assistant.stt.whisper_stt import (
    ModelDownloadError,
    WhisperSTT,
    build_prompt,
    cached_model,
    is_hallucination,
    resolve_settings,
    whisper_download_progress,
    whisper_repo,
)
from assistant.tts.base import SpeechQueue, TeeSpeaker, split_sentences
from assistant.tts.sapi_tts import sapi_rate, token_languages
from tests.conftest import ROOT, wait_until
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


def test_speech_span_counts_pauses_inside_phrase() -> None:
    seg = SpeechSegmenter(SegmenterSettings(min_silence_ms=640, pre_roll_ms=320))
    feed_probs(seg, [0.0] * 15 + [0.9] * 20 + [0.0] * 25)
    assert seg.speech_span_ms == 20 * BLOCK_MS  # без запаса до речи и тишины после
    feed_probs(seg, [0.9] * 10 + [0.0] * 10 + [0.9] * 10 + [0.0] * 25)
    assert seg.speech_span_ms == 30 * BLOCK_MS  # пауза внутри фразы входит


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
    alarm = alarm_chime()  # таймер: дольше и громче, но без перегрузки
    assert 1.0 < len(alarm) / SOUND_RATE < 2.0 and 0.3 < np.abs(alarm).max() <= 0.5


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
    # Пока звучала первая, вторая синтезировалась впрок; третья — уже нет.
    assert engine.texts == ["Первая.", "Вторая."] and len(player.played) == 1
    queue.say("Новая.")
    release.set()
    assert queue.wait_idle(timeout=5)
    assert engine.texts[-1] == "Новая." and len(player.played) == 2  # «Вторая» не прозвучала
    queue.close()


def test_speech_queue_fallback_engine() -> None:
    fallback = FakeEngine("sapi")
    queue = SpeechQueue(FakeEngine("piper", fail=True), FakePlayer(), fallback=fallback)
    queue.say("Привет")
    assert queue.wait_idle(timeout=5)
    assert fallback.texts == ["Привет"]
    queue.close()


def test_speech_queue_switches_to_downloaded_voice() -> None:
    sapi, piper = FakeEngine("sapi"), FakeEngine("piper", fail=True)
    queue = SpeechQueue(sapi, FakePlayer())
    queue.say("Пока голос Windows.")
    assert queue.wait_idle(timeout=5)
    queue.switch(piper)  # голос Piper скачался
    queue.say("Теперь Piper.")
    assert queue.wait_idle(timeout=5)
    # Piper не смог — прежний голос стал запасным.
    assert queue.engine is piper and sapi.texts == ["Пока голос Windows.", "Теперь Piper."]
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
    ("answer", "expected"),
    [
        ("да", True),
        ("подтверждаю", True),
        ("да конечно", True),
        ("нет", False),
        ("не надо", False),
        ("давай", False),
        ("[unk] да", False),  # «да» среди посторонних слов — не согласие
        ("погода да", False),  # слово-приманка рядом с «да» — тоже
        ("да [unk] нет", False),
    ],
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


class FakeVoiceToken:
    """Голос SAPI с атрибутом Language (None — атрибута нет)."""

    def __init__(self, language: str | None) -> None:
        self.language = language

    def GetAttribute(self, name: str) -> str:
        if self.language is None:
            raise OSError("атрибута нет")
        return self.language


@pytest.mark.parametrize(
    ("value", "expected"), [("419", ["419"]), ("409;9", ["409", "9"]), ("0419", ["419"]), (None, [])]
)
def test_sapi_token_languages(value: str | None, expected: list[str]) -> None:
    assert token_languages(FakeVoiceToken(value)) == expected


def test_sapi_rate() -> None:
    assert sapi_rate(1.0) == 0
    assert sapi_rate(2.0) == 6 and sapi_rate(0.5) == -6
    assert sapi_rate(100) == 10


class FakeVoskModel:
    """Словарь модели Vosk: слова из vocabulary в нём есть, остальных нет."""

    def __init__(self, vocabulary: set[str]) -> None:
        self.vocabulary = vocabulary

    def vosk_model_find_word(self, word: str) -> int:
        return 1 if word in self.vocabulary else -1


def test_wake_spotter_grammar_has_decoys(monkeypatch: pytest.MonkeyPatch) -> None:
    import vosk

    grammars: list[list[str]] = []

    class FakeRecognizer:
        def __init__(self, model: object, rate: int, grammar: str) -> None:
            grammars.append(json.loads(grammar))

        def SetWords(self, enabled: bool) -> None:
            pass

    monkeypatch.setattr(vosk, "KaldiRecognizer", FakeRecognizer)
    model = FakeVoskModel({"джарвис", "который", "час", "компьютер"})
    spotter = wake_spotter(model, "Джарвис")
    assert spotter.decoys == ["который", "час", "компьютер"]  # слов не из словаря модели нет
    assert grammars[-1] == ["джарвис", "который", "час", "компьютер", "[unk]"]
    renamed = wake_spotter(model, "Компьютер")  # имя совпало с приманкой — остаётся только именем
    assert renamed.phrases == ["компьютер"] and "компьютер" not in renamed.decoys


def test_stop_spotter_has_no_soundalike_decoys(monkeypatch: pytest.MonkeyPatch) -> None:
    # «стоп» Vosk принимал за похожие приманки «тот» и «сам» — в грамматике стоп-слов их нет.
    import vosk

    from assistant.voice import STOP_PHRASES, STOP_SOUNDALIKES

    class FakeRecognizer:
        def __init__(self, model: object, rate: int, grammar: str) -> None:
            pass

        def SetWords(self, enabled: bool) -> None:
            pass

    monkeypatch.setattr(vosk, "KaldiRecognizer", FakeRecognizer)
    model = FakeVoskModel({*STOP_PHRASES, "тот", "сам", "час", "погода"})
    spotter = phrase_spotter(model, STOP_PHRASES, exclude_decoys=STOP_SOUNDALIKES)
    assert spotter.phrases == list(STOP_PHRASES)
    assert spotter.decoys == ["час", "погода"]


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
    monkeypatch.setattr("assistant.stt.whisper_stt.missing_cuda_library", lambda: "")
    monkeypatch.setattr(WhisperSTT, "_load", fake_load)
    monkeypatch.setattr(WhisperSTT, "_warm_up", failing_warm_up)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    assert loaded == [("large-v3-turbo", "cuda", "float16"), ("small", "cpu", "int8")]
    assert "Library cublas64_12.dll is not found" in stt.cuda_problem


def test_without_cuda_libraries_gpu_model_is_not_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[tuple[str, str, str]] = []

    def fake_load(self: WhisperSTT, model: str, device: str, compute: str) -> None:
        loaded.append((model, device, compute))
        self._model, self.device = object(), device

    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr("assistant.stt.whisper_stt.missing_cuda_library", lambda: "cublas64_12.dll")
    monkeypatch.setattr(WhisperSTT, "_load", fake_load)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    # large-v3-turbo для видеокарты не скачивается и не загружается впустую.
    assert loaded == [("small", "cpu", "int8")] and stt.device == "cpu"
    assert "нет библиотек CUDA (cublas64_12.dll)" in stt.cuda_problem
    assert "uv sync --extra cuda" in stt.cuda_problem and "«Процессор»" in stt.cuda_problem


def gpu_without_libraries(
    monkeypatch: pytest.MonkeyPatch, after_fetch: str
) -> tuple[list[tuple[str, str, str]], list[bool]]:
    """Видеокарта есть, DLL CUDA нет; когда в список fetched что-то добавят (скачали),
    недостаёт after_fetch ('' — ничего). Возвращает загруженные модели и fetched."""
    loaded: list[tuple[str, str, str]] = []
    fetched: list[bool] = []

    def fake_load(self: WhisperSTT, model: str, device: str, compute: str) -> None:
        loaded.append((model, device, compute))
        self._model, self.device = object(), device

    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr(
        "assistant.stt.whisper_stt.missing_cuda_library",
        lambda: after_fetch if fetched else "cublas64_12.dll",
    )
    monkeypatch.setattr("assistant.stt.whisper_stt.register_cuda_dlls", lambda: None)
    monkeypatch.setattr(WhisperSTT, "_load", fake_load)
    monkeypatch.setattr(WhisperSTT, "_warm_up", lambda self: None)
    return loaded, fetched


def test_missing_cuda_libraries_are_downloaded_before_gpu_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded, fetched = gpu_without_libraries(monkeypatch, after_fetch="")
    stt = WhisperSTT(SttConfig(), tmp_path, fetch_cuda=lambda: fetched.append(True))
    stt.load()
    assert fetched == [True] and loaded == [("large-v3-turbo", "cuda", "float16")]
    assert stt.cuda_problem == ""


def test_failed_cuda_download_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded, _ = gpu_without_libraries(monkeypatch, after_fetch="")

    def offline() -> None:
        raise OSError("нет сети")

    stt = WhisperSTT(SttConfig(), tmp_path, fetch_cuda=offline)
    stt.load()
    assert loaded == [("small", "cpu", "int8")]
    assert "не удалось скачать библиотеки CUDA (нет сети)" in stt.cuda_problem
    assert "при следующем запуске" in stt.cuda_problem and "uv sync" not in stt.cuda_problem


def test_downloaded_cuda_library_that_does_not_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded, fetched = gpu_without_libraries(monkeypatch, after_fetch="cudnn_ops64_9.dll")
    stt = WhisperSTT(SttConfig(), tmp_path, fetch_cuda=lambda: fetched.append(True))
    stt.load()
    assert loaded == [("small", "cpu", "int8")]
    assert "cudnn_ops64_9.dll не загружается" in stt.cuda_problem and "Удалите папку" in stt.cuda_problem


def test_whisper_repo_like_faster_whisper() -> None:
    assert whisper_repo("small") == "Systran/faster-whisper-small"
    assert whisper_repo("large-v3-turbo") == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert whisper_repo("someone/my-model") == "someone/my-model"
    assert whisper_repo("unknown") == ""


def test_whisper_download_progress_watches_model_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Window(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.steps: list[tuple[str, float | None]] = []

        def progress(self, text: str, fraction: float | None) -> None:
            self.steps.append((text, fraction))

    monkeypatch.setattr("assistant.stt.whisper_stt.whisper_download_size", lambda repo: 4 << 20)
    window = Window()
    folder = tmp_path / "models--Systran--faster-whisper-small" / "blobs"
    with whisper_download_progress("small", tmp_path, window):
        folder.mkdir(parents=True)
        (folder / "abc.incomplete").write_bytes(bytes(4 << 20))
        assert wait_until(lambda: bool(window.steps) and window.steps[-1][1] == 1.0, timeout=5)
    assert "Скачиваю модель распознавания речи Whisper small (один раз)" in window.getvalue()
    assert window.steps[-1] == ("Скачиваю модель Whisper small: 100% (4 МБ из 4 МБ)", 1.0)
    with whisper_download_progress("small", tmp_path, None):  # некому сообщать
        pass


def fake_cache(folder: Path, model: str = "small") -> Path:
    """Модель Whisper в кэше Hugging Face — так её раскладывает snapshot_download."""
    commit = "0123456789abcdef0123456789abcdef01234567"
    repo = folder / f"models--{whisper_repo(model).replace('/', '--')}"
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(commit, encoding="utf-8")
    snapshot = repo / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.json"):
        (snapshot / name).write_text("{}", encoding="utf-8")
    return snapshot


def fake_whisper_model(
    monkeypatch: pytest.MonkeyPatch, fail_on: str = "", error: str = ""
) -> list[tuple[Path, str]]:
    """Вместо faster_whisper.WhisperModel: запоминает (папка, устройство); на fail_on — error."""
    loaded: list[tuple[Path, str]] = []

    def create(path: str, *, device: str, **options: object) -> object:
        loaded.append((Path(path), device))
        if device == fail_on:
            raise RuntimeError(error)
        return object()

    monkeypatch.setattr("faster_whisper.WhisperModel", create)
    return loaded


def no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    def fetch(*args: object) -> str:
        pytest.fail("лишнее скачивание")

    monkeypatch.setattr("assistant.stt.whisper_stt.fetch_model", fetch)


def test_cached_model_is_found_without_network(tmp_path: Path) -> None:
    assert cached_model("small", tmp_path) is None
    snapshot = fake_cache(tmp_path)
    assert cached_model("small", tmp_path) == str(snapshot)
    assert cached_model("large-v3-turbo", tmp_path) is None
    assert cached_model(str(snapshot), tmp_path / "other") == str(snapshot)  # своя папка с моделью


def test_downloaded_model_is_loaded_from_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot = fake_cache(tmp_path)
    loaded = fake_whisper_model(monkeypatch)
    no_downloads(monkeypatch)
    out = io.StringIO()
    stt = WhisperSTT(SttConfig(device="cpu"), tmp_path, out=out)
    stt.load()
    assert loaded == [(snapshot, "cpu")] and (stt.model_name, stt.device) == ("small", "cpu")
    assert out.getvalue() == ""


def test_model_on_disk_that_does_not_load_is_not_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Так было после обновления: модель на диске не загрузилась, а окно показывало «Скачиваю
    # модель…» и затем «Нужен интернет для первой загрузки» — вместо настоящей причины.
    fake_cache(tmp_path)
    fake_whisper_model(monkeypatch, fail_on="cpu", error="Unable to open file 'model.bin'")
    no_downloads(monkeypatch)
    out = io.StringIO()
    with pytest.raises(RuntimeError, match="Unable to open file 'model.bin'"):
        WhisperSTT(SttConfig(device="cpu"), tmp_path, out=out).load()
    assert "Скачиваю" not in out.getvalue()


def test_missing_model_is_downloaded_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    loaded = fake_whisper_model(monkeypatch)
    fetched: list[str] = []

    def fetch(model: str, folder: Path, out: object) -> str:
        fetched.append(model)
        return str(fake_cache(folder, model))

    monkeypatch.setattr("assistant.stt.whisper_stt.fetch_model", fetch)
    WhisperSTT(SttConfig(device="cpu"), tmp_path).load()
    assert fetched == ["small"] and [device for _, device in loaded] == ["cpu"]


def test_failed_download_is_reported_as_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from huggingface_hub.errors import LocalEntryNotFoundError

    def download_model(model: str, *, cache_dir: str, local_files_only: bool = False) -> str:
        if local_files_only:
            raise LocalEntryNotFoundError("нет в кэше")
        raise ConnectionError("нет сети")

    monkeypatch.setattr("faster_whisper.utils.download_model", download_model)
    with pytest.raises(ModelDownloadError, match=r"не удалось скачать модель Whisper small \(нет сети\)"):
        WhisperSTT(SttConfig(device="cpu"), tmp_path).load()


@pytest.mark.parametrize(
    ("on_disk", "expected"),
    [
        # Модели для процессора нет: та, что для видеокарты, работает и на процессоре —
        # из-за того, что видеокарта не запустилась, полгигабайта не качаются.
        (["large-v3-turbo"], "large-v3-turbo"),
        (["large-v3-turbo", "small"], "small"),
    ],
)
def test_gpu_failure_falls_back_to_model_on_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on_disk: list[str], expected: str
) -> None:
    for model in on_disk:
        fake_cache(tmp_path, model)
    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr("assistant.stt.whisper_stt.missing_cuda_library", lambda: "")
    loaded = fake_whisper_model(monkeypatch, fail_on="cuda", error="CUDA failed with error out of memory")
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    assert [device for _, device in loaded] == ["cuda", "cpu"]
    assert (stt.model_name, stt.device) == (expected, "cpu")
    assert "out of memory" in stt.cuda_problem and "снова попробую видеокарту" in stt.cuda_problem


def test_snapshot_without_model_bin_is_not_a_model(tmp_path: Path) -> None:
    # Так выглядит оборванная загрузка, если список файлов модели не сохранился: остальное есть.
    snapshot = fake_cache(tmp_path)
    (snapshot / "model.bin").unlink()
    assert cached_model("small", tmp_path) is None


def test_model_file_that_did_not_open_is_tried_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Файл модели на диске, но в первый раз не открылся (его держал антивирус): вторая попытка.
    snapshot = fake_cache(tmp_path)
    tries: list[Path] = []

    def create(path: str, *, device: str, **options: object) -> object:
        tries.append(Path(path))
        if len(tries) == 1:
            raise RuntimeError(f"Unable to open file 'model.bin' in model '{path}'")
        return object()

    monkeypatch.setattr("faster_whisper.WhisperModel", create)
    monkeypatch.setattr("assistant.stt.whisper_stt.OPEN_RETRY_PAUSE", 0)
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(device="cpu"), tmp_path)
    stt.load()
    assert tries == [snapshot, snapshot] and stt.model_name == "small"


def test_model_file_that_never_opens_names_the_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_cache(tmp_path)
    fake_whisper_model(monkeypatch, fail_on="cpu", error="Unable to open file 'model.bin'")
    monkeypatch.setattr("assistant.stt.whisper_stt.OPEN_RETRY_PAUSE", 0)
    monkeypatch.setattr("assistant.stt.whisper_stt.open_problem", lambda folder: "занят другой программой")
    no_downloads(monkeypatch)
    with pytest.raises(RuntimeError, match="'model.bin'; model.bin занят другой программой"):
        WhisperSTT(SttConfig(device="cpu"), tmp_path).load()


def test_model_files_are_described_for_the_log(tmp_path: Path) -> None:
    from assistant.stt.whisper_stt import model_files

    snapshot = fake_cache(tmp_path)
    (snapshot / "model.bin").write_bytes(bytes(3 << 20))
    text = model_files(str(snapshot))
    assert "model.bin: 3 МБ" in text and "config.json: 0 МБ" in text
    assert text.endswith("model.bin открывается")


def test_cpu_falls_back_to_other_model_on_disk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # small на диске, но не загружается — голос всё равно работает: на скачанной large-v3-turbo.
    small, turbo = fake_cache(tmp_path, "small"), fake_cache(tmp_path, "large-v3-turbo")
    loaded: list[Path] = []

    def create(path: str, *, device: str, **options: object) -> object:
        loaded.append(Path(path))
        if Path(path) == small:
            raise RuntimeError("Unsupported model binary version")
        return object()

    monkeypatch.setattr("faster_whisper.WhisperModel", create)
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(device="cpu"), tmp_path)
    stt.load()
    assert loaded == [small, turbo] and (stt.model_name, stt.device) == ("large-v3-turbo", "cpu")


def test_auto_cpu_uses_model_on_disk_instead_of_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Видеокарта на этот раз не видна (cuda_available — False), скачана только large-v3-turbo:
    # на процессоре — она, а не скачивание small.
    turbo = fake_cache(tmp_path, "large-v3-turbo")
    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: False)
    loaded = fake_whisper_model(monkeypatch)
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    assert loaded == [(turbo, "cpu")] and stt.model_name == "large-v3-turbo"


def test_chosen_cpu_downloads_its_own_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # «Процессор» выбран в настройках: своя модель (small) скачивается, а не берётся большая.
    fake_cache(tmp_path, "large-v3-turbo")
    loaded = fake_whisper_model(monkeypatch)
    fetched: list[str] = []

    def fetch(model: str, folder: Path, out: object) -> str:
        fetched.append(model)
        return str(fake_cache(folder, model))

    monkeypatch.setattr("assistant.stt.whisper_stt.fetch_model", fetch)
    stt = WhisperSTT(SttConfig(device="cpu"), tmp_path)
    stt.load()
    assert fetched == ["small"] and stt.model_name == "small" and len(loaded) == 1


def test_gpu_reason_is_kept_for_the_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_cache(tmp_path, "large-v3-turbo")
    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr("assistant.stt.whisper_stt.missing_cuda_library", lambda: "")
    fake_whisper_model(monkeypatch, fail_on="cuda", error="CUDA failed with error unknown error")
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    assert stt.gpu_error == "CUDA failed with error unknown error" and stt.device == "cpu"


def test_gpu_error_without_message_still_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Ошибка без текста (MemoryError()) не должна сойти за успешный запуск на видеокарте.
    fake_cache(tmp_path, "large-v3-turbo")
    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr("assistant.stt.whisper_stt.missing_cuda_library", lambda: "")
    fake_whisper_model(monkeypatch)

    def warm_up(self: WhisperSTT) -> None:
        if self.device == "cuda":
            raise MemoryError()

    monkeypatch.setattr(WhisperSTT, "_warm_up", warm_up)
    no_downloads(monkeypatch)
    stt = WhisperSTT(SttConfig(), tmp_path)
    stt.load()
    assert stt.device == "cpu" and stt.gpu_error == "MemoryError" and stt._model is not None


def test_offline_download_without_model_bin_is_a_download_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Без связи с Hugging Face snapshot_download отдаёт папку с диска как есть — без model.bin.
    from assistant.stt.whisper_stt import fetch_model

    snapshot = fake_cache(tmp_path)
    (snapshot / "model.bin").unlink()
    monkeypatch.setattr("faster_whisper.utils.download_model", lambda model, cache_dir: str(snapshot))
    with pytest.raises(ModelDownloadError, match=r"model.bin не скачался"):
        fetch_model("small", tmp_path, None)


def test_gpu_work_waits_for_program_launches() -> None:
    # Пока голос загружает CUDA, external_launch (он на миг меняет PATH всего процесса) ждёт.
    from assistant import frozen

    taken: list[bool] = []
    with frozen.keep_dll_search():
        thread = threading.Thread(target=lambda: taken.append(frozen._LAUNCH_LOCK.acquire(timeout=0.1)))
        thread.start()
        thread.join()
        with frozen.keep_dll_search():  # из того же потока — можно
            pass
    assert taken == [False]


def test_cpu_choice_skips_gpu_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("assistant.stt.whisper_stt.cuda_available", lambda: True)
    monkeypatch.setattr(
        "assistant.stt.whisper_stt.missing_cuda_library", lambda: pytest.fail("лишняя проверка")
    )
    monkeypatch.setattr(WhisperSTT, "_load", lambda self, *args: None)
    stt = WhisperSTT(SttConfig(device="cpu"), tmp_path)
    stt.load()
    assert stt.cuda_problem == ""


def test_missing_cuda_library_searches_like_ctranslate2(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes

    from assistant.stt import whisper_stt

    searched: list[tuple[str, int]] = []

    def fake_dll(name: str, winmode: int) -> object:
        searched.append((name, winmode))
        if name == "cudnn_ops64_9.dll":
            raise FileNotFoundError(name)
        return object()

    monkeypatch.setattr(whisper_stt.sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", fake_dll, raising=False)
    assert whisper_stt.missing_cuda_library() == "cudnn_ops64_9.dll"
    # winmode=0 — обычный поиск LoadLibrary с PATH, как в CTranslate2.
    assert searched == [("cublas64_12.dll", 0), ("cudnn_ops64_9.dll", 0)]


def test_cuda_dlls_are_preloaded_by_full_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # cuBLAS и NVRTC загружены по полному пути: потом их находят по имени, и PATH (его на миг меняет
    # external_launch, пока запускается программа) видеокарте уже не нужен.
    import ctypes

    from assistant.stt import whisper_stt

    for name in ("cublas64_12.dll", "cublasLt64_12.dll", "nvrtc64_120_0.dll", "cudnn_ops64_9.dll", "x.txt"):
        (tmp_path / name).write_bytes(b"")
    loaded: list[tuple[str, int]] = []
    monkeypatch.setattr(whisper_stt.sys, "platform", "win32")
    monkeypatch.setattr(whisper_stt, "cuda_dirs", lambda: [tmp_path])
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, winmode: loaded.append((name, winmode)), raising=False)
    assert whisper_stt.missing_cuda_library() == ""
    full = [(Path(name).name, mode) for name, mode in loaded if Path(name).is_absolute()]
    assert full == [("cublasLt64_12.dll", 8), ("cublas64_12.dll", 8), ("nvrtc64_120_0.dll", 8)]
    assert loaded[-2:] == [("cublas64_12.dll", 0), ("cudnn_ops64_9.dll", 0)]


def test_cuda_libraries_match_cuda_extra() -> None:
    import tomllib

    from assistant.stt.whisper_stt import CUDA_LIBRARIES

    extra = " ".join(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"
        ]["cuda"]
    )
    assert "nvidia-cublas-cu12" in extra and "cublas64_12.dll" in CUDA_LIBRARIES
    assert "nvidia-cudnn-cu12>=9,<10" in extra and "cudnn_ops64_9.dll" in CUDA_LIBRARIES


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


def test_voice_confirmer_answer_tells_silence_from_no() -> None:
    confirmer = VoiceConfirmer(
        FakeSpeech(), FakeSource(silence(6000)), lambda: AnswerSpotter("да"), timeout=5
    )
    assert confirmer.answer("Точно?") is None
    source = FakeSource(np.concatenate([silence(300), speech(3000, 300)]))
    confirmer = VoiceConfirmer(FakeSpeech(), source, lambda: AnswerSpotter("нет"), timeout=5)
    assert confirmer.answer("Точно?") is False


def test_voice_confirmer_stops_listening_when_answered_by_button() -> None:
    cancel = threading.Event()
    cancel.set()
    source = FakeSource(np.concatenate([silence(300), speech(3000, 300)]))
    confirmer = VoiceConfirmer(FakeSpeech(), source, lambda: AnswerSpotter("да"), timeout=5)
    assert confirmer.answer("Точно?", cancel) is None
    assert source.reads == 0 and not confirmer.listening


def test_block_level() -> None:
    assert block_level(np.zeros(512, np.int16)) == 0.0
    assert block_level(np.zeros(0, np.int16)) == 0.0
    voice = (np.sin(np.linspace(0, 100, 512)) * 3000).astype(np.int16)  # обычная речь
    assert 0.3 < block_level(voice) < 0.7
    assert block_level(np.full(512, 32767, np.int16)) == 1.0
