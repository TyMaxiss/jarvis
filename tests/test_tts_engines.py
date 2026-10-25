"""Голоса edge-tts и Silero: текст для Silero, протокол процесса синтеза, запасной голос."""

from __future__ import annotations

import io
import json
import struct
import sys
import types
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from assistant.tts import edge_tts as edge_module
from assistant.tts import silero_tts
from assistant.tts.base import SpeechQueue, TTSError
from assistant.tts.edge_tts import EdgeTTS, edge_rate
from assistant.tts.silero_tts import SileroTTS, rate_word
from assistant.tts.silero_worker import pieces, run_worker
from assistant.tts.spoken import number_words, ordinal_neuter, plural, spoken_text

# --- Текст для Silero ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Громкость 50%", "Громкость пятьдесят процентов"),
        ("Громкость 21%.", "Громкость двадцать один процент."),
        ("Громкость 3%", "Громкость три процента"),
        ("Сейчас 14:35, вторник, 21 октября",
         "Сейчас четырнадцать тридцать пять, вторник, двадцать первое октября"),
        ("Сейчас 09:05, 1 января", "Сейчас девять ноль пять, первое января"),
        ("Сейчас 10:00", "Сейчас десять ноль ноль"),
        ("Подождал 2 с", "Подождал две секунды"),
        ("Встретимся в 5 с друзьями", "Встретимся в пять с друзьями"),
        ("Свободно 120,5 ГБ", "Свободно сто двадцать целых пять десятых гигабайта"),
        ("Осталось 1 мин", "Осталось одна минута"),
        ("Через 11 мин", "Через одиннадцать минут"),
        ("Температура -3°C", "Температура минус три градуса"),
        ("Открыл Telegram и YouTube", "Открыл телеграм и ютуб"),
        ("«Google Chrome» запущен", "гугл хром запущен"),
        ("Подключи USB", "Подключи ю эс би"),
        ("Открой 2GIS", "Открой два джи ай эс"),
        ("Модель large-v3-turbo", "Модель ларг ви три турбо"),
        ("C++ & Python", "си и пайтон"),
    ],
)  # fmt: skip
def test_spoken_text(text: str, expected: str) -> None:
    assert spoken_text(text) == expected


def test_number_words_and_forms() -> None:
    assert number_words(0) == "ноль"
    assert number_words(2026) == "две тысячи двадцать шесть"
    assert number_words(1_000_001) == "один миллион один"
    assert number_words(-12) == "минус двенадцать"
    assert number_words(2, feminine=True) == "две"
    assert [plural(n, ("процент", "процента", "процентов")) for n in (1, 2, 5, 11, 21, 112)] == [
        "процент", "процента", "процентов", "процентов", "процент", "процентов",
    ]  # fmt: skip
    assert [ordinal_neuter(day) for day in (1, 3, 11, 20, 22, 31)] == [
        "первое", "третье", "одиннадцатое", "двадцатое", "двадцать второе", "тридцать первое",
    ]  # fmt: skip


def test_only_cyrillic_remains() -> None:
    text = spoken_text("Ответ: 42! (см. https://example.com/page?x=1) — готово…")
    assert text.startswith("Ответ: сорок два!")
    assert not any(char.isdigit() or ("a" <= char.lower() <= "z") for char in text)


# --- edge-tts --------------------------------------------------------------------------


def mp3_bytes(seconds: float = 0.5, rate: int = 24_000) -> bytes:
    """Настоящий MP3, как от сервиса: синус 440 Гц (кодирует PyAV)."""
    import av

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp3") as container:
        stream = container.add_stream("mp3", rate=rate)
        stream.layout = "mono"
        samples = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * rate)) / rate)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="flt", layout="mono")
        frame.sample_rate = rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return buffer.getvalue()


def test_edge_rate() -> None:
    assert [edge_rate(rate) for rate in (1.0, 1.2, 0.8, 2.0)] == ["+0%", "+20%", "-20%", "+100%"]


def test_edge_tts_collects_audio_and_decodes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    audio = mp3_bytes()

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, **options: Any) -> None:
            calls.append({"text": text, "voice": voice, **options})

        async def stream(self) -> AsyncIterator[dict[str, Any]]:
            yield {"type": "SentenceBoundary", "offset": 0, "duration": 1, "text": "Привет"}
            for start in range(0, len(audio), 4096):
                yield {"type": "audio", "data": audio[start : start + 4096]}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=FakeCommunicate))
    monkeypatch.setattr(edge_module, "system_proxy", lambda: "http://proxy:8080")
    samples, rate = EdgeTTS("ru-RU-SvetlanaNeural", rate=1.1).synthesize("Привет")
    # MP3 добавляет немного тишины по краям; громкость синуса — около 0,3.
    assert rate == 24_000 and samples.dtype == np.float32 and 12_000 <= len(samples) < 14_000
    assert 0.25 < float(np.abs(samples).max()) < 0.35
    assert calls == [
        {
            "text": "Привет", "voice": "ru-RU-SvetlanaNeural", "rate": "+10%", "proxy": "http://proxy:8080",
            "connect_timeout": 5, "receive_timeout": 15,
        }
    ]  # fmt: skip


def test_edge_tts_without_network_is_tts_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Offline:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def stream(self) -> AsyncIterator[dict[str, Any]]:
            raise OSError("Cannot connect to host speech.platform.bing.com")
            yield {}  # pragma: no cover — делает метод асинхронным генератором

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=Offline))
    with pytest.raises(TTSError, match="edge-tts: OSError: Cannot connect"):
        EdgeTTS("ru-RU-DmitryNeural").synthesize("Привет")


# --- Очередь: запасной голос -------------------------------------------------------------


class Engine:
    def __init__(self, name: str, fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.texts: list[str] = []
        self.closed = False

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        self.texts.append(text)
        if self.fail:
            raise TTSError("нет сети")
        return np.full(10, 0.1, dtype=np.float32), 1000

    def close(self) -> None:
        self.closed = True


class Player:
    def reset(self) -> None:
        pass

    def play(self, audio: np.ndarray, samplerate: int) -> bool:
        return True

    def stop(self) -> None:
        pass


def test_failed_primary_is_skipped_for_a_while() -> None:
    now = [100.0]
    edge, piper = Engine("edge", fail=True), Engine("piper")
    queue = SpeechQueue(edge, Player(), piper, retry_after=60, clock=lambda: now[0])
    failures: list[tuple[str, str]] = []
    queue.on_failure = lambda engine, problem: failures.append((engine, problem))
    queue.say("Раз. Два.")
    assert queue.wait_idle(timeout=5)
    assert edge.texts == ["Раз."] and piper.texts == ["Раз.", "Два."]  # «Два» — сразу запасным
    assert failures == [("edge", "нет сети")]  # о сбое — один раз, а не на каждую фразу
    now[0] += 61
    edge.fail = False
    queue.say("Три.")
    assert queue.wait_idle(timeout=5)
    assert edge.texts == ["Раз.", "Три."]
    queue.close()
    assert edge.closed and piper.closed  # процесс Silero тоже останавливается вместе с очередью


# --- Silero: процесс синтеза --------------------------------------------------------------

FAKE_WORKER = """
import json, struct, sys, time
from pathlib import Path

stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
setup = json.loads(stdin.readline())
log = Path(setup["model"]).with_suffix(".requests")
log.write_text(json.dumps(setup, ensure_ascii=False) + "\\n", encoding="utf-8")
if setup["model"].endswith("broken.pt"):
    stdout.write(json.dumps({"ok": False, "error": "RuntimeError: модель повреждена"}).encode() + b"\\n")
    stdout.flush()
    sys.exit(1)
stdout.write(json.dumps({"ok": True, "torch": "fake"}).encode() + b"\\n")
stdout.flush()
for line in stdin:
    request = json.loads(line)
    with log.open("a", encoding="utf-8") as file:
        file.write(json.dumps(request, ensure_ascii=False) + "\\n")
    if request["text"] == "упади":
        sys.exit(3)
    if request["text"] == "зависни":
        time.sleep(60)
    count = len(request["text"]) * 10
    stdout.write(json.dumps({"ok": True, "samples": count}).encode() + b"\\n")
    stdout.write(struct.pack(f"<{count}f", *([0.5] * count)))
    stdout.flush()
"""


@pytest.fixture
def worker(tmp_path: Path) -> list[str]:
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER, encoding="utf-8")
    return [sys.executable, str(script)]


def requests_of(model: Path) -> list[dict[str, Any]]:
    lines = model.with_suffix(".requests").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_silero_speaks_through_worker(worker: list[str], tmp_path: Path) -> None:
    model = tmp_path / "v5_6_ru.pt"
    model.write_bytes(b"model")
    lib = tmp_path / "lib"
    engine = SileroTTS(model, "eugene", rate=1.3, lib=lib, command=worker)
    audio, rate = engine.synthesize("Громкость 50%")
    assert rate == 48_000 and np.allclose(audio, 0.5)
    setup, request = requests_of(model)
    assert (
        setup == {"model": str(model), "lib": str(lib), "threads": setup["threads"]} and setup["threads"] >= 1
    )
    assert request == {
        "text": "Громкость пятьдесят процентов", "speaker": "eugene", "rate": "fast", "sample_rate": 48_000,
    }  # fmt: skip
    assert len(audio) == len(request["text"]) * 10
    # «Прослушать» — другим голосом через тот же процесс.
    engine.speak("Привет", "baya", 1.0)
    assert requests_of(model)[-1]["speaker"] == "baya" and len(requests_of(model)) == 3
    engine.close()


def test_silero_restarts_crashed_worker(worker: list[str], tmp_path: Path) -> None:
    model = tmp_path / "v5_6_ru.pt"
    model.write_bytes(b"model")
    engine = SileroTTS(model, "aidar", command=worker)
    with pytest.raises(TTSError, match=r"завершился \(код 3\)"):
        engine.synthesize("упади")
    audio, _ = engine.synthesize("Снова")
    assert len(audio) == len("Снова") * 10
    engine.close()


def test_silero_hung_worker_is_killed(
    worker: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "v5_6_ru.pt"
    model.write_bytes(b"model")
    monkeypatch.setattr(silero_tts, "REQUEST_TIMEOUT", 0.5)
    engine = SileroTTS(model, "aidar", command=worker)
    with pytest.raises(TTSError):
        engine.synthesize("зависни")
    assert len(engine.synthesize("Живой")[0]) == len("Живой") * 10
    engine.close()


def test_silero_reports_broken_model_and_missing_file(worker: list[str], tmp_path: Path) -> None:
    broken = tmp_path / "broken.pt"
    broken.write_bytes(b"x")
    with pytest.raises(TTSError, match="Silero не загрузился: RuntimeError: модель повреждена"):
        SileroTTS(broken, "aidar", command=worker).synthesize("Привет")
    with pytest.raises(TTSError, match="нет модели Silero missing.pt"):
        SileroTTS(tmp_path / "missing.pt", "aidar", command=worker).synthesize("Привет")


def test_silero_skips_text_without_words(worker: list[str], tmp_path: Path) -> None:
    engine = SileroTTS(tmp_path / "missing.pt", "aidar", command=worker)
    audio, rate = engine.synthesize("…")  # процесс даже не запускается
    assert len(audio) == 0 and rate == 48_000


def test_rate_word() -> None:
    assert [rate_word(rate) for rate in (0.5, 0.9, 1.0, 1.2, 2.0)] == [
        "x-slow",
        "slow",
        "medium",
        "fast",
        "x-fast",
    ]


# --- Silero: сам процесс (с подделкой torch) ---------------------------------------------


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def reshape(self, *shape: int) -> FakeTensor:
        return FakeTensor(self.values.reshape(*shape))

    def to(self, dtype: Any) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to(self, device: Any) -> FakeModel:
        return self

    def apply_tts(self, **kwargs: Any) -> FakeTensor:
        self.calls.append(kwargs)
        return FakeTensor(np.full(5, 0.25))


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"  # type: ignore[attr-defined]
    torch.float32 = "float32"  # type: ignore[attr-defined]
    torch.set_num_threads = lambda count: None  # type: ignore[attr-defined]
    torch.device = lambda name: name  # type: ignore[attr-defined]
    torch.cat = lambda parts: FakeTensor(np.concatenate([part.values for part in parts]))  # type: ignore[attr-defined]
    package = types.ModuleType("torch.package")

    class PackageImporter:
        def __init__(self, path: str) -> None:
            if not Path(path).exists():
                raise FileNotFoundError(path)

        def load_pickle(self, package_name: str, resource: str) -> FakeModel:
            assert (package_name, resource) == ("tts_models", "model")
            return model

    package.PackageImporter = PackageImporter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.package", package)
    return model


def run(lines: list[dict[str, Any]]) -> tuple[int, list[tuple[dict[str, Any], bytes]]]:
    stdin = io.BytesIO(b"".join(json.dumps(line, ensure_ascii=False).encode() + b"\n" for line in lines))
    stdout = io.BytesIO()
    code = run_worker(stdin, stdout)
    stdout.seek(0)
    replies = []
    while header_line := stdout.readline():
        header = json.loads(header_line)
        replies.append((header, stdout.read(header.get("samples", 0) * 4)))
    return code, replies


def test_worker_synthesizes_with_ssml_rate(fake_torch: FakeModel, tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    model.write_bytes(b"x")
    setup = {"model": str(model), "lib": str(tmp_path / "lib"), "threads": 2}
    plain = {"text": "Привет.", "speaker": "aidar", "rate": "medium", "sample_rate": 48_000}
    fast = {**plain, "rate": "fast"}
    code, replies = run([setup, plain, fast])
    assert code == 0
    assert replies[0][0] == {"ok": True, "torch": "0.0-fake"}
    assert [header for header, _ in replies[1:]] == [{"ok": True, "samples": 5}] * 2
    assert struct.unpack("<5f", replies[1][1]) == (0.25,) * 5
    assert fake_torch.calls == [
        {"text": "Привет.", "speaker": "aidar", "sample_rate": 48_000},
        {"ssml_text": '<speak><prosody rate="fast">Привет.</prosody></speak>', "speaker": "aidar",
         "sample_rate": 48_000},
    ]  # fmt: skip
    assert str(tmp_path / "lib") in sys.path  # torch ищется в скачанной папке
    sys.path.remove(str(tmp_path / "lib"))


def test_worker_reports_missing_model(fake_torch: FakeModel, tmp_path: Path) -> None:
    code, replies = run([{"model": str(tmp_path / "missing.pt"), "lib": "", "threads": 1}])
    assert code == 1 and replies[0][0]["ok"] is False and "FileNotFoundError" in replies[0][0]["error"]


def test_long_text_is_split_by_punctuation() -> None:
    text = "Первое предложение довольно длинное, с запятыми. " * 40
    parts = pieces(text)
    assert len(parts) > 1 and all(len(part) <= 800 for part in parts)
    assert " ".join(parts) == text.strip()
