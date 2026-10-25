"""Тесты сборки в main.py: маскировка ключей, сообщения при старте, скачивание моделей при запуске."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pytest

from assistant import main
from assistant.audio.wakeword import VoskModelError
from assistant.config import AppConfig
from assistant.frozen import cuda_dir
from assistant.instance import RESTART_ENV
from assistant.logging_setup import SecretMasker
from assistant.main import build_masker, describe_llm
from assistant.tts.silero_tts import SileroTTS
from tests.test_updates import path_cleaned_to


def test_keys_of_all_profiles_are_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    # Переменная не похожа на ключ по имени (*_KEY, *_TOKEN), но указана в профиле.
    monkeypatch.setenv("MY_LLM_CREDENTIAL", "very-secret-value-2026")
    config = AppConfig.model_validate(
        {"llm": {"profiles": {"my": {"base_url": "https://llm.example/v1", "model": "m",
                                     "api_key_env": "MY_LLM_CREDENTIAL"}}}}
    )  # fmt: skip
    line = "ответ: very-secret-value-2026"
    assert SecretMasker.from_environ().mask(line) == line  # по имени переменной не узнать
    assert "very-secret" not in build_masker(config).mask(line)


def test_start_message_explains_missing_key_and_zen_free_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCODE_API_KEY", raising=False)
    text = describe_llm(AppConfig.model_validate({"llm": {"profile": "opencode_zen"}}))
    assert "профиль opencode_zen, модель deepseek-v4-flash-free" in text
    assert "укажите его в настройках (или OPENCODE_API_KEY в .env)" in text
    assert "403 FreeTierError" in text


def test_start_message_for_ready_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-deepseek-key-123456")
    text = describe_llm(AppConfig.model_validate({"llm": {"profile": "deepseek"}}))
    assert text == "Нейросеть: профиль deepseek, модель deepseek-flash.\n"
    disabled = describe_llm(AppConfig.model_validate({"llm": {"enabled": False}}))
    assert disabled.startswith("Нейросеть выключена")


# --- Первый запуск: модели скачиваются сами ----------------------------------------------


class FakeSpeech:
    """Очередь озвучки: запоминает, на какой голос её переключили."""

    def __init__(self) -> None:
        self.engines: list[Any] = []
        self.engine: Any = FakeTts()

    def switch(self, engine: Any) -> None:
        self.engines.append(engine)
        self.engine = engine


class FakeTts:
    def __init__(self, *args: Any) -> None:
        self.name = "sapi"


class FakePlayer:
    def reset(self) -> None:
        pass

    def stop(self) -> None:
        pass


def piper_config(tmp_path: Path, engine: str = "piper") -> AppConfig:
    voice = tmp_path / "piper" / "ru_RU-dmitri-medium.onnx"
    return AppConfig.model_validate({"tts": {"engine": engine, "piper_voice": str(voice)}})


def test_piper_voice_is_downloaded_and_replaces_windows_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = piper_config(tmp_path)
    voice = config.resolve(config.tts.piper_voice)

    def download(path: Path, out: Any) -> None:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"onnx")
        Path(f"{path}.json").write_text("{}")

    monkeypatch.setattr("assistant.models.ensure_piper", download)
    monkeypatch.setattr("assistant.tts.piper_tts.PiperTTS", lambda path, rate: ("piper", path, rate))
    speech = FakeSpeech()
    main.fetch_voice(config, speech, io.StringIO())  # type: ignore[arg-type]
    assert speech.engines == [("piper", voice, 1.0)]
    main.fetch_voice(config, speech, io.StringIO())  # type: ignore[arg-type]
    assert len(speech.engines) == 1  # уже скачан


def test_piper_download_failure_keeps_windows_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(path: Path, out: Any) -> None:
        raise OSError("нет сети")

    monkeypatch.setattr("assistant.models.ensure_piper", offline)
    speech, out = FakeSpeech(), io.StringIO()
    main.fetch_voice(piper_config(tmp_path), speech, out)  # type: ignore[arg-type]
    assert speech.engines == []
    assert out.getvalue().startswith(
        "! Не удалось скачать голос Piper (нет сети). Пока говорю запасным голосом"
    )
    main.fetch_voice(piper_config(tmp_path, engine="sapi"), speech, out)  # type: ignore[arg-type]
    assert out.getvalue().count("!") == 1  # голос Windows выбран сам — Piper не нужен


def test_speech_starts_with_windows_voice_until_piper_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr("assistant.tts.sapi_tts.SapiTTS", FakeTts)
    monkeypatch.setattr("assistant.tts.player.SoundDevicePlayer", FakePlayer)
    out = io.StringIO()
    speech = main.create_speech(piper_config(tmp_path), out)
    assert speech is not None and speech.engine.name == "sapi"
    assert out.getvalue() == "Голос Piper скачаю при запуске голоса, а пока говорю голосом Windows (SAPI).\n"
    speech.close()


def test_wake_model_is_downloaded_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "vosk-model-small-ru-0.22"
    monkeypatch.setattr("assistant.models.ensure_vosk", lambda folder, out: folder.mkdir())
    monkeypatch.setattr("assistant.audio.wakeword.load_vosk_model", lambda folder: ("vosk", folder))
    assert main.load_wake_model(path, io.StringIO()) == ("vosk", path)


def test_wake_model_problems_do_not_stop_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(folder: Path, out: Any) -> None:
        raise OSError("нет сети")

    monkeypatch.setattr("assistant.models.ensure_vosk", offline)
    out = io.StringIO()
    assert main.load_wake_model(tmp_path / "vosk-model", out) is None
    assert out.getvalue().startswith("! Не удалось скачать модель Vosk (нет сети): пока не работают")

    def broken(folder: Path) -> Any:
        raise VoskModelError("Не удалось загрузить модель Vosk")

    monkeypatch.setattr("assistant.audio.wakeword.load_vosk_model", broken)
    out = io.StringIO()
    assert main.load_wake_model(tmp_path, out) is None  # папка есть — не скачивается
    assert out.getvalue() == "! Не удалось загрузить модель Vosk\n"


def test_only_exe_downloads_cuda_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    assert main.cuda_fetcher(io.StringIO()) is None  # из исходников — uv sync --extra cuda
    monkeypatch.setattr(main, "FROZEN", True)
    targets: list[Path] = []
    monkeypatch.setattr("assistant.models.ensure_cuda", lambda target, out: targets.append(target))
    fetch = main.cuda_fetcher(io.StringIO())
    assert fetch is not None
    fetch()
    assert targets == [cuda_dir()]


def tts_config(tmp_path: Path, engine: str, **tts: Any) -> AppConfig:
    voice = tmp_path / "piper" / "ru_RU-dmitri-medium.onnx"
    model = tmp_path / "silero" / "v5_6_ru.pt"
    return AppConfig.model_validate(
        {"tts": {"engine": engine, "piper_voice": str(voice), "silero_model": str(model), **tts}}
    )


@pytest.fixture
def windows_voices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Как в Windows: голоса SAPI есть (подделка), проигрыватель — подделка."""
    monkeypatch.setattr(main.sys, "platform", "win32")
    monkeypatch.setattr("assistant.tts.sapi_tts.SapiTTS", FakeTts)
    monkeypatch.setattr("assistant.tts.player.SoundDevicePlayer", FakePlayer)


def test_edge_voice_with_windows_voice_as_fallback(tmp_path: Path, windows_voices: None) -> None:
    config = tts_config(tmp_path, "edge", edge_voice="ru-RU-SvetlanaNeural")
    speech = main.create_speech(config, io.StringIO())
    assert speech is not None
    assert (speech.engine.name, speech.engine.voice) == ("edge", "ru-RU-SvetlanaNeural")  # type: ignore[attr-defined]
    assert speech._fallback is not None and speech._fallback.name == "sapi"
    speech.close()


@pytest.mark.parametrize(("engine", "voice"), [("openai", "cedar"), ("yandex", "kirill")])
def test_keyed_voice_with_windows_voice_as_fallback(
    tmp_path: Path, windows_voices: None, engine: str, voice: str
) -> None:
    # Голос по ключу создаётся без сети и без ключа: без них фразу скажет запасной голос.
    speech = main.create_speech(tts_config(tmp_path, engine), io.StringIO())
    assert speech is not None
    assert (speech.engine.name, speech.engine.voice) == (engine, voice)  # type: ignore[attr-defined]
    assert speech._fallback is not None and speech._fallback.name == "sapi"
    speech.close()


def test_silero_is_downloaded_then_replaces_fallback_voice(
    tmp_path: Path, windows_voices: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tts_config(tmp_path, "silero")
    out = io.StringIO()
    speech = main.create_speech(config, out)
    assert speech is not None and speech.engine.name == "sapi"
    assert "Голос Silero скачаю при запуске голоса" in out.getvalue()

    downloads: list[Path] = []
    silero = SileroTTS(config.resolve(config.tts.silero_model), "aidar", command=["unused"])
    started: list[bool] = []
    monkeypatch.setattr(silero, "start", lambda: started.append(True))
    monkeypatch.setattr("assistant.models.ensure_silero", lambda model, out: downloads.append(model))
    monkeypatch.setattr("assistant.tts.silero_tts.silero_engine", lambda config: silero)
    main.fetch_voice(config, speech, out)
    assert downloads == [config.resolve(config.tts.silero_model)] and started == [True]
    assert speech.engine is silero and speech._fallback is not None and speech._fallback.name == "sapi"
    main.fetch_voice(config, speech, out)  # уже Silero — ничего не скачивается
    assert len(downloads) == 1
    speech.close()


def test_ready_silero_starts_with_assistant(
    tmp_path: Path, windows_voices: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tts_config(tmp_path, "silero")
    model = config.resolve(config.tts.silero_model)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    silero = SileroTTS(model, "aidar", command=["unused"])
    started: list[bool] = []
    monkeypatch.setattr(silero, "start", lambda: started.append(True))
    monkeypatch.setattr("assistant.models.silero_runtime", lambda: (True, None))
    monkeypatch.setattr("assistant.tts.silero_tts.silero_engine", lambda config: silero)
    speech = main.create_speech(config, io.StringIO())
    assert speech is not None and speech.engine is silero and started == [True]
    speech.close()


def test_voice_check_reports_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    class Engine:
        def __init__(self, name: str, seconds: float) -> None:
            self.name, self.seconds = name, seconds

        def synthesize(self, text: str) -> tuple[Any, int]:
            return np.zeros(int(self.seconds * 100)), 100

    class Speech:
        def __init__(self, name: str, seconds: float) -> None:
            self.engine = Engine(name, seconds)

        def close(self) -> None:
            pass

    monkeypatch.setattr(main, "create_speech", lambda config, out: Speech("edge", 2.5))
    out = io.StringIO()
    assert main.test_voice(AppConfig(), "edge", out) == 0
    assert out.getvalue().startswith("edge: 2.5 с речи за ")
    # Выбранный голос не заработал — говорил запасной: проверка не пройдена.
    monkeypatch.setattr(main, "create_speech", lambda config, out: Speech("sapi", 2.5))
    assert main.test_voice(AppConfig(), "edge", io.StringIO()) == 1


def test_restart_passes_path_without_program_folders(monkeypatch: pytest.MonkeyPatch) -> None:
    # Новый экземпляр после «Перезапустить» — как запущенный заново: без папок CUDA, Vosk и
    # WebView2, которые этот процесс добавил себе в PATH (иначе PATH растёт с каждым перезапуском).
    started: list[dict[str, str]] = []
    monkeypatch.setattr(main.subprocess, "Popen", lambda command, env: started.append(env))
    monkeypatch.setattr(main, "external_launch", lambda: path_cleaned_to("C:\\Windows"))
    monkeypatch.setenv("PATH", os.pathsep.join(["C:\\Jarvis\\cuda", "C:\\Windows"]))
    main.relaunch(["--debug"])
    assert started[0]["PATH"] == "C:\\Windows" and started[0][RESTART_ENV] == "1"
    assert os.environ["PATH"] == os.pathsep.join(["C:\\Jarvis\\cuda", "C:\\Windows"])


class Closing:
    """Часть голоса, которую закрывают при выходе: запоминает, закрыли ли её."""

    def __init__(self, calls: list[str], name: str, alive: bool = False) -> None:
        self._calls, self._name, self.alive = calls, name, alive

    def close(self) -> None:
        self._calls.append(self._name)

    def stop(self) -> None:
        self._calls.append(self._name)


@pytest.mark.parametrize(("alive", "unloaded"), [(False, True), (True, False)])
def test_voice_runtime_unloads_whisper_only_after_loop_stopped(alive: bool, unloaded: bool) -> None:
    # Выгружать модель, пока цикл ещё распознаёт, нельзя: CTranslate2 уронил бы программу.
    calls: list[str] = []
    runtime = main.VoiceRuntime(
        Closing(calls, "capture"), Closing(calls, "loop", alive), Closing(calls, "hotkeys"),  # type: ignore[arg-type]
        stt=Closing(calls, "stt"),  # type: ignore[arg-type]
    )  # fmt: skip
    runtime.close()
    assert calls == ["loop", "hotkeys", "capture", *(["stt"] if unloaded else [])]
    main.VoiceRuntime(Closing(calls, "capture"), Closing(calls, "loop"), Closing(calls, "hotkeys")).close()  # type: ignore[arg-type]
