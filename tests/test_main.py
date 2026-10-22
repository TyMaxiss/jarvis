"""Тесты сборки в main.py: маскировка ключей, сообщения при старте, скачивание моделей при запуске."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from assistant import main
from assistant.audio.wakeword import VoskModelError
from assistant.config import AppConfig
from assistant.frozen import cuda_dir
from assistant.logging_setup import SecretMasker
from assistant.main import build_masker, describe_llm


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

    def switch(self, engine: Any) -> None:
        self.engines.append(engine)


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
    main.fetch_piper_voice(config, speech, io.StringIO())  # type: ignore[arg-type]
    assert speech.engines == [("piper", voice, 1.0)]
    main.fetch_piper_voice(config, speech, io.StringIO())  # type: ignore[arg-type]
    assert len(speech.engines) == 1  # уже скачан


def test_piper_download_failure_keeps_windows_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(path: Path, out: Any) -> None:
        raise OSError("нет сети")

    monkeypatch.setattr("assistant.models.ensure_piper", offline)
    speech, out = FakeSpeech(), io.StringIO()
    main.fetch_piper_voice(piper_config(tmp_path), speech, out)  # type: ignore[arg-type]
    assert speech.engines == []
    assert out.getvalue().startswith(
        "! Не удалось скачать голос Piper (нет сети). Пока говорю голосом Windows"
    )
    main.fetch_piper_voice(piper_config(tmp_path, engine="sapi"), speech, out)  # type: ignore[arg-type]
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
