"""Настройки из интерфейса: config.yaml с комментариями, ключи в .env, данные для окон."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from assistant.config import AppConfig
from assistant.llm.providers import CUSTOM_KEY_ENV
from assistant.ui.settings import (
    Choices,
    SettingsError,
    SettingsStore,
    check_key,
    key_hint,
    llm_profiles_view,
    needs_restart,
    piper_voices,
    settings_view,
)
from tests.conftest import ROOT


@pytest.fixture
def paths(tmp_path: Path) -> tuple[Path, Path]:
    config = tmp_path / "config.yaml"
    shutil.copyfile(ROOT / "config.example.yaml", config)
    return config, tmp_path / ".env"


@pytest.fixture
def environ() -> dict[str, str]:
    return {}


@pytest.fixture
def settings(paths: tuple[Path, Path], environ: dict[str, str]) -> SettingsStore:
    return SettingsStore(paths[0], paths[1], environ=environ)


def test_update_keeps_comments_and_other_values(settings: SettingsStore, paths: tuple[Path, Path]) -> None:
    config = settings.update(
        {"general": {"dry_run": True}, "voice": {"wake_confidence": 0.85}, "ui": {"theme": "light"}}
    )
    assert config.general.dry_run and config.voice.wake_confidence == 0.85 and config.ui.theme == "light"
    text = paths[0].read_text(encoding="utf-8")
    assert "# true — действия только пишутся в журнал" in text  # комментарии на месте
    assert "dry_run: true" in text and "wake_confidence: 0.85" in text
    assert "name_variants: [джарвиз, жарвис, jarvis]" in text  # остальное как было
    assert settings.load().ui.theme == "light"


def test_update_creates_missing_section(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("general:\n  name: пятница\n", encoding="utf-8")
    config = SettingsStore(path, tmp_path / ".env", environ={}).update({"ui": {"accent": "#22b8cf"}})
    assert config.ui.accent == "#22b8cf" and config.general.name == "пятница"
    assert "ui:\n  accent: '#22b8cf'" in path.read_text(encoding="utf-8")


def test_unknown_or_forbidden_fields_are_rejected(settings: SettingsStore, paths: tuple[Path, Path]) -> None:
    before = paths[0].read_text(encoding="utf-8")
    with pytest.raises(SettingsError, match="llm.profile"):
        settings.update({"llm": {"profile": "ollama"}})  # профиль — только через update_llm
    with pytest.raises(SettingsError, match="logging.dir"):
        settings.update({"logging": {"dir": "C:/"}})
    assert paths[0].read_text(encoding="utf-8") == before


def test_invalid_value_is_explained_and_not_saved(settings: SettingsStore, paths: tuple[Path, Path]) -> None:
    before = paths[0].read_text(encoding="utf-8")
    with pytest.raises(SettingsError) as info:
        settings.update({"ui": {"accent": "red"}})
    assert any("ui.accent" in error for error in info.value.errors)
    with pytest.raises(SettingsError):
        settings.update({"voice": {"wake_confidence": 7}})
    assert paths[0].read_text(encoding="utf-8") == before


def test_ui_values_are_converted(settings: SettingsStore) -> None:
    config = settings.update({"voice": {"input_device": ""}, "general": {"name": " Пятница "}})
    assert config.voice.input_device is None and config.general.name == "пятница"
    assert settings.update({"voice": {"input_device": "USB Audio"}}).voice.input_device == "USB Audio"


def test_update_llm_selects_profile(settings: SettingsStore, paths: tuple[Path, Path]) -> None:
    config = settings.update_llm("openai", {"model": "gpt-5.4-mini"})
    assert config.llm.profile == "openai" and config.llm.active.model == "gpt-5.4-mini"
    assert config.llm.active.effort == "none"  # остальное — из встроенного профиля
    custom = {"title": "Свой", "base_url": "http://localhost:1234/v1", "model": "m", "api_key_env": ""}
    config = settings.update_llm("custom", custom)
    assert config.llm.profile == "custom" and config.llm.active.base_url == "http://localhost:1234/v1"
    with pytest.raises(SettingsError):
        settings.update_llm("custom2", {"model": "m"})  # у openai-профиля без адреса не бывает
    assert settings.load().llm.profile == "custom"


def test_keys_are_written_to_env_and_hidden(
    settings: SettingsStore, paths: tuple[Path, Path], environ: dict[str, str]
) -> None:
    added: list[str] = []
    settings._on_key = added.append
    settings.save_key("OPENAI_API_KEY", "  sk-test-openai-1234567890  ")
    assert environ["OPENAI_API_KEY"] == "sk-test-openai-1234567890" and added == ["sk-test-openai-1234567890"]
    assert "OPENAI_API_KEY='sk-test-openai-1234567890'" in paths[1].read_text(encoding="utf-8")
    settings.save_key("OPENAI_API_KEY", "sk-test-openai-new-key-99")
    assert paths[1].read_text(encoding="utf-8").count("OPENAI_API_KEY") == 1
    settings.forget_key("OPENAI_API_KEY")
    assert "OPENAI_API_KEY" not in environ and "OPENAI_API_KEY" not in paths[1].read_text(encoding="utf-8")


@pytest.mark.parametrize("bad", ["", "short", "sk test key 123", "sk-ключ-1234567890", "sk-1234567890\nX=1"])
def test_bad_keys_are_rejected(bad: str) -> None:
    with pytest.raises(SettingsError):
        check_key(bad)


def test_key_check_names_the_problem() -> None:
    with pytest.raises(SettingsError, match="сокращённая запись ключа"):
        check_key("sk-ant-api03-R2D...7QAA")
    with pytest.raises(SettingsError, match="символ «к» \\(4-й\\)"):
        check_key("sk-ключ-1234567890")
    # Невидимые символы из браузера или мессенджера убираются молча.
    assert check_key("\u200bsk-test-openai-1234567890\ufeff") == "sk-test-openai-1234567890"


def test_key_from_windows_is_marked(settings: SettingsStore, environ: dict[str, str]) -> None:
    environ["ANTHROPIC_API_KEY"] = "sk-ant-api03-from-windows-1234"
    claude = next(v for v in llm_profiles_view(settings.load(), settings) if v["name"] == "anthropic")
    assert claude["key_set"] and claude["key_from_system"] and claude["key_example"] == "sk-ant-api03-…"
    settings.save_key("ANTHROPIC_API_KEY", "sk-ant-api03-saved-key-5678")
    claude = next(v for v in llm_profiles_view(settings.load(), settings) if v["name"] == "anthropic")
    assert claude["key_hint"] == "••••5678" and not claude["key_from_system"]


def test_env_variable_name_is_checked(settings: SettingsStore) -> None:
    with pytest.raises(SettingsError):
        settings.save_key("PATH\nEVIL", "sk-test-openai-1234567890")


def test_key_hint_shows_only_tail() -> None:
    assert key_hint("sk-test-openai-1234567890") == "••••7890"
    assert key_hint("") == "" and key_hint("abc") == "••••"


def test_restart_needed_only_for_voice_stt_tts_and_name() -> None:
    assert not needs_restart(
        {"general": {"dry_run": True}, "ui": {"theme": "light"}, "llm": {"enabled": False}}
    )
    assert not needs_restart({"tts": {"volume": 1.5}})
    assert needs_restart({"voice": {"sounds": False}})
    assert needs_restart({"general": {"name": "пятница"}})
    assert needs_restart({"tts": {"rate": 1.2}})


def test_profiles_view_order_keys_and_custom(settings: SettingsStore, environ: dict[str, str]) -> None:
    environ["DEEPSEEK_API_KEY"] = "sk-deepseek-1234567890abcd"
    views = llm_profiles_view(settings.load(), settings)
    assert [v["name"] for v in views] == [
        "deepseek",
        "openai",
        "anthropic",
        "ollama",
        "opencode_zen",
        "custom",
    ]
    deepseek, openai, ollama, custom = views[0], views[1], views[3], views[-1]
    assert deepseek["key_set"] and deepseek["key_hint"] == "••••abcd"
    assert "sk-deepseek" not in str(views)  # ключ в окно не уходит
    assert (
        not openai["key_set"]
        and openai["models"][0] == "gpt-6-luna"
        and openai["key_url"].startswith("https://")
    )
    assert ollama["key_env"] == "" and ollama["base_url_editable"]
    assert custom == {**custom, "configured": False, "key_env": CUSTOM_KEY_ENV, "key_optional": True}


def test_settings_view_matches_interface(settings: SettingsStore) -> None:
    config = settings.load()
    view = settings_view(config, settings, Choices(microphones=[(1, "Микрофон (Realtek)"), (3, "USB Audio")]))
    # Ключи — как в типе Settings в ui/src/lib/types.ts.
    assert set(view) == {"general", "voice", "stt", "tts", "llm", "ui", "choices"}
    assert set(view["voice"]) == {
        "enabled", "wake_word", "push_to_talk", "push_to_talk_key", "push_to_talk_mode", "stop_key",
        "input_device", "wake_confidence", "followup_seconds", "min_silence_ms", "sounds",
    }  # fmt: skip
    assert set(view["tts"]) == {"engine", "piper_voice", "sapi_voice", "rate", "volume"}
    assert set(view["ui"]) == {"theme", "accent", "overlay", "quick_chat_key"}
    assert set(view["choices"]) == {"microphones", "piper_voices", "sapi_voices", "stt_models"}
    assert view["voice"]["input_device"] == ""
    assert [c["value"] for c in view["choices"]["microphones"]] == ["", "Микрофон (Realtek)", "USB Audio"]
    assert view["tts"]["piper_voice"] == "models/piper/ru_RU-dmitri-medium.onnx"


def test_microphone_number_from_config_is_shown_by_name(settings: SettingsStore) -> None:
    config = settings.load()
    config.voice.input_device = 3  # в config.yaml можно указать и номер устройства
    view = settings_view(config, settings, Choices(microphones=[(3, "USB Audio")]))
    assert view["voice"]["input_device"] == "USB Audio"


def test_piper_voices_are_found_next_to_current(tmp_path: Path) -> None:
    folder = tmp_path / "models" / "piper"
    folder.mkdir(parents=True)
    for name in ("ru_RU-dmitri-medium", "ru_RU-irina-medium", "en_US-amy-low"):
        (folder / f"{name}.onnx").write_bytes(b"")
    config = AppConfig()
    config._base_dir = tmp_path
    voices = dict(piper_voices(config))
    assert voices["models/piper/ru_RU-irina-medium.onnx"] == "Ирина"
    assert voices["models/piper/ru_RU-dmitri-medium.onnx"] == "Дмитрий"
    assert voices["models/piper/en_US-amy-low.onnx"] == "Amy"
