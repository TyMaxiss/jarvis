"""Тесты конфига, маскировки секретов, разбора клавиш, адресов и времени."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from assistant.actions.browser import normalize_url
from assistant.actions.input import parse_hotkey, parse_key
from assistant.actions.registry import ActionError
from assistant.actions.system import format_now
from assistant.config import AppConfig, ConfigError, load_config
from assistant.logging_setup import MaskingFormatter, SecretMasker
from tests.conftest import ROOT


class TestConfig:
    def test_created_from_example(self, tmp_path: Path) -> None:
        (tmp_path / "config.example.yaml").write_text(
            (ROOT / "config.example.yaml").read_text(encoding="utf-8"), encoding="utf-8"
        )
        config = load_config(tmp_path / "config.yaml")
        assert (tmp_path / "config.yaml").exists()
        assert config.general.name == "джарвис"
        assert config.matcher.threshold == 85
        assert config.apps.aliases["телега"] == "Telegram"
        assert config.resolve("commands") == tmp_path / "commands"

    def test_example_matches_defaults(self, tmp_path: Path) -> None:
        # Пример конфига не должен расходиться со значениями по умолчанию в коде.
        example = load_config(ROOT / "config.example.yaml")
        defaults = AppConfig()
        for section in ("general", "matcher", "commands", "browser", "voice", "stt", "tts", "llm", "logging"):
            assert getattr(example, section) == getattr(defaults, section), section

    def test_defaults_without_files(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "config.yaml")
        assert config.model_dump() == AppConfig().model_dump()
        assert config.base_dir == tmp_path

    def test_typo_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("matcher:\n  treshold: 90\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="matcher.treshold: неизвестный параметр"):
            load_config(tmp_path / "config.yaml")

    def test_bad_value_is_reported(self, tmp_path: Path) -> None:
        (tmp_path / "config.yaml").write_text("matcher:\n  threshold: 150\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="matcher.threshold"):
            load_config(tmp_path / "config.yaml")

    def test_env_file_is_loaded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("JARVIS_TEST_KEY", raising=False)
        (tmp_path / ".env").write_text("JARVIS_TEST_KEY=abc123456\n", encoding="utf-8")
        load_config(tmp_path / "config.yaml")
        import os

        assert os.environ["JARVIS_TEST_KEY"] == "abc123456"
        monkeypatch.delenv("JARVIS_TEST_KEY")


class TestSecrets:
    def test_masks_env_secrets_and_key_like_strings(self) -> None:
        masker = SecretMasker.from_environ(
            {"DEEPSEEK_API_KEY": "abcdef123456", "PATH": "C:\\Windows", "SHORT_TOKEN": "abc"}
        )
        text = masker.mask("ключ abcdef123456, другой sk-1234567890abcdefXYZ, путь C:\\Windows, abc")
        assert "abcdef123456" not in text and "sk-1234567890" not in text
        assert "C:\\Windows" in text and text.endswith("abc")

    def test_formatter_masks_tracebacks(self) -> None:
        formatter = MaskingFormatter("%(message)s", SecretMasker(["supersecret"]))
        try:
            raise ValueError("bad key supersecret")
        except ValueError:
            record = logging.LogRecord("x", logging.ERROR, __file__, 1, "ошибка %s", ("supersecret",),
                                       exc_info=__import__("sys").exc_info())  # fmt: skip
        output = formatter.format(record)
        assert "supersecret" not in output
        assert output.startswith("ошибка ***") and output.endswith("ValueError: bad key ***")


class TestKeys:
    def test_hotkeys(self) -> None:
        assert parse_hotkey("win+d") == [0x5B, ord("D")]
        assert parse_hotkey("Ctrl + Shift + Esc") == [0x11, 0x10, 0x1B]
        assert parse_hotkey("alt+f4") == [0x12, 0x73]
        assert parse_hotkey("ctrl+alt+пробел") == [0x11, 0x12, 0x20]
        assert parse_hotkey("ctrl+plus") == [0x11, 0xBB]

    def test_cyrillic_letters_map_to_physical_keys(self) -> None:
        assert parse_key("с") == ord("C")  # кириллическая «с» на клавише C
        assert parse_key("ф") == ord("A")
        assert parse_key("k") == ord("K")

    @pytest.mark.parametrize("bad", ["", "ctrl+", "+a", "ctrl++", "ctrl+banana", "f25"])
    def test_unknown_keys(self, bad: str) -> None:
        with pytest.raises(ActionError):
            parse_hotkey(bad)


class TestUrls:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("youtube.com", "https://youtube.com"),
            ("vk.com/feed", "https://vk.com/feed"),
            ("https://www.google.com/search?q=x", "https://www.google.com/search?q=x"),
            ("http://example.org", "http://example.org"),
        ],
    )
    def test_valid(self, url: str, expected: str) -> None:
        assert normalize_url(url) == expected

    @pytest.mark.parametrize(
        "url", ["file:///C:/Windows/system32/calc.exe", "javascript:alert(1)", "просто текст"]
    )
    def test_rejected(self, url: str) -> None:
        with pytest.raises(ActionError):
            normalize_url(url)


def test_time_in_russian() -> None:
    assert format_now(datetime(2026, 10, 21, 9, 5)) == "Сейчас 09:05, среда, 21 октября"
