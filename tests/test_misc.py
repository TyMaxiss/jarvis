"""Тесты конфига, маскировки секретов, разбора клавиш, адресов и времени."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from assistant.actions.browser import normalize_url
from assistant.actions.input import parse_hotkey, parse_key
from assistant.actions.registry import ActionError
from assistant.actions.system import format_now
from assistant.config import OLD_EXAMPLE_LINES, AppConfig, ConfigError, load_config
from assistant.hotkeys import HotkeyManager
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

    def test_overlay_keep_from_old_example_is_turned_off(self, tmp_path: Path) -> None:
        # В 0.8.0 пример включал overlay_keep, и оверлей «висел» после команды.
        comment = "# итог голосовой команды виден до следующей фразы (клик — спрятать)"
        old = f"  overlay_keep: true{' ' * 13}{comment}"
        path = tmp_path / "config.yaml"
        path.write_bytes(f"ui:\r\n  overlay: true\r\n{old}\r\n".encode())
        assert load_config(path).ui.overlay_keep is False
        text = path.read_bytes().decode()
        assert old not in text and "  overlay_keep: false" in text
        assert text.startswith("ui:\r\n  overlay: true\r\n") and text.endswith("\r\n")  # остальное как было
        example = (ROOT / "config.example.yaml").read_text(encoding="utf-8").splitlines()
        assert [line for line in example if "overlay_keep" in line] == [*OLD_EXAMPLE_LINES.values()]

    def test_own_overlay_keep_stays(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("ui:\n  overlay_keep: true\n", encoding="utf-8")  # включено в настройках
        assert load_config(path).ui.overlay_keep is True
        assert path.read_text(encoding="utf-8") == "ui:\n  overlay_keep: true\n"

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

    def test_env_file_wins_over_windows_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Ключ, сохранённый в настройках, работает, даже если в Windows задан старый.
        monkeypatch.setenv("JARVIS_TEST_KEY", "old-key-from-windows")
        (tmp_path / ".env").write_text("JARVIS_TEST_KEY='new-key-from-settings'\n", encoding="utf-8")
        with caplog.at_level("INFO", logger="assistant.config"):
            load_config(tmp_path / "config.yaml")
        import os

        assert os.environ["JARVIS_TEST_KEY"] == "new-key-from-settings"
        assert "JARVIS_TEST_KEY берётся из .env" in caplog.text and "new-key" not in caplog.text


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


class FakeKeyboard:
    """Вместо библиотеки keyboard: сочетание «нажимается» вызовом press."""

    KEY_DOWN = "down"

    def __init__(self) -> None:
        self.hotkeys: dict[str, Any] = {}
        self.hooks: list[Any] = []

    def parse_hotkey(self, combo: str) -> tuple[tuple[tuple[int, ...], ...]]:
        return (tuple((index,) for index, _ in enumerate(combo.split("+"))),)

    def add_hotkey(self, combo: str, callback: Any, suppress: bool = False) -> Any:
        self.hotkeys[combo] = callback
        return lambda: self.hotkeys.pop(combo)

    def press(self, combo: str) -> None:
        self.hotkeys[combo]()


class TestHotkeyManager:
    def test_paused_hotkeys_do_not_fire(self) -> None:
        paused = threading.Event()
        manager = HotkeyManager(paused)
        manager._keyboard = FakeKeyboard()
        fired: list[str] = []
        manager.add_press("ctrl+alt+f9", lambda: fired.append("f9"))
        paused.set()  # в настройках записывают новое сочетание
        manager._keyboard.press("ctrl+alt+f9")
        assert fired == []
        paused.clear()
        manager._keyboard.press("ctrl+alt+f9")
        assert fired == ["f9"]
        manager.close()
        assert manager._keyboard.hotkeys == {}


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
