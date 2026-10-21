"""Общие фикстуры: настоящие действия, YAML-паки и матчер без обращения к Windows."""

from __future__ import annotations

from pathlib import Path

import pytest

from assistant.actions.apps import AppEntry, AppIndex
from assistant.actions.registry import ActionRegistry
from assistant.actions.winapi import EMPTY_WINDOW, WindowInfo
from assistant.commands.loader import CommandStore
from assistant.config import AppConfig
from assistant.main import build_registry
from assistant.nlu.matcher import Matcher
from assistant.nlu.normalize import Normalizer

ROOT = Path(__file__).resolve().parent.parent

YOUTUBE_WINDOW = WindowInfo(1, "Котики - YouTube - Google Chrome", "Chrome_WidgetWin_1", 100, "chrome.exe")
NOTEPAD_WINDOW = WindowInfo(2, "Без имени — Блокнот", "Notepad", 200, "notepad.exe")
NO_WINDOW = EMPTY_WINDOW

FAKE_APPS = [
    AppEntry("Telegram", "shortcut", r"C:\Start Menu\Telegram.lnk"),
    AppEntry("Discord", "shortcut", r"C:\Start Menu\Discord.lnk"),
    AppEntry("Google Chrome", "shortcut", r"C:\Start Menu\Google Chrome.lnk"),
    AppEntry("Visual Studio Code", "shortcut", r"C:\Start Menu\Visual Studio Code.lnk"),
    AppEntry("Калькулятор", "start_app", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
    AppEntry("Uninstall Telegram", "shortcut", r"C:\Start Menu\Uninstall Telegram.lnk"),
]


class RecordingSpeaker:
    """Запоминает, что ассистент «сказал»."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig()


@pytest.fixture
def speaker() -> RecordingSpeaker:
    return RecordingSpeaker()


@pytest.fixture
def app_index(config: AppConfig) -> AppIndex:
    index = AppIndex(config.apps, scanner=lambda: list(FAKE_APPS))
    index.refresh()
    return index


@pytest.fixture
def registry(config: AppConfig, app_index: AppIndex, speaker: RecordingSpeaker) -> ActionRegistry:
    return build_registry(config, app_index, speaker)


@pytest.fixture
def store(registry: ActionRegistry) -> CommandStore:
    store = CommandStore(ROOT / "commands", "my.yaml", registry)
    store.reload()
    return store


@pytest.fixture
def normalizer(config: AppConfig) -> Normalizer:
    return Normalizer(config.wake_words, config.matcher.filler_words)


@pytest.fixture
def matcher(normalizer: Normalizer, config: AppConfig, store: CommandStore) -> Matcher:
    matcher = Matcher(normalizer, config.matcher)
    matcher.set_commands(store.commands)
    return matcher
