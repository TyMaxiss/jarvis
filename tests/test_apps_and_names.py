"""Тесты поиска приложений по разговорному названию."""

from __future__ import annotations

import pytest

from assistant.actions.apps import AppEntry, AppIndex, filter_entries
from assistant.config import AppsConfig
from assistant.nlu.names import best_matches, name_score, skeleton

NAMES = [
    "Telegram", "Telegram Desktop", "Discord", "Steam", "Steamworks Common Redistributables",
    "Google Chrome", "Visual Studio Code", "Spotify", "Yandex", "Яндекс Браузер", "Калькулятор",
    "OBS Studio", "Adobe Photoshop 2025", "Word", "Excel", "Zoom", "Skype", "Viber", "WhatsApp",
    "YouTube Music", "Notepad++", "Windows PowerShell", "Командная строка", "Диспетчер задач",
]  # fmt: skip


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("телеграм", "Telegram"),
        ("дискорд", "Discord"),
        ("стим", "Steam"),
        ("хром", "Google Chrome"),
        ("гугл хром", "Google Chrome"),
        ("вижуал студио код", "Visual Studio Code"),
        ("спотифай", "Spotify"),
        ("яндекс", "Yandex"),
        ("яндекс браузер", "Яндекс Браузер"),
        ("калькулятор", "Калькулятор"),
        ("обс", "OBS Studio"),
        ("фотошоп", "Adobe Photoshop 2025"),
        ("ворд", "Word"),
        ("эксель", "Excel"),
        ("зум", "Zoom"),
        ("скайп", "Skype"),
        ("вайбер", "Viber"),
        ("ватсап", "WhatsApp"),
        ("ютуб музыка", "YouTube Music"),
        ("повершел", "Windows PowerShell"),
        ("диспетчер задач", "Диспетчер задач"),
        ("Telegram", "Telegram"),
    ],
)
def test_spoken_names_find_apps(query: str, expected: str) -> None:
    found = best_matches(query, NAMES, threshold=80, limit=1)
    assert found and found[0][0] == expected


def test_unrelated_query_finds_nothing() -> None:
    assert best_matches("фотосинтез", ["Telegram", "Discord", "Steam"], threshold=80) == []


def test_skeleton_unifies_scripts() -> None:
    assert skeleton("Ютуб") == skeleton("YouTube")
    assert skeleton("дискорд") == skeleton("Discord")
    assert name_score("Steam", "Steam") == 100


@pytest.fixture
def index() -> AppIndex:
    entries = [
        AppEntry("Telegram", "shortcut", r"C:\Menu\Telegram.lnk"),
        AppEntry("Uninstall Telegram", "shortcut", r"C:\Menu\Uninstall Telegram.lnk"),
        AppEntry("Telegram", "start_app", "TelegramDesktop!App"),
        AppEntry("Калькулятор", "start_app", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
        AppEntry("Google Chrome", "shortcut", r"C:\Menu\Google Chrome.lnk"),
    ]
    config = AppsConfig(aliases={"телега": "Telegram", "блокнот": "notepad.exe", "стим": "Steam"})
    index = AppIndex(config, scanner=lambda: entries)
    index.refresh()
    return index


def test_index_filters_uninstallers_and_duplicates(index: AppIndex) -> None:
    names = [e.name for e in index.entries()]
    assert names == ["Telegram", "Калькулятор", "Google Chrome"]
    assert index.entries()[0].kind == "shortcut"  # первое вхождение побеждает


def test_alias_with_word_form(index: AppIndex) -> None:
    entry = index.find("телегу")
    assert entry is not None and entry.name == "Telegram"


def test_alias_to_executable(index: AppIndex) -> None:
    entry = index.find("блокнот")
    assert entry == AppEntry("блокнот", "path", "notepad.exe")


def test_alias_to_missing_app(index: AppIndex) -> None:
    assert index.find("стим") is None


def test_fuzzy_and_exact_search(index: AppIndex) -> None:
    assert index.find("калькулятор").kind == "start_app"
    assert index.find("хром").name == "Google Chrome"
    assert index.find("фотошоп") is None


def test_match_terms_include_alias_target(index: AppIndex) -> None:
    assert index.match_terms("телегу") == ["телегу", "Telegram"]
    assert index.match_terms("блокнот") == ["блокнот", "notepad"]
    assert index.match_terms("хром") == ["хром"]


def test_filter_entries_is_case_insensitive() -> None:
    entries = [AppEntry("README", "shortcut", "a"), AppEntry("Help and Support", "shortcut", "b"),
               AppEntry("Paint", "shortcut", "c")]  # fmt: skip
    assert [e.name for e in filter_entries(entries, ["readme", "help"])] == ["Paint"]
