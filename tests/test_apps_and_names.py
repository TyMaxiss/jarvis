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


# --- Индекс: русские названия, новые программы ----------------------------------------------


def powershell_answer(stdout: bytes) -> object:
    """Подмена run_program: PowerShell «отвечает» stdout; запоминает, какой скрипт ему дали."""
    import base64
    import subprocess

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        run.scripts.append(base64.b64decode(args[-1]).decode("utf-16-le"))  # type: ignore[attr-defined]
        return subprocess.CompletedProcess(args, 0, stdout, b"")

    run.scripts = []  # type: ignore[attr-defined]
    return run


def test_app_list_keeps_russian_names_and_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    # «Яндекс Музыку» ассистент не видел: русские названия и папки (C:\Users\Александр) приходили
    # из PowerShell искажёнными — его вывод не в UTF-8. Теперь ответ — JSON в base64.
    import base64
    import json
    from pathlib import Path

    from assistant.actions import apps

    answer = {
        "folders": [r"C:\Users\Александр\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"],
        "apps": [{"Name": "Яндекс Музыка", "AppID": "Yandex.Music_x!App"}, {"Name": "", "AppID": "x"}],
    }
    run = powershell_answer(base64.b64encode(json.dumps(answer, ensure_ascii=False).encode()) + b"\r\n")
    monkeypatch.setattr(apps, "run_program", run)
    for var in ("ProgramData", "PUBLIC", "USERPROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    folders, entries = apps._query_powershell()
    assert entries == [AppEntry("Яндекс Музыка", "start_app", "Yandex.Music_x!App")]
    # Папки по переменным окружения — всегда: если PowerShell ответил не то, ярлыки всё равно видны.
    assert folders == [
        Path(answer["folders"][0]),
        Path(str(tmp_path)) / r"Microsoft\Windows\Start Menu\Programs",
    ]
    assert "ToBase64String" in run.scripts[0] and "Get-StartApps" in run.scripts[0]  # type: ignore[attr-defined]


def test_garbled_app_list_falls_back_to_start_menu_folders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    from pathlib import Path

    from assistant.actions import apps

    monkeypatch.setattr(apps, "run_program", powershell_answer("Яндекс Музыка".encode("cp866")))
    for var in ("ProgramData", "PUBLIC", "USERPROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    folders, entries = apps._query_powershell()
    assert entries == [] and folders == [Path(str(tmp_path)) / r"Microsoft\Windows\Start Menu\Programs"]


def test_index_is_rebuilt_when_app_is_missing() -> None:
    # Программу установили, пока ассистент работал: не найдя её, он пересобирает список.
    clock = [0.0]
    installed = [AppEntry("Telegram", "shortcut", "t.lnk")]
    index = AppIndex(AppsConfig(), scanner=lambda: list(installed), clock=lambda: clock[0])
    index.refresh()
    installed.append(AppEntry("Яндекс Музыка", "shortcut", "y.lnk"))
    assert index.find("яндекс музыку") is None  # список только что собран — не пересобирается
    clock[0] = 61.0
    found = index.find("яндекс музыку")
    assert found is not None and found.name == "Яндекс Музыка"


def test_missing_app_error_names_similar_ones() -> None:
    from assistant.actions.apps import register
    from assistant.actions.registry import ActionError, ActionRegistry

    entries = [AppEntry("Яндекс Браузер", "shortcut", "b.lnk"), AppEntry("Telegram", "shortcut", "t.lnk")]
    index = AppIndex(AppsConfig(), scanner=lambda: entries)
    index.refresh()
    registry = ActionRegistry()
    register(registry, index, AppsConfig())
    action = registry.get("open_app")
    with pytest.raises(ActionError, match="Не нашёл приложение «Яндекс Музыку». Похожие: Яндекс Браузер$"):
        action.run(action.parse_args({"name": "Яндекс Музыку"}))
    with pytest.raises(ActionError, match="Не нашёл приложение «Фотошоп»$"):
        action.run(action.parse_args({"name": "Фотошоп"}))
