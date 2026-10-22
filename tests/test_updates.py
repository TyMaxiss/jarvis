"""Обновления: версии, ответ GitHub, скачивание установщика с проверкой sha256 и его запуск.

GitHub и раздачу файлов изображает локальный HTTP-сервер — интернет тестам не нужен.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import re
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterator
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from assistant import __version__, autostart, instance, updates
from assistant.frozen import RESET_ENV
from assistant.updates import (
    Release,
    UpdateError,
    Updater,
    is_newer,
    latest_release,
    release_view,
    version_key,
)
from tests.conftest import ROOT
from tests.test_downloads import Files, Window, payload

INSTALLER = payload(300_000)
INSTALLER_SHA = hashlib.sha256(INSTALLER).hexdigest()


@pytest.fixture
def server() -> Iterator[Files]:
    files = Files()
    yield files
    files.server.shutdown()


def make_release(server: Files | None = None, version: str = "0.7.0", sha256: str = INSTALLER_SHA) -> Release:
    name = f"Jarvis-Setup-{version}.exe"
    url = server.add(f"/download/v{version}/{name}", INSTALLER) if server else f"https://example.test/{name}"
    return Release(
        version=version,
        title=f"Джарвис {version}",
        notes="Новые голоса",
        page=f"https://github.com/TyMaxiss/jarvis/releases/tag/v{version}",
        installer=url,
        size=len(INSTALLER),
        sha256=sha256,
    )


# --- Версии ------------------------------------------------------------------------------


def test_version_order() -> None:
    order = ["0.5.9", "0.6.0.dev0", "0.6.0.dev2", "0.6.0", "v0.6.1", "0.10.0", "1.0.0"]
    keys = [version_key(version) for version in order]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert is_newer("0.6.0", "0.6.0.dev0") and not is_newer("0.6.0", "0.6.0")
    assert not is_newer("0.6.0.dev1", "0.6.0")
    for bad in ("", "0.6", "latest", "0.6.0-beta"):
        with pytest.raises(ValueError):
            version_key(bad)


def test_program_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == __version__
    version_key(__version__)  # понятна проверке обновлений


def test_changelog_describes_release_versions() -> None:
    # Описание релиза на GitHub берётся из CHANGELOG.md (задача release в CI).
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if ".dev" not in __version__:
        assert f"\n## {__version__} " in changelog


# --- Ответ GitHub ------------------------------------------------------------------------


def github_answer(assets: list[dict[str, Any]], tag: str = "v0.7.0") -> dict[str, Any]:
    return {
        "tag_name": tag,
        "name": f"Джарвис {tag[1:]}",
        "body": "## Что нового\n\nНовые голоса\n",
        "html_url": f"https://github.com/TyMaxiss/jarvis/releases/tag/{tag}",
        "assets": assets,
    }


def asset(server: Files, name: str, data: bytes, digest: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "size": len(data),
        "digest": digest,
        "browser_download_url": server.add(f"/download/v0.7.0/{name}", data),
    }


@pytest.fixture
def github(server: Files, monkeypatch: pytest.MonkeyPatch) -> Files:
    monkeypatch.setattr(updates, "LATEST_RELEASE", f"{server.url}/releases/latest")
    return server


def answer(server: Files, data: dict[str, Any]) -> None:
    server.add("/releases/latest", json.dumps(data, ensure_ascii=False).encode())


def test_latest_release_uses_github_digest(github: Files) -> None:
    answer(
        github,
        github_answer(
            [
                asset(github, "Jarvis-0.7.0-win-x64.zip", b"zip"),
                asset(github, "Jarvis-Setup-0.7.0.exe", INSTALLER, digest=f"sha256:{INSTALLER_SHA}"),
            ]
        ),
    )
    release = latest_release()
    assert release.version == "0.7.0" and release.title == "Джарвис 0.7.0"
    assert release.notes == "## Что нового\n\nНовые голоса"
    assert release.page.endswith("/releases/tag/v0.7.0")
    assert release.installer.endswith("/Jarvis-Setup-0.7.0.exe")
    assert release.size == len(INSTALLER) and release.sha256 == INSTALLER_SHA


def test_latest_release_takes_checksum_from_sha256sums(github: Files) -> None:
    sums = f"{'0' * 64}  Jarvis-0.7.0-win-x64.zip\n{INSTALLER_SHA.upper()} *Jarvis-Setup-0.7.0.exe\n"
    answer(
        github,
        github_answer(
            [
                asset(github, "Jarvis-Setup-0.7.0.exe", INSTALLER),
                asset(github, "SHA256SUMS.txt", sums.encode()),
            ]
        ),
    )
    assert latest_release().sha256 == INSTALLER_SHA


def test_release_without_installer_or_checksum_is_refused(github: Files) -> None:
    answer(github, github_answer([asset(github, "Jarvis-0.7.0-win-x64.zip", b"zip")]))
    with pytest.raises(UpdateError, match="нет установщика"):
        latest_release()
    answer(github, github_answer([asset(github, "Jarvis-Setup-0.7.0.exe", INSTALLER)]))
    with pytest.raises(UpdateError, match="нет контрольной суммы"):
        latest_release()


def test_installer_name_must_be_a_plain_file_name(github: Files) -> None:
    answer(
        github,
        github_answer([asset(github, "Jarvis-Setup-..\\..\\x.exe", INSTALLER, f"sha256:{INSTALLER_SHA}")]),
    )
    with pytest.raises(UpdateError, match="нет установщика"):
        latest_release()


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (404, "релизов не найдено"),
        (403, "ограничил запросы"),
        (429, "ограничил запросы"),
        (500, "ошибкой 500"),
    ],
)
def test_github_errors_are_readable(monkeypatch: pytest.MonkeyPatch, code: int, message: str) -> None:
    def refuse(url: str, accept: str) -> bytes:
        raise urllib.error.HTTPError(url, code, "ошибка", Message(), io.BytesIO())

    monkeypatch.setattr(updates, "_get", refuse)
    with pytest.raises(UpdateError, match=message):
        latest_release()


def test_no_network_is_readable(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(url: str, accept: str) -> bytes:
        raise urllib.error.URLError("нет сети")

    monkeypatch.setattr(updates, "_get", offline)
    with pytest.raises(UpdateError, match="не удалось связаться с GitHub"):
        latest_release()


def test_requests_introduce_the_program(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> Any:
        seen.append(request)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    updates._get(updates.LATEST_RELEASE, "application/vnd.github+json")
    headers = dict(seen[0].header_items())
    # Без User-Agent GitHub API отвечает 403.
    assert headers["User-agent"] == f"Jarvis/{__version__}"
    assert headers["Accept"] == "application/vnd.github+json"


# --- Проверка раз в сутки ------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def test_updater_checks_once_a_day(tmp_path: Path) -> None:
    clock = Clock()
    state = tmp_path / "update.json"
    updater = Updater(state, "0.6.0", fetch=lambda: make_release(version="0.7.0"), clock=clock)
    assert updater.due()
    release = updater.check()
    assert release is not None and release.version == "0.7.0"
    assert json.loads(state.read_text(encoding="utf-8")) == {"checked": clock.now}
    clock.now += 23 * 3600
    assert not updater.due()
    clock.now += 3600
    assert updater.due()
    state.write_text("не json", encoding="utf-8")
    assert updater.due()


def test_updater_returns_only_newer_release(tmp_path: Path) -> None:
    def updater(version: str) -> Updater:
        return Updater(tmp_path / "update.json", "0.6.0", fetch=lambda: make_release(version=version))

    assert updater("0.6.0").check() is None
    assert updater("0.5.0").check() is None
    assert updater("0.6.1").check() is not None
    with pytest.raises(UpdateError, match="непонятная версия"):
        updater("nightly").check()


def test_failed_check_is_repeated_next_time(tmp_path: Path) -> None:
    def offline() -> Release:
        raise UpdateError("нет сети")

    state = tmp_path / "update.json"
    updater = Updater(state, "0.6.0", fetch=offline)
    with pytest.raises(UpdateError):
        updater.check()
    assert not state.exists() and updater.due()


def test_only_the_built_program_installs_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    updater = Updater(tmp_path / "update.json")
    assert updater.current == __version__ and not updater.supported  # из исходников — git pull
    monkeypatch.setattr(updates, "FROZEN", True)
    assert updater.supported


# --- Скачивание и установка ------------------------------------------------------------------


def test_download_checks_sha256_and_reuses_file(server: Files, tmp_path: Path) -> None:
    release = make_release(server)
    window = Window()
    installer = Updater(tmp_path / "update.json").download(release, window, tmp_path / "setup")
    assert installer == tmp_path / "setup" / "Jarvis-Setup-0.7.0.exe"
    assert installer.read_bytes() == INSTALLER
    assert window.getvalue() == "Скачиваю версию 0.7.0\n"
    assert window.steps[-1] == ("Скачиваю обновление: 100% (0 МБ из 0 МБ)", 1.0)
    requests = len(server.requests)
    assert Updater(tmp_path / "update.json").download(release, Window(), tmp_path / "setup") == installer
    assert len(server.requests) == requests  # уже скачан и цел — не качается снова


def test_damaged_installer_is_downloaded_again(server: Files, tmp_path: Path) -> None:
    release = make_release(server)
    folder = tmp_path / "setup"
    folder.mkdir()
    (folder / "Jarvis-Setup-0.7.0.exe").write_bytes("испорчен".encode())
    installer = Updater(tmp_path / "update.json").download(release, Window(), folder)
    assert installer.read_bytes() == INSTALLER


def test_wrong_checksum_is_refused(server: Files, tmp_path: Path) -> None:
    release = make_release(server, sha256="0" * 64)
    with pytest.raises(UpdateError, match="не совпала контрольная сумма"):
        Updater(tmp_path / "update.json").download(release, Window(), tmp_path / "setup")
    assert list((tmp_path / "setup").iterdir()) == []


def test_download_refuses_strange_releases(tmp_path: Path) -> None:
    updater = Updater(tmp_path / "update.json")
    with pytest.raises(UpdateError, match="нет контрольной суммы"):
        updater.download(make_release(sha256=""), Window(), tmp_path)
    strange = Release("0.7.0", "", "", "", "https://example.test/..%5Cevil.exe", 1, "0" * 64)
    with pytest.raises(UpdateError, match="непонятное имя"):
        updater.download(strange, Window(), tmp_path)


def test_old_installers_are_removed(server: Files, tmp_path: Path) -> None:
    folder = tmp_path / "setup"
    folder.mkdir()
    for name in ("Jarvis-Setup-0.5.0.exe", "Jarvis-Setup-0.6.0.exe.part", "заметки.txt"):
        (folder / name).write_bytes(b"x")
    Updater(tmp_path / "update.json").download(make_release(server), Window(), folder)
    assert sorted(path.name for path in folder.iterdir()) == ["Jarvis-Setup-0.7.0.exe", "заметки.txt"]


def test_no_network_while_downloading(tmp_path: Path) -> None:
    release = dataclasses.replace(make_release(), installer="http://127.0.0.1:9/Jarvis-Setup-0.7.0.exe")
    with pytest.raises(UpdateError, match="не удалось скачать обновление"):
        Updater(tmp_path / "update.json").download(release, Window(), tmp_path)


class Launched:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.error: OSError | None = None

    def __call__(self, arguments: list[str], **kwargs: Any) -> object:
        if self.error is not None:
            raise self.error
        self.calls.append((arguments, kwargs))
        return object()


def test_install_runs_installer_silently(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    launched = Launched()
    monkeypatch.setattr(updates.subprocess, "Popen", launched)
    log_file = tmp_path / "logs" / "update.log"
    installer = tmp_path / "Jarvis-Setup-0.7.0.exe"
    Updater(tmp_path / "update.json", log_file=log_file).install(installer, tmp_path / "Jarvis")
    arguments, options = launched.calls[0]
    assert arguments == [
        str(installer),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        f"/DIR={tmp_path / 'Jarvis'}",
        "/RELAUNCH=yes",
        f"/LOG={log_file}",
    ]
    # Установщик живёт дольше программы, а новую версию запускает самостоятельным экземпляром.
    assert options["creationflags"] == 0x00000008 and options["env"][RESET_ENV] == "1"
    assert log_file.parent.is_dir()
    launched.error = OSError("нет доступа")
    with pytest.raises(UpdateError, match="не удалось запустить установщик"):
        Updater(tmp_path / "update.json").install(installer, tmp_path / "Jarvis")


@contextlib.contextmanager
def path_cleaned_to(path: str) -> Iterator[None]:
    """Вместо external_launch: на время блока PATH — path, потом прежний."""
    original = os.environ["PATH"]
    os.environ["PATH"] = path
    try:
        yield
    finally:
        os.environ["PATH"] = original


def test_installer_gets_path_without_program_folders(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Установщик запускает новую версию со своим окружением: папки CUDA, Vosk и WebView2,
    # которые этот процесс добавил себе в PATH, ей не нужны.
    launched = Launched()
    monkeypatch.setattr(updates.subprocess, "Popen", launched)
    monkeypatch.setattr(updates, "external_launch", lambda: path_cleaned_to("C:\\Windows"))
    monkeypatch.setenv("PATH", os.pathsep.join(["C:\\Jarvis\\cuda", "C:\\Windows"]))
    Updater(tmp_path / "update.json").install(tmp_path / "Jarvis-Setup-0.7.2.exe", tmp_path / "Jarvis")
    assert launched.calls[0][1]["env"]["PATH"] == "C:\\Windows"
    assert os.environ["PATH"] == os.pathsep.join(["C:\\Jarvis\\cuda", "C:\\Windows"])


def test_release_view_is_ready_for_windows() -> None:
    view = release_view(make_release())
    assert view is not None and json.loads(json.dumps(view, ensure_ascii=False)) == view
    assert set(view) == {"version", "title", "notes", "page", "installer", "size", "sha256"}
    assert release_view(None) is None


# --- Установщик ----------------------------------------------------------------------------


def test_installer_script_matches_program() -> None:
    script = (ROOT / "packaging" / "jarvis.iss").read_bytes()
    assert script.startswith(b"\xef\xbb\xbf")  # Inno Setup читает русский текст как UTF-8 только с BOM
    text = script.decode("utf-8-sig")
    sections = re.findall(r"^\[(\w+)\]$", text, re.M)
    assert sections == [
        "Setup",
        "Languages",
        "Tasks",
        "InstallDelete",
        "Files",
        "Icons",
        "Run",
        "UninstallDelete",
        "Code",
    ]
    # Установщик ждёт выхода программы по её мьютексу и убирает её ярлык автозапуска.
    assert f"AppMutex = 'Local\\{instance.NAME}';" in text
    assert f'Name: "{{userstartup}}\\{autostart.SHORTCUT}"' in text
    # После тихого обновления (Updater.install) программа запускается снова.
    assert "{param:RELAUNCH|no}" in text and "Check: Relaunch" in text
    assert re.search(r"^PrivilegesRequired=lowest$", text, re.M)  # ставится без прав администратора
    # CI проверяет запись в «Приложениях» по тому же AppId.
    app_id = re.search(r"^AppId=\{\{([0-9A-F-]{36})\}$", text, re.M)
    assert app_id is not None
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert f"{{{app_id[1]}}}_is1" in workflow


def test_pascal_comments_do_not_nest() -> None:
    # В Паскале комментарий { … } заканчивается на первой }: «{app}» внутри него сломал бы скрипт.
    code = (ROOT / "packaging" / "jarvis.iss").read_text(encoding="utf-8-sig").split("\n[Code]\n", 1)[1]
    code = re.sub(r"'[^'\n]*'", "''", code)  # строки в кавычках — не комментарии
    for comment in re.findall(r"\{[^}]*\}", code):
        assert "{" not in comment[1:], comment
