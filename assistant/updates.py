"""Обновления: новая версия — релиз на GitHub с установщиком.

Раз в сутки (и по кнопке в настройках) программа спрашивает GitHub о последнем релизе.
Если он новее, окна предлагают обновиться: установщик скачивается с проверкой sha256,
тихо ставится поверх этой же папки, а программа закрывается — установщик дождётся её
выхода и запустит новую версию. Настройки, ключи, свои команды и модели остаются.

Обновляется только собранная программа: из исходников — git pull и uv sync.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from assistant import __version__
from assistant.frozen import FROZEN, RESET_ENV, UPDATED_ENV, app_dir, external_launch
from assistant.models import download_file, file_sha256
from assistant.progress import Progress

log = logging.getLogger(__name__)

REPOSITORY = "TyMaxiss/jarvis"
LATEST_RELEASE = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases"
CHECK_INTERVAL = 24 * 3600  # секунд между автоматическими проверками
INSTALLER_RE = re.compile(r"^Jarvis-Setup-[\w.+-]+\.exe$")  # без слэшей: это имя файла в папке загрузок
CHECKSUMS = "SHA256SUMS.txt"
UPDATE_LOG = "update.log"  # журнал установщика — в папке логов программы
_VERSION_RE = re.compile(r"v?(\d+)\.(\d+)\.(\d+)(?:\.dev(\d+))?")
_TIMEOUT = 15
_DETACHED_PROCESS = 0x00000008


class UpdateError(Exception):
    """Проверить или поставить обновление не удалось; текст — для окон."""


@dataclass(frozen=True)
class Release:
    """Релиз на GitHub."""

    version: str
    title: str
    notes: str  # что нового (описание релиза)
    page: str  # страница релиза
    installer: str  # адрес установщика
    size: int
    sha256: str


def version_key(version: str) -> tuple[int, int, int, int, int]:
    """Для сравнения версий: «0.6.0.dev2» раньше «0.6.0», а «0.6.0» раньше «0.6.1».

    Raises:
        ValueError: строка не похожа на версию.
    """
    match = _VERSION_RE.fullmatch(version.strip())
    if match is None:
        raise ValueError(f"не похоже на версию: {version}")
    major, minor, patch, dev = match.groups()
    return int(major), int(minor), int(patch), 0 if dev is not None else 1, int(dev or 0)


def is_newer(candidate: str, current: str) -> bool:
    """candidate — более новая версия, чем current."""
    return version_key(candidate) > version_key(current)


def _get(url: str, accept: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": f"Jarvis/{__version__}",  # без него GitHub API отвечает 403
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return response.read()


def latest_release() -> Release:
    """Последний релиз программы на GitHub.

    Raises:
        UpdateError: нет сети, GitHub не ответил или в релизе нет установщика.
    """
    try:
        data = json.loads(_get(LATEST_RELEASE, "application/vnd.github+json"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("релизов не найдено (репозиторий закрыт или релизов ещё нет)") from exc
        if exc.code in (403, 429):
            raise UpdateError("GitHub временно ограничил запросы — попробую позже") from exc
        raise UpdateError(f"GitHub ответил ошибкой {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise UpdateError(f"не удалось связаться с GitHub: {exc}") from exc
    assets = {asset["name"]: asset for asset in data.get("assets") or []}
    installer = next((asset for name, asset in assets.items() if INSTALLER_RE.match(name)), None)
    if installer is None:
        raise UpdateError("в последнем релизе нет установщика")
    return Release(
        version=str(data.get("tag_name", "")).removeprefix("v"),
        title=data.get("name") or data.get("tag_name", ""),
        notes=(data.get("body") or "").strip(),
        page=data.get("html_url") or RELEASES_PAGE,
        installer=installer["browser_download_url"],
        size=int(installer.get("size") or 0),
        sha256=_installer_sha256(installer, assets.get(CHECKSUMS)),
    )


def _installer_sha256(installer: dict[str, Any], checksums: dict[str, Any] | None) -> str:
    """sha256 установщика: GitHub считает его сам (digest), а ещё он есть в SHA256SUMS.txt."""
    digest = str(installer.get("digest") or "")
    if digest.startswith("sha256:"):
        return digest.removeprefix("sha256:")
    if checksums is not None:
        try:
            text = _get(checksums["browser_download_url"], "application/octet-stream").decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise UpdateError(f"не удалось скачать {CHECKSUMS}: {exc}") from exc
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*") == installer["name"]:
                return parts[0].lower()
    raise UpdateError("у установщика нет контрольной суммы — ставить его небезопасно")


class Updater:
    """Проверка, скачивание и установка обновлений.

    Args:
        state_file: где помнить время последней проверки.
        current: версия этой программы.
        log_file: куда установщик пишет журнал установки (None — без журнала).
        fetch: как узнать последний релиз (тестам — подделка).
        clock: часы (тестам — свои).
    """

    def __init__(
        self,
        state_file: Path,
        current: str = __version__,
        *,
        log_file: Path | None = None,
        fetch: Callable[[], Release] = latest_release,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._state_file = state_file
        self.current = current
        self._log_file = log_file
        self._fetch = fetch
        self._clock = clock

    @property
    def supported(self) -> bool:
        """Обновляться может только собранная программа (из исходников — git pull)."""
        return FROZEN

    def due(self) -> bool:
        """Пора проверить снова: с прошлой проверки прошли сутки."""
        try:
            checked = float(json.loads(self._state_file.read_text(encoding="utf-8"))["checked"])
        except (OSError, ValueError, KeyError, TypeError):
            return True
        return self._clock() - checked >= CHECK_INTERVAL

    def check(self) -> Release | None:
        """Новый релиз, если он есть (None — у вас последняя версия).

        Raises:
            UpdateError: проверить не удалось.
        """
        release = self._fetch()
        try:
            self._state_file.write_text(json.dumps({"checked": self._clock()}), encoding="utf-8")
        except OSError:
            log.warning("Не удалось записать %s", self._state_file, exc_info=True)
        try:
            newer = is_newer(release.version, self.current)
        except ValueError as exc:
            raise UpdateError(f"непонятная версия релиза: {release.version}") from exc
        log.info("Обновления: последняя версия %s, у нас %s", release.version, self.current)
        return release if newer else None

    def download(self, release: Release, out: Any, folder: Path | None = None) -> Path:
        """Скачивает установщик (с докачкой и проверкой sha256), а прежние удаляет.

        Args:
            release: что скачать.
            out: куда сообщать о ходе загрузки (как у assistant.progress.Progress).
            folder: папка для установщиков; по умолчанию %TEMP%\\Jarvis.

        Raises:
            UpdateError: скачать не удалось.
        """
        folder = folder or Path(tempfile.gettempdir()) / "Jarvis"
        name = release.installer.rsplit("/", 1)[-1]
        if not INSTALLER_RE.match(name):
            raise UpdateError(f"непонятное имя установщика: {name}")
        if not release.sha256:
            raise UpdateError("у установщика нет контрольной суммы — ставить его небезопасно")
        target = folder / name
        _remove_old_installers(folder, keep=name)
        if target.is_file() and file_sha256(target) == release.sha256:
            return target  # уже скачан раньше
        out.write(f"Скачиваю версию {release.version}\n")
        progress = Progress(out, "Скачиваю обновление", total=release.size)
        try:
            download_file(release.installer, target, progress, sha256=release.sha256)
        except OSError as exc:
            raise UpdateError(f"не удалось скачать обновление: {exc}") from exc
        return target

    def install(self, installer: Path, folder: Path | None = None) -> None:
        """Запускает тихую установку поверх папки программы; программа после этого закрывается.

        Установщик ждёт, пока программа выйдет, ставит новую версию и запускает её.

        Raises:
            UpdateError: установщик не запустился.
        """
        arguments = [
            str(installer),
            "/VERYSILENT",
            "/SUPPRESSMSGBOXES",
            "/NORESTART",
            f"/DIR={folder or app_dir()}",
            "/RELAUNCH=yes",
        ]
        if self._log_file is not None:
            arguments.append(f"/LOG={self._log_file}")
        log.info("Запускаю установку: %s", " ".join(arguments))
        try:
            if self._log_file is not None:
                self._log_file.parent.mkdir(parents=True, exist_ok=True)
            with external_launch():
                # Новую версию запустит установщик — через Проводник, как с ярлыка (packaging/jarvis.iss),
                # а если Проводника нет, сам: тогда она должна быть самостоятельным экземпляром,
                # а не дочерним процессом этого exe (RESET_ENV), и получить PATH без папок, которые
                # добавил себе этот процесс (external_launch).
                env = {**os.environ, RESET_ENV: "1", UPDATED_ENV: "1"}
                subprocess.Popen(arguments, creationflags=_DETACHED_PROCESS, close_fds=True, env=env)
        except OSError as exc:
            raise UpdateError(f"не удалось запустить установщик: {exc}") from exc


def _remove_old_installers(folder: Path, keep: str) -> None:
    """Удаляет из папки загрузок установщики других версий (и их недокачанные .part)."""
    if not folder.is_dir():
        return
    for path in folder.iterdir():
        name = path.name.removesuffix(".part")
        if name != keep and INSTALLER_RE.match(name):
            try:
                path.unlink()
            except OSError:  # установщик ещё работает — удалится в следующий раз
                log.debug("Не удалось удалить %s", path, exc_info=True)


def release_view(release: Release | None) -> dict[str, Any] | None:
    """Релиз для окон (тип UpdateInfo в ui/src/lib/types.ts)."""
    return None if release is None else asdict(release)
