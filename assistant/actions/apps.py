"""Поиск, запуск и закрытие приложений.

При старте в фоне строится индекс: ярлыки (.lnk, .url) из меню «Пуск» и с рабочего стола
плюс приложения из PowerShell Get-StartApps (включая UWP, они запускаются через
explorer shell:AppsFolder\\<AppID>). Поверх индекса работают алиасы из конфига.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import psutil
from pydantic import Field
from rapidfuzz import fuzz

from assistant.actions import winapi
from assistant.actions.registry import ActionError, ActionRegistry
from assistant.actions.windows import window_score
from assistant.config import AppsConfig
from assistant.frozen import external_launch, run_program
from assistant.nlu.names import best_matches, name_score

log = logging.getLogger(__name__)

_CREATE_NO_WINDOW = 0x08000000
_SHORTCUT_SUFFIXES = (".lnk", ".url")
_PATH_SUFFIXES = (".exe", ".bat", ".cmd", ".msc", ".lnk", ".url")
_ALIAS_THRESHOLD = 80

# Один вызов PowerShell отдаёт и папки ярлыков (с учётом переноса рабочего стола
# в OneDrive), и список приложений меню «Пуск».
_POWERSHELL_SCRIPT = r"""
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$folders = @(
  [Environment]::GetFolderPath('CommonPrograms'),
  [Environment]::GetFolderPath('Programs'),
  [Environment]::GetFolderPath('CommonDesktopDirectory'),
  [Environment]::GetFolderPath('DesktopDirectory')
)
$apps = @()
try { $apps = @(Get-StartApps | ForEach-Object { @{ Name = $_.Name; AppID = $_.AppID } }) } catch {}
@{ folders = $folders; apps = $apps } | ConvertTo-Json -Compress -Depth 3
"""


@dataclass(frozen=True)
class AppEntry:
    """Приложение, которое можно запустить."""

    name: str
    kind: Literal["shortcut", "start_app", "path"]
    target: str

    def launch(self) -> None:
        """Запускает приложение.

        Raises:
            OSError: Windows не смог открыть ярлык или приложение.
        """
        with external_launch():
            if self.kind == "start_app":
                os.startfile(f"shell:AppsFolder\\{self.target}")  # type: ignore[attr-defined]
            else:
                os.startfile(self.target)  # type: ignore[attr-defined]


def _fallback_folders() -> list[Path]:
    """Папки ярлыков по переменным окружения (если PowerShell недоступен)."""
    folders = []
    for var, sub in (
        ("ProgramData", r"Microsoft\Windows\Start Menu\Programs"),
        ("APPDATA", r"Microsoft\Windows\Start Menu\Programs"),
        ("PUBLIC", "Desktop"),
        ("USERPROFILE", "Desktop"),
    ):
        if base := os.environ.get(var):
            folders.append(Path(base) / sub)
    return folders


def _query_powershell(timeout: float = 30.0) -> tuple[list[Path], list[AppEntry]]:
    """Папки ярлыков и приложения меню «Пуск» через PowerShell."""
    # -EncodedCommand (UTF-16LE в base64) избавляет от проблем с кавычками и переводами строк.
    encoded = base64.b64encode(_POWERSHELL_SCRIPT.encode("utf-16-le")).decode("ascii")
    try:
        completed = run_program(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-EncodedCommand", encoded],
            capture_output=True, timeout=timeout, creationflags=_CREATE_NO_WINDOW,
        )  # fmt: skip
        data = json.loads(completed.stdout.decode("utf-8-sig", errors="replace") or "{}")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        log.exception("Не удалось получить список приложений через PowerShell")
        return _fallback_folders(), []
    folders = [Path(p) for p in data.get("folders") or [] if p]
    apps = [
        AppEntry(str(item["Name"]), "start_app", str(item["AppID"]))
        for item in data.get("apps") or []
        if isinstance(item, dict) and item.get("Name") and item.get("AppID")
    ]
    return folders or _fallback_folders(), apps


def scan_shortcuts(folders: list[Path]) -> list[AppEntry]:
    """Ярлыки .lnk и .url из папок (рекурсивно)."""
    entries = []
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.suffix.lower() in _SHORTCUT_SUFFIXES and path.is_file():
                entries.append(AppEntry(path.stem, "shortcut", str(path)))
    return entries


def scan_installed_apps() -> list[AppEntry]:
    """Полный список приложений: сначала ярлыки, затем Get-StartApps (включая UWP)."""
    if not winapi.IS_WINDOWS:
        return []
    folders, start_apps = _query_powershell()
    return scan_shortcuts(folders) + start_apps


def filter_entries(entries: list[AppEntry], exclude: list[str]) -> list[AppEntry]:
    """Убирает деинсталляторы, справку и дубликаты (первое вхождение побеждает)."""
    stop_words = [word.lower() for word in exclude]
    seen: set[str] = set()
    result = []
    for entry in entries:
        key = entry.name.lower()
        if key in seen or any(word in key for word in stop_words):
            continue
        seen.add(key)
        result.append(entry)
    return result


def looks_like_path(target: str) -> bool:
    """Алиас указывает на файл, команду или URI, а не на название приложения."""
    lowered = target.lower()
    return any(ch in target for ch in "\\/:") or lowered.endswith(_PATH_SUFFIXES)


def _plain(text: str) -> str:
    return text.strip().lower().replace("ё", "е")


class AppIndex:
    """Индекс установленных приложений с нечётким поиском по названию.

    Args:
        config: настройки приложений (алиасы, порог, исключения).
        scanner: функция сканирования (подменяется в тестах).
    """

    def __init__(
        self, config: AppsConfig, scanner: Callable[[], list[AppEntry]] = scan_installed_apps
    ) -> None:
        self._config = config
        self._scanner = scanner
        self._entries: list[AppEntry] = []
        self._lock = threading.Lock()
        self._ready = threading.Event()

    def start(self) -> None:
        """Строит индекс в фоновом потоке, чтобы не задерживать старт."""
        threading.Thread(target=self.refresh, name="app-index", daemon=True).start()

    def refresh(self) -> None:
        """Пересканирует приложения."""
        started = time.perf_counter()
        try:
            entries = filter_entries(self._scanner(), self._config.exclude)
        except Exception:
            log.exception("Ошибка при построении индекса приложений")
            entries = []
        with self._lock:
            self._entries = entries
        self._ready.set()
        log.info("Индекс приложений: %d шт. за %.1f с", len(entries), time.perf_counter() - started)

    def entries(self, timeout: float = 15.0) -> list[AppEntry]:
        """Текущий список приложений; при первом обращении ждёт окончания сканирования."""
        self._ready.wait(timeout)
        with self._lock:
            return list(self._entries)

    def alias(self, query: str) -> str | None:
        """Цель алиаса для разговорного названия («телегу» → Telegram) или None."""
        best_score, best_target = 0.0, None
        for key, target in self._config.aliases.items():
            score = fuzz.ratio(_plain(query), _plain(key))
            if score > best_score:
                best_score, best_target = score, target
        return best_target if best_score >= _ALIAS_THRESHOLD else None

    def search(self, name: str) -> AppEntry | None:
        """Ищет приложение в индексе: сначала точное название, потом нечёткое."""
        entries = self.entries()
        for entry in entries:
            if _plain(entry.name) == _plain(name):
                return entry
        found = best_matches(
            name, entries, key=lambda e: e.name, threshold=self._config.match_threshold, limit=1
        )
        return found[0][0] if found else None

    def find(self, query: str) -> AppEntry | None:
        """Находит приложение по разговорному названию с учётом алиасов."""
        target = self.alias(query)
        if target is None:
            return self.search(query)
        entry = self.search(target)
        if entry is None and looks_like_path(target):
            entry = AppEntry(query, "path", target)
        return entry

    def match_terms(self, query: str) -> list[str]:
        """Варианты названия для поиска окон и процессов: запрос и цель алиаса."""
        terms = [query]
        target = self.alias(query)
        if target:
            terms.append(Path(target).stem if looks_like_path(target) else target)
        return terms


def _matching_processes(terms: list[str], threshold: float, exclude: set[int]) -> list[psutil.Process]:
    """Процессы, чьё имя подходит под одно из названий."""
    found = []
    for proc in psutil.process_iter(["name"]):
        name = (proc.info.get("name") or "").removesuffix(".exe")
        if proc.pid in exclude or not name:
            continue
        if max(name_score(term, name) for term in terms) >= threshold:
            found.append(proc)
    return found


def register(registry: ActionRegistry, index: AppIndex, config: AppsConfig) -> None:
    """Регистрирует действия с приложениями."""

    @registry.action("open_app", "Открыть (запустить) программу по названию")
    def open_app(
        name: Annotated[str, Field(min_length=1, description="Название программы, например Telegram")],
    ) -> str:
        entry = index.find(name)
        if entry is None:
            raise ActionError(f"Не нашёл приложение «{name}»")
        before = 0
        if winapi.IS_WINDOWS:
            before = winapi.api().GetForegroundWindow() or 0
        try:
            entry.launch()
        except OSError as exc:
            raise ActionError(f"Не удалось запустить «{entry.name}»") from exc
        if winapi.IS_WINDOWS and config.launch_wait > 0:
            winapi.wait_foreground_change(before, config.launch_wait)
        return f"Открываю {entry.name}"

    @registry.action(
        "close_app",
        "Закрыть программу по названию. force=true принудительно завершает процесс "
        "(несохранённые данные пропадут)",
    )
    def close_app(
        name: Annotated[str, Field(min_length=1, description="Название программы")],
        force: Annotated[bool, Field(description="Принудительно завершить процесс")] = False,
    ) -> str:
        terms = index.match_terms(name)
        threshold = config.match_threshold
        own = winapi.own_process_tree()

        if force:
            processes = _matching_processes(terms, threshold, own)
            if not processes:
                raise ActionError(f"Не нашёл запущенную программу «{name}»")
            for proc in processes:
                try:
                    proc.terminate()
                except psutil.Error:
                    log.warning("Не удалось завершить процесс %s", proc.pid)
            return f"Завершил «{name}»"

        windows = [w for w in winapi.list_windows() if w.pid not in own]
        # Сначала окна, чей процесс подходит по имени: так «телеграм» не закроет
        # браузер со вкладкой «Telegram Web».
        targets = [w for w in windows if max(name_score(t, w.process_stem) for t in terms) >= threshold]
        if not targets:
            scored = [(w, max(window_score(t, w) for t in terms)) for w in windows]
            scored = [pair for pair in scored if pair[1] >= threshold]
            if scored:
                targets = [max(scored, key=lambda pair: pair[1])[0]]
        if not targets:
            raise ActionError(f"Не нашёл открытое окно «{name}»")
        for window in targets:
            winapi.api().PostMessageW(window.hwnd, winapi.WM_CLOSE, 0, 0)
        return f"Закрываю «{name}»"
