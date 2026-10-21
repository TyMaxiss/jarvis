"""Файлы и папки: создать папку, открыть её в проводнике, записать текстовый файл.

Путь — абсолютный или относительно домашней папки. Первым элементом можно назвать
известную папку по-русски или по-английски: «Рабочий стол/Проекты», «Загрузки».
Известные папки ищутся через SHGetKnownFolderPath — так находятся и папки,
перенесённые в OneDrive. Перезапись существующего файла требует подтверждения
(правило в safety/policy.py).
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from pydantic import Field

from assistant.actions.registry import ActionError, ActionRegistry

log = logging.getLogger(__name__)

# Идентификаторы известных папок Windows (KNOWNFOLDERID).
_KNOWN_FOLDER_IDS = {
    "desktop": "B4BFCC3A-DB2C-424C-B029-7FE99A87C641",
    "documents": "FDD39AD0-238F-46AF-ADB4-6C85480369C7",
    "downloads": "374DE290-123F-4565-9164-39C4925E467B",
    "pictures": "33E28130-4E1E-4676-835A-98395C3BC3BB",
    "music": "4BD8D571-6D19-48D3-BE97-422220080E43",
    "videos": "18989B1D-99B5-455B-841C-AB7C74E4DDFC",
    "startup": "B97D20BB-F46A-4C97-BA10-5E3608430854",  # «Автозагрузка» (shell:startup)
}
_FALLBACK_NAMES = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "startup": "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup",
}
FOLDER_ALIASES = {
    "рабочий стол": "desktop", "desktop": "desktop",
    "документы": "documents", "мои документы": "documents", "documents": "documents",
    "загрузки": "downloads", "downloads": "downloads",
    "изображения": "pictures", "картинки": "pictures", "pictures": "pictures",
    "музыка": "music", "music": "music",
    "видео": "videos", "videos": "videos",
    "домашняя папка": "home", "home": "home",
}  # fmt: skip
_TEXT_LIMIT = 1_000_000


class _GUID(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    )


def known_folder(key: str) -> Path:
    """Путь известной папки: desktop, documents, downloads, pictures, music, videos, startup или home."""
    home = Path.home()
    if key == "home":
        return home
    if os.name == "nt":
        from ctypes import wintypes

        guid = _GUID.from_buffer_copy(uuid.UUID(_KNOWN_FOLDER_IDS[key]).bytes_le)
        result = ctypes.c_wchar_p()
        shell32 = ctypes.WinDLL("shell32")  # type: ignore[attr-defined]
        ole32 = ctypes.WinDLL("ole32")  # type: ignore[attr-defined]
        shell32.SHGetKnownFolderPath.argtypes = (
            ctypes.POINTER(_GUID),
            wintypes.DWORD,
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_wchar_p),
        )
        shell32.SHGetKnownFolderPath.restype = ctypes.c_long
        ole32.CoTaskMemFree.argtypes = (ctypes.c_void_p,)
        ole32.CoTaskMemFree.restype = None
        try:
            if shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(result)) == 0:
                return Path(result.value or "")
        finally:
            ole32.CoTaskMemFree(ctypes.cast(result, ctypes.c_void_p))
        log.debug("SHGetKnownFolderPath не нашёл папку %s", key)
    return home / _FALLBACK_NAMES[key]


def resolve_user_path(text: str, folders: Callable[[str], Path] = known_folder) -> Path:
    """Превращает путь из фразы или от нейросети в абсолютный.

    «Рабочий стол/Проекты» → C:/Users/…/Desktop/Проекты, «~/x» и «%USERPROFILE%» раскрываются,
    остальные относительные пути считаются от домашней папки.

    Raises:
        ActionError: путь пустой.
    """
    cleaned = os.path.expandvars(text.strip().strip("\"'«»"))
    if not cleaned:
        raise ActionError("Не указан путь")
    if cleaned.startswith("~"):
        return Path(cleaned).expanduser()
    path = Path(cleaned)
    if path.is_absolute():
        return path
    relative = cleaned.replace("\\", "/")
    first, _, rest = relative.partition("/")
    key = FOLDER_ALIASES.get(re.sub(r"\s+", " ", first.lower().replace("ё", "е")))
    if key is not None:
        base = folders(key)
        return base / rest if rest else base
    return folders("home") / relative


_PATH_HELP = "Путь: абсолютный, от домашней папки или от известной папки, например «Рабочий стол/Проекты»"


def register(
    registry: ActionRegistry,
    opener: Callable[[str], None] | None = None,
    folders: Callable[[str], Path] = known_folder,
) -> None:
    """Регистрирует действия с файлами.

    Args:
        registry: реестр действий.
        opener: как открыть папку (по умолчанию os.startfile — в проводнике).
        folders: поиск известных папок (подменяется в тестах).
    """

    def open_in_explorer(path: str) -> None:
        if opener is not None:
            opener(path)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            raise ActionError("Открывать папки можно только в Windows")

    @registry.action("create_folder", "Создать папку (вместе с недостающими родительскими)")
    def create_folder(path: Annotated[str, Field(min_length=1, description=_PATH_HELP)]) -> str:
        folder = resolve_user_path(path, folders)
        if folder.is_dir():
            return f"Папка уже есть: {folder}"
        try:
            folder.mkdir(parents=True)
        except OSError as exc:
            raise ActionError(f"Не удалось создать папку {folder}: {exc.strerror or exc}") from exc
        return f"Создал папку {folder}"

    @registry.action("open_folder", "Открыть папку в проводнике")
    def open_folder(path: Annotated[str, Field(min_length=1, description=_PATH_HELP)]) -> str:
        folder = resolve_user_path(path, folders)
        if not folder.is_dir():
            raise ActionError(f"Папки {folder} нет")
        open_in_explorer(str(folder))
        return f"Открываю {folder}"

    @registry.action(
        "write_file",
        "Записать текст в файл в кодировке UTF-8. Существующий файл перезаписывается только с подтверждения",
    )
    def write_file(
        path: Annotated[str, Field(min_length=1, description=_PATH_HELP)],
        text: Annotated[str, Field(max_length=_TEXT_LIMIT, description="Содержимое файла")],
    ) -> str:
        file = resolve_user_path(path, folders)
        if file.is_dir():
            raise ActionError(f"{file} — это папка, а не файл")
        existed = file.exists()
        try:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ActionError(f"Не удалось записать {file}: {exc.strerror or exc}") from exc
        return f"{'Перезаписал' if existed else 'Создал'} файл {file}"
