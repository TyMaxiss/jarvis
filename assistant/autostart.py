"""Автозапуск: ярлык ассистента в папке «Автозагрузка» (shell:startup) текущего пользователя.

Ярлык виден и самому пользователю (Диспетчер задач → Автозагрузка); включается
и выключается переключателем в настройках. Ассистент, запущенный ярлыком (--autostart),
сразу сворачивается в трей.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from assistant.actions.files import known_folder
from assistant.frozen import FROZEN, app_dir

log = logging.getLogger(__name__)

SHORTCUT = "Джарвис.lnk"
DESCRIPTION = "Голосовой ассистент Джарвис"
AUTOSTART_ARG = "--autostart"
_COM_TIMEOUT = 15.0


def launch_command(config_path: Path) -> tuple[Path, str, Path]:
    """Чем запустить ассистента при входе в Windows: программа, аргументы и рабочая папка.

    Из исходников — pythonw (без окна консоли) с -m assistant из папки проекта: пакет не
    устанавливается в окружение, его находит python в текущей папке.
    """
    config = f'--config "{config_path.resolve()}" {AUTOSTART_ARG}'
    if FROZEN:
        return Path(sys.executable).resolve(), config, app_dir()
    python = Path(sys.executable)
    pythonw = python.with_name("pythonw.exe")
    return (pythonw if pythonw.exists() else python), f"-m assistant {config}", app_dir()


class Autostart:
    """Ярлык автозапуска.

    Args:
        config_path: config.yaml, с которым запускать ассистента.
        folder: папка «Автозагрузка» (в тестах — временная).
    """

    def __init__(self, config_path: Path, folder: Path | None = None) -> None:
        self._config_path = config_path
        self._folder = folder

    @property
    def path(self) -> Path:
        """Где лежит ярлык."""
        return (self._folder or known_folder("startup")) / SHORTCUT

    def enabled(self) -> bool:
        """Ярлык есть."""
        return self.path.exists()

    def set(self, enabled: bool) -> None:
        """Создаёт или удаляет ярлык.

        Raises:
            OSError: ярлык не создать или не удалить.
        """
        path = self.path
        if not enabled:
            path.unlink(missing_ok=True)
            log.info("Автозапуск выключен: %s удалён", path)
            return
        if sys.platform != "win32":
            raise OSError("автозапуск работает только в Windows")
        program, arguments, workdir = launch_command(self._config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        in_com_thread(lambda: write_shortcut(path, program, arguments, workdir))
        log.info("Автозапуск включён: %s → %s %s", path, program, arguments)


def write_shortcut(path: Path, program: Path, arguments: str, workdir: Path) -> None:
    """Создаёт ярлык .lnk (IShellLink + IPersistFile; COM уже инициализирован)."""
    import comtypes.client
    from comtypes.persist import IPersistFile
    from comtypes.shelllink import IShellLinkW, ShellLink

    link = comtypes.client.CreateObject(ShellLink, interface=IShellLinkW)
    link.SetPath(str(program))
    link.SetArguments(arguments)
    link.SetWorkingDirectory(str(workdir))
    link.SetDescription(DESCRIPTION)
    link.QueryInterface(IPersistFile).Save(str(path), True)


def read_shortcut(path: Path) -> tuple[str, str, str]:
    """Программа, аргументы и рабочая папка ярлыка (COM уже инициализирован)."""
    import comtypes.client
    from comtypes.persist import IPersistFile
    from comtypes.shelllink import IShellLinkW, ShellLink

    link = comtypes.client.CreateObject(ShellLink, interface=IShellLinkW)
    link.QueryInterface(IPersistFile).Load(str(path), 0)
    return link.GetPath(0), link.GetArguments(), link.GetWorkingDirectory()


def in_com_thread[T](work: Callable[[], T]) -> T:
    """Выполняет work в своём потоке с COM: вызывающий поток может быть любым.

    Raises:
        OSError: work не удалась или не успела.
    """
    outcome: list[tuple[bool, Any]] = []

    def target() -> None:
        import comtypes

        comtypes.CoInitialize()
        try:
            outcome.append((True, work()))
        except Exception as exc:
            # Только текст: объекты COM из исключения не должны пережить CoUninitialize.
            log.warning("Ярлык: %s", exc, exc_info=True)
            outcome.append((False, OSError(str(exc) or type(exc).__name__)))
        finally:
            comtypes.CoUninitialize()

    thread = threading.Thread(target=target, name="shortcut", daemon=True)
    thread.start()
    thread.join(_COM_TIMEOUT)
    if not outcome:
        raise OSError("Windows не ответил вовремя")
    ok, value = outcome[0]
    if not ok:
        raise value
    return value
