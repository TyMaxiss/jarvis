"""Собранный exe (PyInstaller): где лежат файлы и как запускать внешние программы.

В сборке рядом с Jarvis.exe лежат config.yaml, commands/, models/ и logs/, а библиотеки —
в папке _internal (sys._MEIPASS). Загрузчик PyInstaller ищет DLL в _internal
(SetDllDirectoryW), vosk добавляет свою папку в PATH, и программа, которую запускает
ассистент, унаследовала бы и то и другое — и могла бы загрузить чужие DLL. Поэтому внешние
программы запускаются внутри external_launch (документация PyInstaller: «Launching external
programs from the frozen application»).
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

FROZEN = bool(getattr(sys, "frozen", False))
CONSOLE_EXE = "JarvisConsole.exe"  # та же сборка с окном консоли (Jarvis.exe — без него)
RESET_ENV = "PYINSTALLER_RESET_ENVIRONMENT"  # новый экземпляр exe — не дочерний процесс этого
# Поиск DLL и PATH — общие для всего процесса: запуски из разных потоков идут по очереди.
_LAUNCH_LOCK = threading.Lock()


def app_dir() -> Path:
    """Папка программы: где Jarvis.exe в сборке, корень проекта при запуске из исходников."""
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def run_hint(arguments: str) -> str:
    """Как запустить ассистента с аргументами: в сборке — JarvisConsole.exe, иначе через uv."""
    return f"{CONSOLE_EXE} {arguments}" if FROZEN else f"uv run python -m assistant {arguments}"


def clean_path(path: str, bundle: str) -> str:
    """PATH без папок сборки (bundle — _internal)."""
    prefix = os.path.normcase(os.path.abspath(bundle))
    return os.pathsep.join(
        part
        for part in path.split(os.pathsep)
        if part and not os.path.normcase(os.path.abspath(part)).startswith(prefix)
    )


@contextlib.contextmanager
def external_launch() -> Iterator[None]:
    """На время запуска внешней программы — обычные поиск DLL и PATH (вне сборки ничего не делает).

    Подмена общая для всего процесса, поэтому внутри — только создание процесса (Popen,
    os.startfile), а не ожидание его конца: параллельно ассистент может загружать свои DLL.
    """
    bundle = getattr(sys, "_MEIPASS", "")
    if not FROZEN or not bundle or sys.platform != "win32":
        yield
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.SetDllDirectoryW.argtypes = (ctypes.c_wchar_p,)
    with _LAUNCH_LOCK:
        original = os.environ.get("PATH", "")
        os.environ["PATH"] = clean_path(original, bundle)
        kernel32.SetDllDirectoryW(None)
        try:
            yield
        finally:
            kernel32.SetDllDirectoryW(bundle)
            os.environ["PATH"] = original


def run_program(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    capture_output: bool = False,
    check: bool = False,
    **popen: Any,
) -> subprocess.CompletedProcess[Any]:
    """Как subprocess.run, но процесс создаётся внутри external_launch, а ждём его уже вне.

    Raises:
        subprocess.TimeoutExpired: не завершился за timeout (процесс завершается).
        subprocess.CalledProcessError: check и код выхода не 0.
    """
    if capture_output:
        popen.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with external_launch():
        process = subprocess.Popen(args, **popen)
    with process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
