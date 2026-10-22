"""Собранный exe (PyInstaller): где лежат файлы и как запускать внешние программы.

В сборке рядом с Jarvis.exe лежат config.yaml, commands/, models/ и logs/, а библиотеки —
в папке _internal (sys._MEIPASS). Загрузчик PyInstaller ищет DLL в _internal
(SetDllDirectoryW), vosk добавляет свою папку в PATH, и программа, которую запускает
ассистент, унаследовала бы и то и другое — и могла бы загрузить чужие DLL. Поэтому внешние
программы запускаются внутри external_launch (документация PyInstaller: «Launching external
programs from the frozen application»). Папку cuda с DLL для Whisper (она тоже в PATH) запущенные
программы не видят по той же причине.
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
UPDATED_ENV = "JARVIS_UPDATED"  # запуск установщиком после обновления (для лога)
CUDA_DIR = "cuda"  # DLL CUDA для Whisper на видеокарте — рядом с программой
SILERO_DIR = "silero"  # PyTorch для голоса Silero — рядом с программой
# Поиск DLL и PATH — общие для всего процесса: запуски из разных потоков идут по очереди, а пока
# голос ищет и загружает DLL CUDA (keep_dll_search), ждут. Повторный вход из того же потока можно.
_LAUNCH_LOCK = threading.RLock()


def app_dir() -> Path:
    """Папка программы: где Jarvis.exe в сборке, корень проекта при запуске из исходников."""
    if FROZEN:
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def cuda_dir() -> Path:
    """Папка с DLL CUDA рядом с программой: собранный exe скачивает их туда сам (models.ensure_cuda)."""
    return app_dir() / CUDA_DIR


def silero_dir() -> Path:
    """Папка с PyTorch для голоса Silero: скачивается, когда Silero выбирают (models.ensure_silero)."""
    return app_dir() / SILERO_DIR


def run_hint(arguments: str) -> str:
    """Как запустить ассистента с аргументами: в сборке — JarvisConsole.exe, иначе через uv."""
    return f"{CONSOLE_EXE} {arguments}" if FROZEN else f"uv run python -m assistant {arguments}"


def clean_path(path: str, *folders: str) -> str:
    """PATH без папок folders и их подпапок (папка сборки _internal, папка cuda)."""
    prefixes = [os.path.normcase(os.path.abspath(folder)) for folder in folders]

    def inside(part: str) -> bool:
        part = os.path.normcase(os.path.abspath(part))
        return any(part == prefix or part.startswith(prefix + os.sep) for prefix in prefixes)

    return os.pathsep.join(part for part in path.split(os.pathsep) if part and not inside(part))


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
        os.environ["PATH"] = clean_path(original, bundle, str(cuda_dir()))
        kernel32.SetDllDirectoryW(None)
        try:
            yield
        finally:
            kernel32.SetDllDirectoryW(bundle)
            os.environ["PATH"] = original


@contextlib.contextmanager
def keep_dll_search() -> Iterator[None]:
    """Пока выполняется блок, PATH и поиск DLL процесса не меняются: external_launch ждёт его конца.

    CTranslate2 и CUDA находят свои DLL через PATH (папка cuda). Если в это время ассистент
    запускает программу (PowerShell для списка приложений при старте), external_launch на миг
    убирает папку cuda из PATH и вернёт свой PATH без только что добавленной папки. Блок —
    короткий: проверка и загрузка DLL, а не вся загрузка модели (whisper_stt.missing_cuda_library).
    """
    with _LAUNCH_LOCK:
        yield


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
