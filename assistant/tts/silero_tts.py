"""Голос Silero — нейросеть на PyTorch, офлайн. Синтез — в отдельном процессе (silero_worker).

Процесс — это сам ассистент с --silero-worker (в сборке — JarvisConsole.exe): он один раз
загружает модель и синтезирует фразы по запросам. Silero читает только кириллицу, поэтому
текст сначала проходит через spoken_text (числа и латиница — словами).

Модели Silero распространяются по лицензии CC BY-NC 4.0 — только для некоммерческого
использования.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any

import numpy as np

from assistant.config import AppConfig
from assistant.frozen import CONSOLE_EXE, FROZEN, RESET_ENV, app_dir, external_launch
from assistant.tts.base import TTSError
from assistant.tts.spoken import spoken_text

log = logging.getLogger(__name__)

SILERO_SPEAKERS = (
    ("aidar", "Айдар"),
    ("eugene", "Евгений"),
    ("baya", "Бая"),
    ("kseniya", "Ксения"),
    ("xenia", "Ксения (xenia)"),
)
SAMPLE_RATE = 48_000
WORKER_ARG = "--silero-worker"
LOG_FILE = "silero.log"  # в папке логов: что процесс синтеза пишет в stderr
START_TIMEOUT = 180.0  # torch и модель с медленного диска в первый раз — до минуты и больше
REQUEST_TIMEOUT = 60.0
_CREATE_NO_WINDOW = 0x08000000


def rate_word(rate: float) -> str:
    """Скорость речи для SSML Silero (<prosody rate>): x-slow … x-fast."""
    for limit, word in ((0.75, "x-slow"), (0.95, "slow"), (1.1, "medium"), (1.4, "fast")):
        if rate < limit:
            return word
    return "x-fast"


def silero_engine(config: AppConfig, speaker: str | None = None) -> SileroTTS:
    """Голос Silero по настройкам: PyTorch и модель уже скачаны (models.ensure_silero)."""
    from assistant.models import silero_runtime

    return SileroTTS(
        config.resolve(config.tts.silero_model),
        speaker or config.tts.silero_speaker,
        config.tts.rate,
        lib=silero_runtime()[1],
        log_file=config.resolve(config.logging.dir) / LOG_FILE,
    )


def worker_command() -> list[str]:
    """Как запустить процесс синтеза: в сборке — JarvisConsole.exe, из исходников — python -m assistant."""
    if FROZEN:
        return [str(app_dir() / CONSOLE_EXE), WORKER_ARG]
    return [sys.executable, "-m", "assistant", WORKER_ARG]


class SileroTTS:
    """Голос Silero.

    Args:
        model_path: файл модели (.pt — пакет torch.package).
        speaker: голос: aidar, eugene, baya, kseniya, xenia.
        rate: скорость речи (1.0 — обычная).
        lib: папка с PyTorch, скачанным для Silero; None — torch из окружения Python.
        log_file: куда процесс синтеза пишет свои ошибки (stderr); None — никуда.
        command: чем запускать процесс (тестам — подделка).
    """

    name = "silero"

    def __init__(
        self,
        model_path: Path,
        speaker: str,
        rate: float = 1.0,
        *,
        lib: Path | None = None,
        log_file: Path | None = None,
        command: list[str] | None = None,
    ) -> None:
        self._model = model_path
        self.speaker = speaker
        self._rate = rate
        self._lib = lib
        self._log_file = log_file
        self._command = command or worker_command()
        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()  # один запрос за раз: синтез и «Прослушать» из разных потоков
        self._closed = False

    def start(self) -> None:
        """Запускает процесс синтеза в фоне: модель загрузится, пока голос не нужен."""

        def run() -> None:
            try:
                with self._lock:
                    self._started()
            except TTSError as exc:
                log.warning("Silero не запустился: %s", exc)

        threading.Thread(target=run, name="silero-start", daemon=True).start()

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Звук фразы (float32, 48 кГц) выбранным голосом.

        Raises:
            TTSError: процесс синтеза не запустился, упал или не ответил вовремя.
        """
        return self.speak(text, self.speaker, self._rate)

    def speak(self, text: str, speaker: str, rate: float) -> tuple[np.ndarray, int]:
        """Звук фразы любым голосом модели (так работает «Прослушать» в настройках)."""
        spoken = spoken_text(text)
        if not any(char.isalpha() for char in spoken):  # «…», «!» — читать нечего
            return np.zeros(0, dtype=np.float32), SAMPLE_RATE
        request = {"text": spoken, "speaker": speaker, "rate": rate_word(rate), "sample_rate": SAMPLE_RATE}
        with self._lock:
            process = self._started()
            header, data = self._exchange(process, request, REQUEST_TIMEOUT, with_data=True)
        if not header.get("ok"):
            raise TTSError(f"Silero: {header.get('error', 'ошибка синтеза')}")
        return np.frombuffer(data, dtype="<f4").astype(np.float32), SAMPLE_RATE

    def close(self) -> None:
        """Останавливает процесс синтеза: он выходит, когда закрывается его stdin."""
        self._closed = True
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    # --- Процесс ---------------------------------------------------------------------

    def _started(self) -> subprocess.Popen[bytes]:
        """Работающий процесс синтеза; если его нет — запускает и ждёт загрузки модели."""
        if self._closed:
            raise TTSError("Silero остановлен")
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self._model.is_file():
            raise TTSError(f"нет модели Silero {self._model.name}")
        stderr: IO[bytes] | int = subprocess.DEVNULL
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            stderr = self._log_file.open("ab")
        try:
            with external_launch():
                env = {**os.environ, RESET_ENV: "1"} if FROZEN else None  # PATH уже без наших папок
                process = subprocess.Popen(
                    self._command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    cwd=app_dir(),
                    env=env,
                    creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
        except OSError as exc:
            raise TTSError(f"не удалось запустить процесс Silero: {exc}") from exc
        finally:
            if not isinstance(stderr, int):
                stderr.close()  # у процесса своя копия дескриптора
        setup = {"model": str(self._model), "lib": str(self._lib or ""), "threads": _threads()}
        header, _ = self._exchange(process, setup, START_TIMEOUT, with_data=False)
        if not header.get("ok"):
            process.kill()
            raise TTSError(f"Silero не загрузился: {header.get('error', 'нет ответа')}")
        log.info("Silero готов: torch %s, модель %s", header.get("torch"), self._model.name)
        self._process = process
        return process

    def _exchange(
        self, process: subprocess.Popen[bytes], message: dict[str, Any], timeout: float, *, with_data: bool
    ) -> tuple[dict[str, Any], bytes]:
        """Отправляет строку JSON и читает ответ; не успел за timeout — процесс завершается."""
        assert process.stdin is not None and process.stdout is not None
        watchdog = threading.Timer(timeout, process.kill)  # чтение из канала само не прервётся
        watchdog.start()
        try:
            process.stdin.write(json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                code = process.wait(timeout=5)
                raise TTSError(f"процесс Silero завершился (код {code})")
            header = json.loads(line)
            size = int(header.get("samples", 0)) * 4 if with_data and header.get("ok") else 0
            data = process.stdout.read(size) if size else b""
            if len(data) != size:
                raise TTSError("процесс Silero оборвал ответ")
            return header, data
        except (OSError, ValueError, subprocess.TimeoutExpired, TTSError) as exc:
            process.kill()  # после сбоя ответы могли разойтись с запросами — процесс начнётся заново
            self._process = None
            if isinstance(exc, TTSError):
                raise
            raise TTSError(f"процесс Silero не ответил: {exc}") from exc
        finally:
            watchdog.cancel()


def _threads() -> int:
    """Потоков для синтеза: физические ядра, но не больше 4 — синтез идёт параллельно с остальным."""
    import psutil

    return max(1, min(4, psutil.cpu_count(logical=False) or 1))
