"""Перечитывание команд при изменении YAML-файлов (hot-reload).

Папка опрашивается раз в interval секунд: сравниваются время изменения и размер
файлов *.yaml. Изменение засчитывается, когда файлы перестали меняться (редактор
мог записать файл в несколько приёмов).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

Snapshot = dict[str, tuple[int, int]]


def snapshot(directory: Path) -> Snapshot:
    """Время изменения и размер каждого файла команд."""
    state: Snapshot = {}
    if not directory.is_dir():
        return state
    for path in sorted(directory.glob("*.y*ml")):
        try:
            info = path.stat()
        except OSError:
            continue
        state[path.name] = (info.st_mtime_ns, info.st_size)
    return state


class CommandWatcher:
    """Следит за папкой команд и вызывает on_change после того, как файлы перестали меняться.

    Args:
        directory: папка с паками команд.
        on_change: что сделать (перечитать команды); вызывается в потоке наблюдателя.
        interval: как часто проверять, секунды.
    """

    def __init__(self, directory: Path, on_change: Callable[[], None], interval: float = 1.0) -> None:
        self._directory = directory
        self._on_change = on_change
        self._interval = interval
        self._known = snapshot(directory)
        self._pending: Snapshot | None = None
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def check(self) -> bool:
        """Одна проверка. True — изменения устоялись и on_change вызван."""
        current = snapshot(self._directory)
        if current == self._known:
            self._pending = None
            return False
        if current != self._pending:  # ещё меняется — подождём следующей проверки
            self._pending = current
            return False
        self._known, self._pending = current, None
        log.info("Файлы команд изменились — перечитываю")
        try:
            self._on_change()
        except Exception:
            log.exception("Не удалось перечитать команды")
        return True

    def start(self) -> None:
        """Запускает наблюдение в фоновом потоке."""
        self._thread = threading.Thread(target=self._run, name="commands-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stopped.wait(self._interval):
            self.check()

    def stop(self) -> None:
        """Останавливает наблюдение."""
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
