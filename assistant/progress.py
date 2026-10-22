"""Ход долгих загрузок: скачивание моделей и библиотек при первом запуске.

Окна показывают его на главной строкой с процентами (у их «вывода» есть метод progress),
консоль и лог получают строку каждые 10%.
"""

from __future__ import annotations

import contextlib
import os
import stat
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_MB = 1 << 20
_GB = 1 << 30


def megabytes(size: int) -> str:
    """Размер для людей: «45 МБ», «1,3 ГБ»."""
    if size >= _GB:
        return f"{size / _GB:.1f} ГБ".replace(".", ",")
    return f"{round(size / _MB)} МБ"


class Progress:
    """Сколько скачано — и сообщение об этом.

    Окнам — каждый процент (не чаще interval секунд), консоли и логу — каждые 10%.
    Если размер неизвестен, пишется, сколько уже скачано: окнам — каждый мегабайт,
    консоли — каждые 100 МБ.

    Args:
        out: куда сообщать: консоль, лог или окна (у них есть метод progress(text, fraction)).
        title: что делается: «Скачиваю модель Vosk».
        total: сколько байт всего; 0 — неизвестно.
        done: сколько уже было (докачка).
        interval: как часто, самое частое, обновлять окна.
    """

    def __init__(self, out: Any, title: str, total: int = 0, done: int = 0, interval: float = 0.2) -> None:
        self.title = title
        self.total = total
        self.done = done
        self._out = out
        self._window: Callable[[str, float | None], None] | None = getattr(out, "progress", None)
        self._interval = interval
        self._shown: int | None = None  # последний шаг, показанный окнам
        self._shown_at = 0.0  # когда его показали (time.monotonic)
        self._written: int | None = None  # последний шаг, записанный строкой

    def advance(self, count: int) -> None:
        """Скачано ещё count байт."""
        self.update(self.done + count)

    def update(self, done: int) -> None:
        """Всего скачано done байт."""
        self.done = done
        fraction: float | None
        if self.total:
            percent = min(100, done * 100 // self.total)
            text = f"{self.title}: {percent}% ({megabytes(done)} из {megabytes(self.total)})"
            fraction = min(1.0, done / self.total)
            fine, coarse, last = percent, percent // 10, percent == 100
        else:
            text = f"{self.title}: {megabytes(done)}"
            fraction = None
            fine, coarse, last = done // _MB, done // (100 * _MB), False
        if self._window is None:
            if coarse != self._written:
                self._written = coarse
                self._out.write(f"  {text}\n")
                self._out.flush()
            return
        now = time.monotonic()
        due = last or self._shown is None or now - self._shown_at >= self._interval
        if fine != self._shown and due:
            self._shown, self._shown_at = fine, now
            self._window(text, fraction)


def folder_size(folder: Path) -> int:
    """Сколько байт в файлах папки и её подпапок (ссылки не считаются)."""
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            try:
                status = os.lstat(os.path.join(root, name))
            except OSError:  # файл переименовали или удалили, пока считали
                continue
            if stat.S_ISREG(status.st_mode):
                total += status.st_size
    return total


@contextlib.contextmanager
def watch_folder(folder: Path, progress: Progress, interval: float = 0.5) -> Iterator[None]:
    """Пока выполняется блок, сообщает, сколько байт в папке: так виден ход чужого скачивания."""
    stop = threading.Event()

    def run() -> None:
        while not stop.wait(interval):
            progress.update(folder_size(folder))

    thread = threading.Thread(target=run, name="download-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
