"""Один экземпляр ассистента в сеансе пользователя.

Два экземпляра слушали бы один микрофон и отвечали бы хором, а таймеры звенели бы дважды.
Запущенный экземпляр держит именованный мьютекс Windows; второй запуск через именованное
событие просит его показать окно и завершается. После «Перезапустить» новый экземпляр
ждёт, пока прежний закроется (переменная окружения RESTART_ENV).
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from typing import Any

log = logging.getLogger(__name__)

NAME = "JarvisVoiceAssistant"
RESTART_ENV = "JARVIS_RESTART"  # запуск после «Перезапустить»: прежний экземпляр ещё закрывается
RESTART_WAIT = 20.0  # сколько ждать его закрытия, секунды
WAIT_OBJECT_0 = 0x0
WAIT_ABANDONED = 0x80
INFINITE = 0xFFFFFFFF
EVENT_MODIFY_STATE = 0x0002


def _kernel32() -> Any:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    handle, boolean, dword = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
    kernel32.CreateMutexW.restype = handle
    kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, boolean, wintypes.LPCWSTR)
    kernel32.ReleaseMutex.restype = boolean
    kernel32.ReleaseMutex.argtypes = (handle,)
    kernel32.CreateEventW.restype = handle
    kernel32.CreateEventW.argtypes = (ctypes.c_void_p, boolean, boolean, wintypes.LPCWSTR)
    kernel32.OpenEventW.restype = handle
    kernel32.OpenEventW.argtypes = (dword, boolean, wintypes.LPCWSTR)
    kernel32.SetEvent.restype = boolean
    kernel32.SetEvent.argtypes = (handle,)
    kernel32.WaitForSingleObject.restype = dword
    kernel32.WaitForSingleObject.argtypes = (handle, dword)
    kernel32.CloseHandle.restype = boolean
    kernel32.CloseHandle.argtypes = (handle,)
    return kernel32


class SingleInstance:
    """Мьютекс «ассистент запущен» и событие «покажи окно».

    Вне Windows ничего не делает: там ассистент не запускается с микрофоном.

    Args:
        name: имя объектов Windows (тестам — своё, чтобы не мешать запущенному ассистенту).
    """

    def __init__(self, name: str = NAME) -> None:
        self._mutex_name = f"Local\\{name}"
        self._event_name = f"Local\\{name}-show"
        self._kernel32 = _kernel32() if sys.platform == "win32" else None
        self._mutex: int | None = None
        self._event: int | None = None
        self._stopped = threading.Event()

    def acquire(self, wait: float = 0.0) -> bool:
        """Делает этот экземпляр единственным; False — другой уже запущен (ждали wait секунд)."""
        kernel32 = self._kernel32
        if kernel32 is None or self._mutex is not None:
            return True
        mutex = kernel32.CreateMutexW(None, False, self._mutex_name)
        if not mutex:
            log.warning("Не удалось создать мьютекс %s: ошибка %d", self._mutex_name, ctypes.get_last_error())
            return True  # проверка не удалась — запуску это не мешает
        # Брошенный мьютекс (прежний экземпляр упал) тоже наш.
        if kernel32.WaitForSingleObject(mutex, round(wait * 1000)) in (WAIT_OBJECT_0, WAIT_ABANDONED):
            self._mutex = mutex
            return True
        kernel32.CloseHandle(mutex)
        return False

    def listen(self, on_show: Callable[[], None]) -> None:
        """В фоне ждёт, когда второй запуск попросит показать окно, и вызывает on_show."""
        kernel32 = self._kernel32
        if kernel32 is None or self._event is not None:
            return
        event = kernel32.CreateEventW(None, False, False, self._event_name)
        if not event:
            log.warning("Не удалось создать событие %s: ошибка %d", self._event_name, ctypes.get_last_error())
            return
        self._event = event

        def run() -> None:
            while (
                kernel32.WaitForSingleObject(event, INFINITE) == WAIT_OBJECT_0 and not self._stopped.is_set()
            ):
                log.info("Повторный запуск: показываю окно уже запущенного ассистента")
                try:
                    on_show()
                except Exception:
                    log.exception("Не удалось показать окно")

        threading.Thread(target=run, name="single-instance", daemon=True).start()

    def notify(self) -> bool:
        """Просит запущенный экземпляр показать окно; False — его событие не найдено."""
        kernel32 = self._kernel32
        if kernel32 is None:
            return False
        event = kernel32.OpenEventW(EVENT_MODIFY_STATE, False, self._event_name)
        if not event:
            return False
        try:
            return bool(kernel32.SetEvent(event))
        finally:
            kernel32.CloseHandle(event)

    def release(self) -> None:
        """Освобождает мьютекс (из того же потока, что его занял) и останавливает ожидание."""
        kernel32 = self._kernel32
        if kernel32 is None:
            return
        if self._mutex is not None:
            kernel32.ReleaseMutex(self._mutex)
            kernel32.CloseHandle(self._mutex)
            self._mutex = None
        if self._event is not None:
            # Поток ожидания просыпается и выходит; само событие закроется вместе с процессом.
            self._stopped.set()
            kernel32.SetEvent(self._event)
            self._event = None
