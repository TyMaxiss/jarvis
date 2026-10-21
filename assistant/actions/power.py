"""Блокировка, сон, выключение и перезагрузка компьютера.

Сон, выключение и перезагрузка требуют подтверждения. Выключение и перезагрузка
начинаются через несколько секунд — их ещё можно отменить командой shutdown /a.
"""

from __future__ import annotations

import ctypes
import logging
import subprocess

from assistant.actions import winapi
from assistant.actions.registry import ActionError, ActionRegistry, Risk

log = logging.getLogger(__name__)

SHUTDOWN_DELAY = 5  # секунд до выключения или перезагрузки


class PowerControl:
    """Обращения к Windows (в тестах подменяются)."""

    def lock(self) -> None:
        if not winapi.api().LockWorkStation():
            raise ActionError("Windows не заблокировал компьютер")

    def sleep(self) -> None:
        powrprof = ctypes.WinDLL("powrprof")  # type: ignore[attr-defined]
        powrprof.SetSuspendState.argtypes = (ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ubyte)
        powrprof.SetSuspendState.restype = ctypes.c_ubyte
        # Сон, а не гибернация. Если в системе включена гибернация, Windows может выбрать её.
        if not powrprof.SetSuspendState(0, 0, 0):
            raise ActionError("Windows не перевёл компьютер в сон")

    def shutdown(self, restart: bool) -> None:
        mode = "/r" if restart else "/s"
        try:
            subprocess.run(
                ["shutdown", mode, "/t", str(SHUTDOWN_DELAY)], check=True, capture_output=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ActionError(f"Не удалось {'перезагрузить' if restart else 'выключить'} компьютер") from exc


def register(registry: ActionRegistry, control: PowerControl | None = None) -> None:
    """Регистрирует действия управления питанием."""
    power = control or PowerControl()

    @registry.action("lock", "Заблокировать компьютер (экран блокировки)")
    def lock() -> str:
        power.lock()
        return "Компьютер заблокирован"

    @registry.action("sleep", "Перевести компьютер в спящий режим", risk=Risk.CONFIRM)
    def sleep() -> str:
        power.sleep()
        return "Перевожу компьютер в сон"

    @registry.action("shutdown", "Выключить компьютер", risk=Risk.CONFIRM)
    def shutdown() -> str:
        power.shutdown(restart=False)
        return f"Выключаю компьютер через {SHUTDOWN_DELAY} секунд"

    @registry.action("restart", "Перезагрузить компьютер", risk=Risk.CONFIRM)
    def restart() -> str:
        power.shutdown(restart=True)
        return f"Перезагружаю компьютер через {SHUTDOWN_DELAY} секунд"
