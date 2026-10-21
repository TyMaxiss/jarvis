"""Команды PowerShell.

run_shell всегда требует подтверждения, а опасные команды (форматирование и разметка
дисков, рекурсивное удаление, правка реестра и загрузчика…) запрещены политикой
безопасности (safety/policy.py) и не выполняются даже с подтверждением.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from typing import Annotated, Any

from pydantic import Field

from assistant.actions.registry import ActionError, ActionRegistry, Risk
from assistant.frozen import run_program

log = logging.getLogger(__name__)

TIMEOUT = 30  # секунд на команду
OUTPUT_LIMIT = 1500  # символов вывода для ответа (нейросеть читает его целиком)
_CREATE_NO_WINDOW = 0x08000000


def shorten(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Обрезает длинный вывод, сохраняя начало."""
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"… (ещё {len(text) - limit} символов)"


def powershell_args(command: str) -> list[str]:
    """Аргументы запуска: вывод в UTF-8, иначе кириллица придёт в кодировке консоли."""
    script = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; " + command
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script]


def register(registry: ActionRegistry, runner: Callable[..., Any] = run_program) -> None:
    """Регистрирует run_shell.

    Args:
        registry: реестр действий.
        runner: запуск процесса (как subprocess.run; подменяется в тестах).
    """

    @registry.action(
        "run_shell",
        "Выполнить команду PowerShell и получить её вывод. Всегда требует подтверждения пользователя",
        risk=Risk.CONFIRM,
    )
    def run_shell(
        command: Annotated[str, Field(min_length=1, max_length=2000, description="Команда PowerShell")],
    ) -> str:
        try:
            result = runner(
                powershell_args(command),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT,
                creationflags=_CREATE_NO_WINDOW,
            )
        except subprocess.TimeoutExpired as exc:
            raise ActionError(f"Команда не завершилась за {TIMEOUT} секунд") from exc
        except OSError as exc:
            raise ActionError(f"Не удалось запустить PowerShell: {exc}") from exc
        output = shorten(result.stdout or "")
        if result.returncode != 0:
            error = shorten(result.stderr or "") or output
            raise ActionError(f"Команда завершилась с ошибкой (код {result.returncode}): {error}")
        return output or "Команда выполнена, вывода нет"
