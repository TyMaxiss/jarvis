"""JSONL-журнал действий: каждая строка — одно выполненное (или отклонённое) действие."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from assistant.logging_setup import SecretMasker

log = logging.getLogger(__name__)


class ActionJournal:
    """Потокобезопасная запись журнала действий.

    Args:
        path: файл журнала (папка создаётся).
        masker: маскировщик секретов — ключи не должны попасть и в журнал.
    """

    def __init__(self, path: Path, masker: SecretMasker) -> None:
        self.path = path
        self._masker = masker
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        *,
        source: str,
        action: str,
        args: dict[str, Any],
        status: str,
        result: str,
        duration_ms: float,
        command_id: str | None = None,
        phrase: str | None = None,
    ) -> None:
        """Добавляет запись. Ошибка записи логируется, но не прерывает работу."""
        entry = {
            "time": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "source": source,
            "command": command_id,
            "phrase": phrase,
            "action": action,
            "args": args,
            "status": status,
            "result": result,
            "duration_ms": round(duration_ms, 1),
        }
        line = self._masker.mask(json.dumps(entry, ensure_ascii=False, default=str))
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
        except OSError:
            log.exception("Не удалось записать журнал действий %s", self.path)
